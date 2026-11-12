"""
4.6 Document layout understanding and OCR.

Pipeline for one document image:
  1. Qwen2.5-VL transcribes the page to Markdown (headings, tables, fields).
  2. Qwen2.5-VL extracts key fields as JSON (names, dates, amounts, ...).
  3. A deterministic regex pass re-extracts phones/emails/dates/amounts from
     the transcription as an independent cross-check.
  4. Markdown tables are converted to pandas DataFrames for CSV export.
  5. Quality issues (model-reported + classical CV checks) are surfaced.
"""

import re
from typing import Dict, List

import pandas as pd
from PIL import Image

from vas.core.vlm.base import VLMBackend, image_part, system_message, text_part, user_message
from vas.utils.image_utils import prepare_for_model
from vas.utils.parsing import extract_json

from .prompts import BASE_SYSTEM_PROMPT, OCR_FIELDS_PROMPT, OCR_MARKDOWN_PROMPT

FIELD_KEYS = ["names", "dates", "amounts", "addresses", "phone_numbers", "reference_numbers", "emails"]

# Deterministic patterns (intentionally conservative)
REGEX_PATTERNS: Dict[str, str] = {
    "emails": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone_numbers": r"(?<![\w/-])(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,5}\)?[\s.-])?\d{3,5}[\s.-]?\d{3,5}(?![\w/-])",
    "dates": (r"\b(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
              r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4}|"
              r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b"),
    "amounts": r"(?:[$€£₹]\s?\d[\d,]*(?:\.\d{2})?|\b\d[\d,]*\.\d{2}\s?(?:USD|EUR|GBP|INR|Rs\.?)\b|\b(?:Rs\.?|INR)\s?\d[\d,]*(?:\.\d{2})?)",
}


def regex_extract(text: str) -> Dict[str, List[str]]:
    """Find emails, phone numbers, dates and amounts with regular expressions."""
    results = {}
    dates = set(re.findall(REGEX_PATTERNS["dates"], text or "", flags=re.IGNORECASE))
    for key, pattern in REGEX_PATTERNS.items():
        found = re.findall(pattern, text or "", flags=re.IGNORECASE)
        cleaned = []
        for value in found:
            value = value.strip(" .,;")
            # A date like 12-03-2024 also looks like a phone number; keep it as a date only
            if key == "phone_numbers" and (value in dates or sum(c.isdigit() for c in value) < 7):
                continue
            if value and value not in cleaned:
                cleaned.append(value)
        results[key] = cleaned
    return results


def parse_markdown_tables(markdown: str) -> List[pd.DataFrame]:
    """Convert every Markdown table in the text into a DataFrame."""
    tables, block = [], []
    for line in (markdown or "").splitlines() + [""]:
        if line.strip().startswith("|"):
            block.append(line.strip())
            continue
        if len(block) >= 2:
            rows = [[c.strip() for c in row.strip("|").split("|")] for row in block]
            rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) or c == "" for c in r)]
            if rows:
                header, body = rows[0], rows[1:]
                width = len(header)
                body = [(r + [""] * width)[:width] for r in body]
                # Make header names unique so pandas does not complain
                header = [h or f"col{i + 1}" for i, h in enumerate(header)]
                header = [h if header.count(h) == 1 else f"{h}_{i}" for i, h in enumerate(header)]
                tables.append(pd.DataFrame(body, columns=header))
        block = []
    return tables


def parse_fields_response(raw: str) -> dict:
    """Parse the key-field JSON; always returns the full structure."""
    parsed = extract_json(raw)
    parsed = parsed if isinstance(parsed, dict) else {}
    raw_fields = parsed.get("fields") if isinstance(parsed.get("fields"), dict) else parsed
    fields = {}
    for key in FIELD_KEYS:
        value = raw_fields.get(key, []) if isinstance(raw_fields, dict) else []
        if isinstance(value, str):
            value = [value] if value.strip() else []
        fields[key] = [str(v).strip() for v in value if str(v).strip()] if isinstance(value, list) else []
    issues = parsed.get("quality_issues") or []
    return {"document_type": str(parsed.get("document_type", "unknown")), "fields": fields,
            "quality_issues": [str(i) for i in issues] if isinstance(issues, list) else [str(issues)],
            "parse_ok": bool(parsed)}


def fields_table(model_fields: Dict[str, List[str]], regex_fields: Dict[str, List[str]]) -> pd.DataFrame:
    """Merge model and regex results into one table showing which source found each value."""
    rows = []
    for key in FIELD_KEYS:
        model_values = model_fields.get(key, [])
        regex_values = regex_fields.get(key, [])
        norm = lambda v: re.sub(r"\W", "", v.lower())  # noqa: E731 - tiny local helper
        regex_norm = {norm(v) for v in regex_values}
        for value in model_values:
            rows.append({"field": key, "value": value,
                         "source": "model + regex" if norm(value) in regex_norm else "model"})
        model_norm = {norm(v) for v in model_values}
        for value in regex_values:
            if norm(value) not in model_norm:
                rows.append({"field": key, "value": value, "source": "regex only"})
    return pd.DataFrame(rows, columns=["field", "value", "source"])


def extract_document(backend: VLMBackend, image: Image.Image, max_pixels: int, extract_fields: bool = True) -> dict:
    """Run the full document pipeline on one image."""
    model_image = prepare_for_model(image, max_pixels)
    messages = [system_message(BASE_SYSTEM_PROMPT), user_message(image_part(model_image), text_part(OCR_MARKDOWN_PROMPT))]
    markdown = backend.generate(messages, max_new_tokens=2048).strip()
    # Strip a wrapping ```markdown fence if the model added one
    markdown = re.sub(r"^```(?:markdown|md)?\s*|\s*```$", "", markdown).strip()

    fields_result = {"document_type": "unknown", "fields": {k: [] for k in FIELD_KEYS}, "quality_issues": [], "parse_ok": False}
    if extract_fields:
        messages = [system_message(BASE_SYSTEM_PROMPT), user_message(image_part(model_image), text_part(OCR_FIELDS_PROMPT))]
        fields_result = parse_fields_response(backend.generate(messages, max_new_tokens=800))

    regex_fields = regex_extract(markdown)
    illegible_count = markdown.lower().count("[illegible]")
    issues = list(fields_result["quality_issues"])
    if illegible_count:
        issues.append(f"{illegible_count} word(s) marked [illegible]")
    return {
        "markdown": markdown,
        "document_type": fields_result["document_type"],
        "fields": fields_result["fields"],
        "regex_fields": regex_fields,
        "quality_issues": issues,
        "table_count": len(parse_markdown_tables(markdown)),
    }
