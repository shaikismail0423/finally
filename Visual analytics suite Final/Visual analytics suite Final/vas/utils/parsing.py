"""
Robust parsing of model output.

VLMs often wrap JSON in markdown fences, add a sentence before it, or leave a
trailing comma. These helpers recover the structured data whenever possible.
"""

import json
import re
from typing import Any, List, Optional

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _try_json(text: str) -> Optional[Any]:
    """Attempt json.loads, also after removing trailing commas."""
    for candidate in (text, re.sub(r",\s*([\]}])", r"\1", text)):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _balanced_chunk(text: str, open_char: str, close_char: str) -> Optional[str]:
    """Return the first balanced {...} or [...] substring of `text`."""
    start = text.find(open_char)
    while start != -1:
        depth, in_string, escape = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find(open_char, start + 1)
    return None


def extract_json(text: str) -> Optional[Any]:
    """
    Extract the first JSON object/array from free-form model text.

    Returns the parsed Python object, or None if nothing parseable is found.
    """
    if not text:
        return None
    # 1) Fenced code blocks first (```json ... ```)
    for block in _FENCE_RE.findall(text):
        parsed = _try_json(block.strip())
        if parsed is not None:
            return parsed
    # 2) Whole text
    parsed = _try_json(text.strip())
    if parsed is not None:
        return parsed
    # 3) First balanced object or array, whichever appears first
    candidates = []
    for open_c, close_c in (("{", "}"), ("[", "]")):
        chunk = _balanced_chunk(text, open_c, close_c)
        if chunk:
            candidates.append((text.find(chunk), chunk))
    for _, chunk in sorted(candidates):
        parsed = _try_json(chunk)
        if parsed is not None:
            return parsed
    return None


def normalize_tag(tag: str) -> str:
    """Lower-case, trim and collapse whitespace so tags compare cleanly."""
    tag = re.sub(r"[_\-]+", " ", str(tag)).strip().lower()
    return re.sub(r"\s+", " ", tag)


def unique_tags(tags: List[str]) -> List[str]:
    """Normalise and de-duplicate a tag list, preserving order."""
    seen, result = set(), []
    for tag in tags:
        norm = normalize_tag(tag)
        if norm and norm not in seen and norm not in {"none", "n/a", "unknown", "null"}:
            seen.add(norm)
            result.append(norm)
    return result
