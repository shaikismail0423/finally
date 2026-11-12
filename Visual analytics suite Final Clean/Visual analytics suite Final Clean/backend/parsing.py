"""Turn the model's text answers into Python data: JSON, boxes, tags, insights, document fields and tables."""

import json
import re

import pandas as pd

TAG_CATEGORIES = ["objects", "scene", "setting", "people", "vehicles", "clothing", "text_present"]
FIELD_KEYS = ["names", "dates", "amounts", "addresses", "phone_numbers", "reference_numbers", "emails"]


def extract_json(text: str):
    """
    Find the JSON object or list in a model answer and return it as Python data (None if there is none).

    Copes with ```json fences, sentences before/after the JSON and trailing commas like [1, 2,].
    """
    text = text or ""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return None
    start = min(starts)
    end = text.rfind("}" if text[start] == "{" else "]")
    chunk = text[start:end + 1]
    for candidate in (chunk, re.sub(r",\s*([\]}])", r"\1", chunk)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def normalize_tag(tag: str) -> str:
    """'White_Van ' -> 'white van', so the same tag is never listed twice."""
    return " ".join(re.sub(r"[_\-]+", " ", str(tag)).lower().split())


def unique_tags(tags) -> list:
    """Normalise tags, drop duplicates and empty values like 'none', keep the order."""
    result = []
    for tag in map(normalize_tag, tags):
        if tag and tag not in result and tag not in ("none", "n/a", "unknown", "null"):
            result.append(tag)
    return result


def parse_boxes(raw: str, sent_size: tuple, original_size: tuple, bbox_mode: str, default_label: str) -> list:
    """
    Read the model's boxes and convert them to pixels of the original image.

    Returns [{"label": str, "box": [x1, y1, x2, y2]}]. "absolute" coordinates are pixels of the image we
    sent; "relative" coordinates are 0-1000 of the image.
    """
    data = extract_json(raw)
    if isinstance(data, dict):  # e.g. {"objects": [...]} or a single box
        data = next((data[key] for key in ("objects", "detections", "boxes") if isinstance(data.get(key), list)), [data])
    if not isinstance(data, list):
        return []

    width, height = original_size
    if bbox_mode == "relative":
        scale_x, scale_y = width / 1000, height / 1000
    else:
        scale_x, scale_y = width / sent_size[0], height / sent_size[1]

    detections = []
    for item in data:
        coords = (item.get("bbox_2d") or item.get("bbox") or item.get("box")) if isinstance(item, dict) else None
        if not isinstance(coords, list) or len(coords) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(c) for c in coords)
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((x1 * scale_x, x2 * scale_x))
        y1, y2 = sorted((y1 * scale_y, y2 * scale_y))
        x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(width, x2), min(height, y2)  # keep inside the image
        box = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
        if x2 - x1 < 2 or y2 - y1 < 2 or any(d["box"] == box for d in detections):
            continue  # too small, off the image, or a repeat
        detections.append({"label": str(item.get("label") or default_label).strip(), "box": box})
    return detections


def parse_tags(raw: str, focus_tags: list) -> dict:
    """Returns {"tags": [...], "categories": {category: [...]}, "focus_hits": [{"item", "evidence"}]}."""
    data = extract_json(raw)
    if not isinstance(data, dict):
        raise ValueError("The model's tag answer could not be read — please try again.")

    categories = {}
    for key in TAG_CATEGORIES:
        values = data.get(key) or []
        if isinstance(values, str):
            values = values.split(",")
        if not isinstance(values, list):
            values = [values]
        tags = unique_tags(v for v in values if not isinstance(v, (dict, list)))
        if tags:
            categories[key] = tags
    time_of_day = normalize_tag(data.get("time_of_day") or "")
    if time_of_day and time_of_day != "unknown":
        categories["time_of_day"] = [time_of_day]

    # The model may say "license plates" for the focus item "license plate", so match by containment
    wanted = [normalize_tag(t) for t in focus_tags]
    focus_hits = []
    for entry in data.get("focus_items") or []:
        if not isinstance(entry, dict):
            continue
        item = normalize_tag(entry.get("item", ""))
        present = entry.get("present")
        if isinstance(present, str):
            present = present.strip().lower() in ("true", "yes")
        match = next((f for f in wanted if item and (f in item or item in f)), None)
        if present and match and all(hit["item"] != match for hit in focus_hits):
            focus_hits.append({"item": match, "evidence": str(entry.get("evidence", ""))})

    tags = unique_tags([t for values in categories.values() for t in values] + [hit["item"] for hit in focus_hits])
    return {"tags": tags, "categories": categories, "focus_hits": focus_hits}


def parse_insights(raw: str, image_ids: list) -> dict:
    """Read Connect-the-Dots insights and replace image numbers (1..N) with image ids."""
    data = extract_json(raw)
    if not isinstance(data, dict):  # the model ignored the format: keep its text as one observation
        return {"insights": [{"type": "other", "description": (raw or "").strip()[:1500], "image_ids": [],
                              "confidence": "low"}], "overall_summary": ""}
    insights = []
    for item in data.get("insights") or []:
        if not isinstance(item, dict) or not item.get("description"):
            continue
        # Accepts [1, 3], ["Image 1", "3"] or "1 and 3"; drops numbers the model invented
        numbers = sorted({int(n) for n in re.findall(r"\d+", str(item.get("images", "")))
                          if 1 <= int(n) <= len(image_ids)})
        confidence = str(item.get("confidence", "medium")).lower()
        insights.append({
            "type": str(item.get("type") or "other").split("|")[0].strip() or "other",
            "description": str(item["description"]).strip(),
            "image_ids": [image_ids[n - 1] for n in numbers],
            "confidence": confidence if confidence in ("high", "medium", "low") else "medium",
        })
    return {"insights": insights, "overall_summary": str(data.get("overall_summary") or "")}


def parse_fields(raw: str) -> dict:
    """Read the document key-details answer; always returns every field (empty lists when missing)."""
    data = extract_json(raw)
    data = data if isinstance(data, dict) else {}
    found = data.get("fields") if isinstance(data.get("fields"), dict) else data
    fields = {}
    for key in FIELD_KEYS:
        values = found.get(key) or []
        if not isinstance(values, list):
            values = [values]
        fields[key] = [str(v).strip() for v in values if str(v).strip()]
    issues = data.get("quality_issues") or []
    if not isinstance(issues, list):
        issues = [issues]
    return {"document_type": str(data.get("document_type") or "unknown"), "fields": fields,
            "quality_issues": [str(issue) for issue in issues]}


def markdown_tables(markdown: str) -> list:
    """Convert every Markdown table (| a | b |) in the text into a pandas DataFrame."""
    tables, rows = [], []
    for line in (markdown or "").splitlines() + [""]:
        if line.strip().startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", cell) or cell == "" for cell in cells):  # skip |---|---|
                rows.append(cells)
            continue
        if len(rows) >= 2:
            header = [name or f"col{i + 1}" for i, name in enumerate(rows[0])]
            header = [name if header.count(name) == 1 else f"{name}_{i}" for i, name in enumerate(header)]
            body = [(row + [""] * len(header))[:len(header)] for row in rows[1:]]
            tables.append(pd.DataFrame(body, columns=header))
        rows = []
    return tables
