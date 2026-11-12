"""4.3 Automatic image tagging, including a case-specific focus list."""

import re
from typing import List

from PIL import Image

from vas.core.vlm.base import VLMBackend, image_part, system_message, text_part, user_message
from vas.utils.image_utils import prepare_for_model
from vas.utils.parsing import extract_json, normalize_tag, unique_tags

from .prompts import BASE_SYSTEM_PROMPT, TAGGING_PROMPT

TAG_CATEGORIES = ["objects", "scene", "setting", "people", "vehicles", "clothing", "text_present"]


def _as_list(value) -> List[str]:
    """Model sometimes returns a string where a list was expected."""
    if value is None:
        return []
    if isinstance(value, str):
        return [v for v in re.split(r"[,;]", value) if v.strip()]
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and not isinstance(v, (dict, list))]
    return [str(value)]


def parse_tag_response(raw: str, focus_tags: List[str]) -> dict:
    """
    Turn the model's JSON into a flat tag list plus focus-item hits.

    Returns {"tags": [...], "categories": {...}, "focus_hits": [{"item", "evidence"}]}.
    Falls back to comma-separated parsing if the model did not return JSON.
    """
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        fallback = unique_tags(re.split(r"[,\n;]", re.sub(r"[\[\]{}\"`*]", "", raw or "")))
        return {"tags": [t for t in fallback if len(t) <= 40][:25], "categories": {}, "focus_hits": []}

    categories = {cat: unique_tags(_as_list(parsed.get(cat))) for cat in TAG_CATEGORIES}
    time_of_day = normalize_tag(parsed.get("time_of_day") or "")
    if time_of_day and time_of_day != "unknown":
        categories["time_of_day"] = [time_of_day]

    focus_lookup = {normalize_tag(t) for t in focus_tags}
    focus_hits = []
    for entry in parsed.get("focus_items") or []:
        if not isinstance(entry, dict):
            continue
        item = normalize_tag(entry.get("item", ""))
        present = entry.get("present")
        if isinstance(present, str):
            present = present.strip().lower() in ("true", "yes", "1")
        # Map "license plates" -> focus tag "license plate" (simple containment match)
        matched = next((f for f in focus_lookup if f in item or item in f), None)
        if item and present and matched and all(h["item"] != matched for h in focus_hits):
            focus_hits.append({"item": matched, "evidence": str(entry.get("evidence", ""))})

    tags = unique_tags([t for values in categories.values() for t in values] + [h["item"] for h in focus_hits])
    return {"tags": tags, "categories": categories, "focus_hits": focus_hits}


def tag_image(backend: VLMBackend, image: Image.Image, focus_tags: List[str], max_pixels: int) -> dict:
    """Run the tagging prompt on one image and return parsed tags (+ raw text)."""
    focus_list = ", ".join(focus_tags) if focus_tags else "none"
    model_image = prepare_for_model(image, max_pixels)
    messages = [system_message(BASE_SYSTEM_PROMPT),
                user_message(image_part(model_image), text_part(TAGGING_PROMPT.format(focus_list=focus_list)))]
    raw = backend.generate(messages)
    result = parse_tag_response(raw, focus_tags)
    result["raw"] = raw
    return result
