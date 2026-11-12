"""
4.5 Multi-image insights ("connecting the dots").

Qwen2.5-VL compares the images directly (visual mode) or, for larger
groups, reasons over cached per-image summaries/tags (text mode).
Every insight is mapped back to the image ids that support it.
"""

import re
from typing import List, Optional, Tuple

from PIL import Image

from vas.core.vlm.base import VLMBackend, image_part, system_message, text_part, user_message
from vas.utils.image_utils import prepare_for_model
from vas.utils.parsing import extract_json
from vas.utils.safety import check_request

from .prompts import BASE_SYSTEM_PROMPT, MULTI_IMAGE_FOLLOWUP_PROMPT, MULTI_IMAGE_PROMPT, MULTI_IMAGE_TEXT_PROMPT

VALID_CONFIDENCE = {"high", "medium", "low"}


def _image_numbers(value) -> List[int]:
    """Accept [1, 3], ["Image 1", "3"], "1, 3" ... and return [1, 3]."""
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, str):
        return [int(n) for n in re.findall(r"\d+", value)]
    numbers = []
    for v in value or []:
        numbers.extend(_image_numbers(v))
    return numbers


def parse_insights(raw: str, image_ids: List[str]) -> dict:
    """
    Parse the model's JSON and attach supporting image ids to each insight.

    Image numbers outside 1..N are dropped (the model sometimes invents them).
    """
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        # Model ignored the format: keep the text as a single unverified observation
        return {"insights": [{"type": "other", "description": (raw or "").strip()[:1500],
                              "images": [], "image_ids": [], "confidence": "low"}],
                "overall_summary": "", "parse_ok": False}
    insights = []
    for item in parsed.get("insights") or []:
        if not isinstance(item, dict) or not item.get("description"):
            continue
        numbers = sorted({n for n in _image_numbers(item.get("images")) if 1 <= n <= len(image_ids)})
        confidence = str(item.get("confidence", "medium")).lower()
        insights.append({
            "type": str(item.get("type", "other")).split("|")[0].strip() or "other",
            "description": str(item["description"]).strip(),
            "images": numbers,
            "image_ids": [image_ids[n - 1] for n in numbers],
            "confidence": confidence if confidence in VALID_CONFIDENCE else "medium",
        })
    return {"insights": insights, "overall_summary": str(parsed.get("overall_summary", "")), "parse_ok": True}


def find_connections(backend: VLMBackend, images: List[Tuple[str, Image.Image]], max_pixels: int,
                     extra_instruction: str = "", text_descriptions: Optional[List[str]] = None) -> dict:
    """
    Compare a group of images and return insights.

    images: list of (image_id, PIL image) in display order (Image 1, Image 2, ...).
    text_descriptions: if given, use *text mode* (one description per image)
                       instead of sending all images — scales to bigger groups.
    """
    if len(images) < 2:
        raise ValueError("Select at least two images to find connections.")
    if extra_instruction and check_request(extra_instruction):
        raise ValueError(check_request(extra_instruction))
    image_ids = [img_id for img_id, _ in images]
    extra = f"Analyst focus: {extra_instruction.strip()}" if extra_instruction.strip() else ""
    n = len(images)

    if text_descriptions:
        descriptions = "\n".join(f"Image {i}: {d}" for i, d in enumerate(text_descriptions, start=1))
        prompt = MULTI_IMAGE_TEXT_PROMPT.format(n=n, descriptions=descriptions, extra=extra)
        messages = [system_message(BASE_SYSTEM_PROMPT), user_message(text_part(prompt))]
        mode = "text"
    else:
        parts = []
        for i, (_, image) in enumerate(images, start=1):
            parts += [text_part(f"Image {i}:"), image_part(prepare_for_model(image, max_pixels))]
        parts.append(text_part(MULTI_IMAGE_PROMPT.format(n=n, extra=extra)))
        messages = [system_message(BASE_SYSTEM_PROMPT), user_message(*parts)]
        mode = "visual"

    raw = backend.generate(messages, max_new_tokens=1500)
    result = parse_insights(raw, image_ids)
    result.update({"raw": raw, "mode": mode, "image_ids": image_ids})
    return result


def ask_group_question(backend: VLMBackend, images: List[Tuple[str, Image.Image]], question: str,
                       max_pixels: int) -> str:
    """Follow-up question about a group, e.g. 'Which images contain the white van?'."""
    question = (question or "").strip()
    if not question:
        raise ValueError("Please type a question.")
    policy = check_request(question)
    if policy:
        return policy
    parts = []
    for i, (_, image) in enumerate(images, start=1):
        parts += [text_part(f"Image {i}:"), image_part(prepare_for_model(image, max_pixels))]
    parts.append(text_part(MULTI_IMAGE_FOLLOWUP_PROMPT.format(n=len(images), question=question)))
    messages = [system_message(BASE_SYSTEM_PROMPT), user_message(*parts)]
    return backend.generate(messages).strip()
