"""The six capabilities. Each one builds a prompt, asks the model and returns plain Python data."""

import re

from PIL import Image

import config
from backend import prompts
from backend.model_client import chat
from backend.parsing import parse_boxes, parse_fields, parse_insights, parse_tags

SUMMARY_MAX_TOKENS = {"One-line caption": 80, "Short paragraph": 300, "Detailed description": 900}


def ask(image: Image.Image, question: str, history: list) -> str:
    """Answer a question about one image; `history` holds the earlier questions and answers."""
    question = question.strip()
    if not question:
        raise ValueError("Please type a question.")
    return chat(question, [image], history=history, system=prompts.QA_SYSTEM_PROMPT).strip()


def locate(image: Image.Image, target: str) -> dict:
    """Find every `target` in the image. Returns {"target", "detections", "raw"} with boxes in original pixels."""
    target = target.strip().rstrip(".?")
    # "Highlight all vehicles" -> "vehicles"
    target = re.sub(r"^(please\s+)?(highlight|find|mark|show|locate|detect)\s+(all\s+|every\s+|the\s+)?",
                    "", target, flags=re.IGNORECASE) or target
    if not target:
        raise ValueError("Please describe what to find, e.g. 'vehicles' or 'red backpack'.")
    # Qwen2.5-VL gives box coordinates in pixels of the image it sees, and the server rounds both sides
    # to multiples of 28. Doing that rounding here means the model sees exactly the image we sent,
    # so its boxes can be scaled back to the original image precisely.
    size = (max(28, round(image.width / 28) * 28), max(28, round(image.height / 28) * 28))
    sent = image if size == image.size else image.resize(size, Image.Resampling.BICUBIC)
    raw = chat(prompts.LOCATE_PROMPT.format(target=target), [sent])
    detections = parse_boxes(raw, sent.size, image.size, config.BBOX_MODE, default_label=target)
    return {"target": target, "detections": detections, "raw": raw}


def tag(image: Image.Image, focus_tags: list) -> dict:
    """Tag an image and check each case focus item. Returns {"tags", "categories", "focus_hits"}."""
    raw = chat(prompts.TAGS_PROMPT.format(focus_list=", ".join(focus_tags) or "none"), [image])
    return parse_tags(raw, focus_tags)


def summarize(image: Image.Image, length: str, focus: str = "") -> str:
    """Describe an image. `length` is one of prompts.SUMMARY_LENGTHS; `focus` is optional."""
    focus = focus.strip()
    prompt = prompts.SUMMARY_PROMPT.format(length_instruction=prompts.SUMMARY_LENGTHS[length],
                                           focus_instruction=f"Focus on: {focus}. " if focus else "")
    return chat(prompt, [image], max_tokens=SUMMARY_MAX_TOKENS[length]).strip()


def connect(images: dict, focus: str = "") -> dict:
    """
    Compare several images ({image_id: image}, in display order) and find what links them.

    Returns {"insights": [...], "overall_summary": str, "raw": str}; each insight lists its supporting image ids.
    """
    if not 2 <= len(images) <= config.MAX_IMAGES:
        raise ValueError(f"Select between 2 and {config.MAX_IMAGES} images.")
    focus = focus.strip()
    prompt = prompts.CONNECT_PROMPT.format(n=len(images), focus=f"Analyst focus: {focus}" if focus else "")
    raw = chat(prompt, list(images.values()), max_tokens=1500)
    return {**parse_insights(raw, list(images)), "raw": raw}


def ask_group(images: list, question: str) -> str:
    """Follow-up question about a group of images, e.g. 'Which images contain the white van?'."""
    question = question.strip()
    if not question:
        raise ValueError("Please type a question.")
    return chat(prompts.CONNECT_FOLLOWUP_PROMPT.format(n=len(images), question=question), images).strip()


def read_document(image: Image.Image, with_fields: bool = True) -> dict:
    """
    Read a document image: its text as Markdown (headings, tables, fields) and, optionally, key details.

    Returns {"markdown", "document_type", "fields", "quality_issues"}.
    """
    markdown = chat(prompts.OCR_TEXT_PROMPT, [image], max_tokens=2048).strip()
    markdown = re.sub(r"^```(?:markdown|md)?\s*|\s*```$", "", markdown).strip()  # drop a wrapping ``` fence
    result = {"markdown": markdown, "document_type": "unknown", "fields": {}, "quality_issues": []}
    if with_fields:
        result.update(parse_fields(chat(prompts.OCR_FIELDS_PROMPT, [image], max_tokens=800)))
    illegible = markdown.lower().count("[illegible]")
    if illegible:
        result["quality_issues"].append(f"{illegible} word(s) marked [illegible]")
    return result
