"""
4.2 Visual grounding ("show me where").

Qwen2.5-VL returns boxes as JSON: [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]
in *absolute pixel coordinates of the image it received*. Because we resize
the image ourselves before sending it (see image_utils.prepare_for_grounding),
we know that size and can scale boxes back to the original image. Images are
sent at >= 0.8 MP because some servers enlarge smaller images themselves,
which would otherwise shift every box.
"""

import re
from dataclasses import asdict, dataclass
from typing import List, Tuple

from PIL import Image

from vas.core.vlm.base import VLMBackend, image_part, system_message, text_part, user_message
from vas.utils.image_utils import prepare_for_grounding
from vas.utils.parsing import extract_json
from vas.utils.safety import check_request

from .prompts import BASE_SYSTEM_PROMPT, GROUNDING_PROMPT


def iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Intersection-over-Union of two (x1, y1, x2, y2) boxes (used to drop near-duplicate boxes)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Detection:
    """One highlighted region, in ORIGINAL image pixel coordinates."""

    label: str
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2

    def to_dict(self) -> dict:
        return asdict(self)


def detections_from_record(record: dict) -> List[Detection]:
    """Rebuild Detection objects from a saved grounding result (case.json)."""
    return [Detection(label=d["label"], box=tuple(d["box"])) for d in record.get("detections", [])]


def _boxes_from_json(parsed) -> List[Tuple[str, list]]:
    """Pull (label, [x1,y1,x2,y2]) pairs out of whatever JSON shape the model used."""
    if isinstance(parsed, dict):  # e.g. {"objects": [...]} or a single object
        for key in ("objects", "detections", "results", "boxes"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
        else:
            parsed = [parsed]
    pairs = []
    if not isinstance(parsed, list):
        return pairs
    for item in parsed:
        if isinstance(item, dict):
            coords = item.get("bbox_2d") or item.get("bbox") or item.get("box") or item.get("coordinates")
            label = str(item.get("label") or item.get("name") or item.get("object") or "")
        elif isinstance(item, (list, tuple)):
            coords, label = item, ""
        else:
            continue
        if isinstance(coords, (list, tuple)) and len(coords) == 4:
            try:
                pairs.append((label, [float(c) for c in coords]))
            except (TypeError, ValueError):
                continue
    return pairs


def _boxes_from_tokens(raw: str) -> List[Tuple[str, list]]:
    """Fallback for the older Qwen-VL token format: <|box_start|>(x1,y1),(x2,y2)<|box_end|>."""
    pattern = r"(?:<\|object_ref_start\|>(.*?)<\|object_ref_end\|>)?\s*<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)"
    return [(m[0], [float(v) for v in m[1:]]) for m in re.findall(pattern, raw)]


def parse_detections(raw: str, sent_size: Tuple[int, int], original_size: Tuple[int, int],
                     bbox_mode: str = "absolute", default_label: str = "") -> List[Detection]:
    """
    Convert raw model text into Detection objects on the original image.

    bbox_mode:
      "absolute" — coords are pixels on the image we sent (Qwen2.5-VL)
      "relative" — coords are normalised to 0..1000 (Qwen2-VL and some others)
    """
    pairs = _boxes_from_json(extract_json(raw)) or _boxes_from_tokens(raw)
    sent_w, sent_h = sent_size
    orig_w, orig_h = original_size
    if bbox_mode == "relative":
        sx, sy = orig_w / 1000.0, orig_h / 1000.0
    else:
        sx, sy = orig_w / sent_w, orig_h / sent_h

    detections: List[Detection] = []
    for label, (x1, y1, x2, y2) in pairs:
        x1, x2 = sorted((x1 * sx, x2 * sx))
        y1, y2 = sorted((y1 * sy, y2 * sy))
        x1, x2 = max(0.0, x1), min(float(orig_w), x2)
        y1, y2 = max(0.0, y1), min(float(orig_h), y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue  # degenerate / off-image box
        det = Detection(label=label.strip() or default_label, box=(round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)))
        # Drop near-duplicate boxes (models sometimes repeat themselves)
        if all(iou(det.box, d.box) < 0.9 for d in detections):
            detections.append(det)
    return detections


def ground_objects(backend: VLMBackend, image: Image.Image, target: str, max_pixels: int,
                   bbox_mode: str = "absolute") -> dict:
    """
    Ask the model to locate `target` and return detections on the original image.

    Returns {"detections": [Detection], "raw": str, "blocked": bool, "message": str}.
    """
    target = (target or "").strip().rstrip(".?")
    if not target:
        raise ValueError("Please describe what to find, e.g. 'vehicles' or 'red backpack'.")
    policy = check_request(target)
    if policy:
        return {"detections": [], "raw": "", "blocked": True, "message": policy}

    # Accept natural requests like "Highlight all vehicles" -> "vehicles"
    target = re.sub(r"^(please\s+)?(highlight|find|mark|show|locate|where is|where are|detect)\s+(all\s+|every\s+|the\s+)?",
                    "", target, flags=re.IGNORECASE) or target

    # Always >= 0.8 MP so the server does not resize it again (see prepare_for_grounding)
    model_image = prepare_for_grounding(image, max_pixels)
    messages = [system_message(BASE_SYSTEM_PROMPT),
                user_message(image_part(model_image), text_part(GROUNDING_PROMPT.format(target=target)))]
    raw = backend.generate(messages)
    # Boxes are in pixels of the image we sent; scale them back to the original
    detections = parse_detections(raw, model_image.size, image.size, bbox_mode, default_label=target)
    return {"detections": detections, "raw": raw, "blocked": False, "message": "", "target": target}
