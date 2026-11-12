"""Draw grounding results onto images."""

from typing import List, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

# Colour-blind-friendly palette (Okabe–Ito)
BOX_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#FFFFFF"]


def _font(size: int) -> ImageFont.ImageFont:
    """Load a scalable font; fall back to PIL's bitmap font on old Pillow."""
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def draw_boxes(image: Image.Image, boxes: Sequence[Sequence[float]], labels: Optional[List[str]] = None) -> Image.Image:
    """
    Return a copy of `image` with labelled boxes drawn on it.

    Line width and font size scale with image size so results look the same
    on small screenshots and large photos.
    """
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    scale = max(canvas.size) / 1000
    width = max(2, int(3 * scale))
    font = _font(max(12, int(18 * scale)))
    labels = labels or [""] * len(boxes)

    for i, (box, label) in enumerate(zip(boxes, labels)):
        box_color = BOX_COLORS[i % len(BOX_COLORS)]
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=width)
        text = f"{i + 1}. {label}"
        tx1, ty1, tx2, ty2 = draw.textbbox((0, 0), text, font=font)
        tw, th = tx2 - tx1, ty2 - ty1
        ty = y1 - th - 6 if y1 - th - 6 > 0 else y1  # put label above the box when possible
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=box_color)
        draw.text((x1 + 4, ty + 2 - ty1), text, fill="black", font=font)
    return canvas
