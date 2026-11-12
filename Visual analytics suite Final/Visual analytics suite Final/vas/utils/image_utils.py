"""
Image helper functions: loading, validation, Qwen-style resizing,
base64 encoding and thumbnails.
"""

import base64
import io
import math
from pathlib import Path
from typing import Tuple, Union

from PIL import Image, ImageOps, UnidentifiedImageError

# Qwen2.5-VL splits images into 14x14 patches and merges 2x2 of them,
# so every side of the input image must be a multiple of 28 pixels.
QWEN_PATCH_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28


class ImageValidationError(ValueError):
    """Raised when an uploaded file is not a usable image."""


def load_image(source: Union[str, Path, bytes, io.BytesIO]) -> Image.Image:
    """
    Load an image from a path or raw bytes, fix EXIF rotation and convert
    to RGB so the rest of the pipeline only ever sees RGB images.
    """
    try:
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)
        image = Image.open(source)
        image = ImageOps.exif_transpose(image)  # respect phone-camera rotation
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError(f"Could not read image: {exc}") from exc


def validate_image_bytes(data: bytes, max_mb: int = 25) -> Image.Image:
    """Check size and decodability of an uploaded file; return the image."""
    if not data:
        raise ImageValidationError("The file is empty.")
    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_mb:
        raise ImageValidationError(f"File is {size_mb:.1f} MB; the limit is {max_mb} MB.")
    image = load_image(data)
    if min(image.size) < 32:
        raise ImageValidationError(f"Image is too small ({image.size[0]}x{image.size[1]} px).")
    return image


def smart_resize_dims(
    height: int, width: int, max_pixels: int, min_pixels: int = MIN_PIXELS, factor: int = QWEN_PATCH_FACTOR
) -> Tuple[int, int]:
    """
    Compute the (height, width) Qwen2.5-VL will actually "see".

    Mirrors `smart_resize` from the official `qwen_vl_utils` package:
    both sides become multiples of 28 and the total pixel count is kept
    between `min_pixels` and `max_pixels` while preserving aspect ratio.
    """
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def prepare_for_model(image: Image.Image, max_pixels: int) -> Image.Image:
    """
    Resize an image to Qwen-friendly dimensions *before* sending it.

    Doing this ourselves means the bounding boxes returned by the model are
    expressed in a coordinate system we know exactly, so we can map them
    back onto the original image reliably (whatever server we use).
    """
    new_h, new_w = smart_resize_dims(image.height, image.width, max_pixels)
    if (new_w, new_h) == image.size:
        return image
    return image.resize((new_w, new_h), Image.Resampling.BICUBIC)


# Grounding images are never sent below this size (1024 x 28 x 28 = ~0.8 MP).
GROUNDING_MIN_PIXELS = 1024 * 28 * 28


def prepare_for_grounding(image: Image.Image, target_pixels: int) -> Image.Image:
    """
    Resize an image for box detection to ~`target_pixels` (never below 0.8 MP), sides multiples of 28.

    Small images are ENLARGED on purpose. Some servers' image processors silently enlarge anything
    below ~0.8 MP before the model sees it; the box coordinates then refer to that bigger, unknown
    image and come back 15-25 % too large. An image that is already >= 0.8 MP is left unchanged,
    so the model's coordinates always refer to exactly the image we sent.
    """
    target = max(target_pixels, GROUNDING_MIN_PIXELS)
    scale = math.sqrt(target / (image.width * image.height))
    new_w = max(QWEN_PATCH_FACTOR, math.ceil(image.width * scale / QWEN_PATCH_FACTOR) * QWEN_PATCH_FACTOR)
    new_h = max(QWEN_PATCH_FACTOR, math.ceil(image.height * scale / QWEN_PATCH_FACTOR) * QWEN_PATCH_FACTOR)
    return image.resize((new_w, new_h), Image.Resampling.BICUBIC)


def image_to_base64(image: Image.Image, fmt: str = "JPEG", quality: int = 92) -> str:
    """Encode an image as a base64 string (used for API requests)."""
    buffer = io.BytesIO()
    save_kwargs = {"quality": quality} if fmt.upper() == "JPEG" else {}
    image.save(buffer, format=fmt, **save_kwargs)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Serialise an image to bytes (used for download buttons)."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def make_thumbnail(image: Image.Image, max_side: int = 320, letterbox: bool = True) -> Image.Image:
    """
    Return a small copy of the image for galleries.

    With `letterbox=True` the thumbnail is centred on a fixed 4:3 canvas so
    gallery cards line up neatly regardless of each image's aspect ratio.
    """
    thumb = image.copy()
    if not letterbox:
        thumb.thumbnail((max_side, max_side))
        return thumb
    canvas_w, canvas_h = max_side, int(max_side * 3 / 4)
    thumb.thumbnail((canvas_w, canvas_h))
    canvas = Image.new("RGB", (canvas_w, canvas_h), "#E9EDF3")
    canvas.paste(thumb, ((canvas_w - thumb.width) // 2, (canvas_h - thumb.height) // 2))
    return canvas
