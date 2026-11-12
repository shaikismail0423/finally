"""
Central configuration for the Visual Analytics Suite.

Every setting can be overridden with an environment variable (or a `.env`
file in the project root). For the office server only three are needed:
VAS_API_BASE_URL, VAS_API_MODEL and VAS_API_KEY.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional; load .env if it is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("VAS_DATA_DIR", PROJECT_ROOT / "data"))
CASES_DIR = DATA_DIR / "cases"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"

ALLOWED_IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]
MAX_UPLOAD_MB = 25


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to a default."""
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back to a default."""
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Model settings
# --------------------------------------------------------------------------- #
@dataclass
class ModelSettings:
    """All knobs that control how we talk to Qwen2.5-VL on the office server."""

    # Office model server (OpenAI-compatible /v1/chat/completions API)
    api_base_url: str = os.getenv("VAS_API_BASE_URL", "http://localhost:8000/v1")
    api_model_name: str = os.getenv("VAS_API_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    api_key: str = os.getenv("VAS_API_KEY", "not-needed")
    request_timeout_s: int = _env_int("VAS_REQUEST_TIMEOUT", 300)

    # Generation
    temperature: float = _env_float("VAS_TEMPERATURE", 0.1)
    max_new_tokens: int = _env_int("VAS_MAX_NEW_TOKENS", 1024)

    # Image resolution sent to the model (in pixels). Qwen2.5-VL works on
    # 28x28 patches, so the image is resized to a multiple of 28.
    max_pixels_single: int = _env_int("VAS_MAX_PIXELS", 1024 * 28 * 28)  # ~0.8 MP
    max_pixels_multi: int = _env_int("VAS_MAX_PIXELS_MULTI", 256 * 28 * 28)  # max ~0.2 MP per image
    # Total pixel budget shared by ALL images in one multi-image request, so the
    # prompt size stays bounded however many images are selected.
    max_pixels_multi_total: int = _env_int("VAS_MAX_PIXELS_MULTI_TOTAL", 1024 * 28 * 28)

    # Bounding-box coordinate convention returned by the model.
    #   "absolute"  -> pixel coords on the image sent (Qwen2.5-VL default)
    #   "relative"  -> coords normalised to 0..1000 (Qwen2-VL / some other VLMs)
    bbox_mode: str = os.getenv("VAS_BBOX_MODE", "absolute")


@dataclass
class AppSettings:
    """Application-level settings (not model-specific)."""

    app_title: str = "Visual Analytics Suite"
    max_images_visual_multi: int = _env_int("VAS_MAX_MULTI_IMAGES", 6)
    default_focus_tags: list = field(
        default_factory=lambda: ["weapon", "currency", "license plate", "drug paraphernalia", "mobile phone"]
    )


APP_SETTINGS = AppSettings()


def multi_image_pixels(settings: "ModelSettings", n_images: int) -> int:
    """Per-image pixel budget for a multi-image request (shrinks as more images are added)."""
    share = settings.max_pixels_multi_total // max(n_images, 1)
    return max(64 * 28 * 28, min(settings.max_pixels_multi, share))


def ensure_directories() -> None:
    """Create the data folders on first run."""
    CASES_DIR.mkdir(parents=True, exist_ok=True)
