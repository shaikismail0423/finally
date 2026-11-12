"""Settings, read from the .env file in the project folder (see .env.example)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Office model server (OpenAI-compatible API)
API_BASE_URL = os.getenv("VAS_API_BASE_URL", "http://localhost:8000/v1").rstrip("/")
API_MODEL = os.getenv("VAS_API_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
API_KEY = os.getenv("VAS_API_KEY", "not-needed")
REQUEST_TIMEOUT = int(os.getenv("VAS_REQUEST_TIMEOUT", "300"))  # seconds

# Answers
TEMPERATURE = float(os.getenv("VAS_TEMPERATURE", "0.1"))  # low = consistent, factual
MAX_NEW_TOKENS = int(os.getenv("VAS_MAX_NEW_TOKENS", "1024"))

# Most images that can be compared at once in Connect the Dots
MAX_IMAGES = int(os.getenv("VAS_MAX_IMAGES", "20"))

# Box coordinates returned by the model: "absolute" pixels (Qwen2.5-VL) or "relative" 0-1000 (Qwen2-VL)
BBOX_MODE = os.getenv("VAS_BBOX_MODE", "absolute")

# Files and folders
PROJECT_DIR = Path(__file__).resolve().parent
CASES_DIR = Path(os.getenv("VAS_DATA_DIR", PROJECT_DIR / "data")) / "cases"
SAMPLE_DIR = PROJECT_DIR / "sample_data"
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]

# Items every new case checks for when tagging (editable per case in the Tags tab)
DEFAULT_FOCUS_TAGS = ["weapon", "currency", "license plate", "drug paraphernalia", "mobile phone"]
