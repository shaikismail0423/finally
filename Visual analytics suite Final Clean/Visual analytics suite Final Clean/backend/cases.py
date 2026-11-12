"""
Case storage: plain files, no database.

    data/cases/<case-id>/case.json      all results, stored per image id
    data/cases/<case-id>/images/<image-id>.<ext>
"""

import hashlib
import io
import json
import re
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

import config


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_image(source) -> Image.Image:
    """Open an image (path or file object), apply phone-camera rotation and convert to RGB."""
    with Image.open(source) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


class Case:
    """One case folder. `data` is the content of case.json; call save() after changing it."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.data = json.loads((folder / "case.json").read_text(encoding="utf-8"))

    @property
    def case_id(self) -> str:
        return self.folder.name

    @property
    def name(self) -> str:
        return self.data.get("name", self.case_id)

    @property
    def images(self) -> dict:
        """{image_id: record} — the record holds the filename and every result for that image."""
        return self.data.setdefault("images", {})

    @property
    def insights(self) -> list:
        """Connect-the-Dots runs, newest first."""
        return self.data.setdefault("insights", [])

    @property
    def focus_tags(self) -> list:
        return self.data.setdefault("focus_tags", list(config.DEFAULT_FOCUS_TAGS))

    def save(self) -> None:
        """Write case.json via a temporary file, so a crash never leaves it half-written."""
        temp = self.folder / "case.json.tmp"
        temp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(self.folder / "case.json")

    def add_image(self, data: bytes, filename: str) -> str:
        """Store an image file and return its id. The id is a hash of the content, so duplicates are skipped."""
        try:
            load_image(io.BytesIO(data))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError(f"{filename} is not a readable image.") from exc
        image_id = hashlib.sha1(data).hexdigest()[:10]
        if image_id not in self.images:
            file = image_id + (Path(filename).suffix.lower() or ".png")
            (self.folder / "images").mkdir(exist_ok=True)
            (self.folder / "images" / file).write_bytes(data)
            self.images[image_id] = {"filename": filename, "file": file, "uploaded_at": now()}
            self.save()
        return image_id

    def image_path(self, image_id: str) -> Path:
        return self.folder / "images" / self.images[image_id]["file"]

    def load_image(self, image_id: str) -> Image.Image:
        return load_image(self.image_path(image_id))

    def update_image(self, image_id: str, **fields) -> None:
        self.images[image_id].update(fields)
        self.save()

    def delete_image(self, image_id: str) -> None:
        self.image_path(image_id).unlink(missing_ok=True)
        del self.images[image_id]
        self.save()

    def missing_images(self) -> list:
        """Ids of images whose file was deleted from the images/ folder outside the app."""
        return [image_id for image_id in self.images if not self.image_path(image_id).exists()]

    def label(self, image_id: str) -> str:
        return f"{self.images[image_id]['filename']} ({image_id[:6]})"


def list_cases() -> list:
    """All cases, newest first."""
    config.CASES_DIR.mkdir(parents=True, exist_ok=True)
    cases = [Case(folder) for folder in config.CASES_DIR.iterdir() if (folder / "case.json").exists()]
    return sorted(cases, key=lambda case: case.data.get("created_at", ""), reverse=True)


def create_case(name: str, description: str = "") -> Case:
    name = name.strip()
    if not name:
        raise ValueError("Case name cannot be empty.")
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60] or "case"
    folder, number = config.CASES_DIR / slug, 2
    while folder.exists():
        folder, number = config.CASES_DIR / f"{slug}-{number}", number + 1
    (folder / "images").mkdir(parents=True)
    data = {"name": name, "description": description, "created_at": now(),
            "focus_tags": list(config.DEFAULT_FOCUS_TAGS), "images": {}, "insights": []}
    (folder / "case.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return Case(folder)


def open_case(case_id: str) -> Case:
    return Case(config.CASES_DIR / case_id)
