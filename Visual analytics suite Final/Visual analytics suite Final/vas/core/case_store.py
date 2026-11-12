"""
Case workspace storage (simple, file-based — no database needed).

Layout on disk:

    data/cases/<case-id>/
        case.json          <- all metadata and results for the case
        images/<image-id>.<ext>

Every result (tags, summaries, chat, highlights, OCR, insights) is stored
against its source image id, which gives the "evidence traceability"
required by the spec.
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from config import APP_SETTINGS, CASES_DIR
from vas.utils.image_utils import load_image, validate_image_bytes
from vas.utils.quality import assess_quality


def now_iso() -> str:
    """Current local time as an ISO string (seconds precision)."""
    return datetime.now().isoformat(timespec="seconds")


def slugify(name: str) -> str:
    """Turn a case name into a safe folder name."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return slug[:60] or "case"


class Case:
    """One case folder: images plus a JSON file of results."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.json_path = folder / "case.json"
        self.images_dir = folder / "images"
        self.data: dict = self._load()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> dict:
        if self.json_path.exists():
            with open(self.json_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def save(self) -> None:
        """Write case.json atomically (write temp file, then rename)."""
        tmp = self.json_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2, ensure_ascii=False)
        tmp.replace(self.json_path)

    # ------------------------------------------------------------------ #
    # Basic properties
    # ------------------------------------------------------------------ #
    @property
    def case_id(self) -> str:
        return self.folder.name

    @property
    def name(self) -> str:
        return self.data.get("name", self.case_id)

    @property
    def images(self) -> Dict[str, dict]:
        """Mapping image_id -> metadata/results dict."""
        return self.data.setdefault("images", {})

    @property
    def focus_tags(self) -> List[str]:
        return self.data.setdefault("focus_tags", list(APP_SETTINGS.default_focus_tags))

    def set_focus_tags(self, tags: List[str]) -> None:
        self.data["focus_tags"] = tags
        self.save()

    # ------------------------------------------------------------------ #
    # Images
    # ------------------------------------------------------------------ #
    def add_image(self, data: bytes, filename: str, is_document: bool = False) -> str:
        """
        Validate and store an uploaded image. Returns its id.

        The id is a hash of the file content, so uploading the same file
        twice does not create duplicates.
        """
        image = validate_image_bytes(data)
        image_id = hashlib.sha1(data).hexdigest()[:10]
        if image_id in self.images:
            return image_id
        ext = Path(filename).suffix.lower() or ".png"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        (self.images_dir / f"{image_id}{ext}").write_bytes(data)
        self.images[image_id] = {
            "filename": filename,
            "file": f"{image_id}{ext}",
            "uploaded_at": now_iso(),
            "is_document": is_document,
            "width": image.width,
            "height": image.height,
            "quality": assess_quality(image, is_document).to_dict(),
            "tags": [], "model_tags": [], "focus_hits": [],
            "summaries": {}, "chat": [], "groundings": [], "ocr": None,
        }
        self.save()
        return image_id

    def image_path(self, image_id: str) -> Path:
        return self.images_dir / self.images[image_id]["file"]

    def missing_images(self) -> List[str]:
        """Ids of images whose file is no longer in the images/ folder (e.g. deleted outside the app)."""
        return [image_id for image_id in self.images if not self.image_path(image_id).exists()]

    def load_image(self, image_id: str) -> Image.Image:
        """Load the original image for an id."""
        if image_id not in self.images:
            raise KeyError(f"Unknown image id {image_id}")
        return load_image(self.image_path(image_id))

    def update_image(self, image_id: str, **fields) -> None:
        """Merge fields into an image record and save."""
        self.images[image_id].update(fields)
        self.save()

    def set_document_flag(self, image_id: str, is_document: bool) -> None:
        """Mark an image as a document and recompute quality (adds skew check)."""
        quality = assess_quality(self.load_image(image_id), is_document).to_dict()
        self.update_image(image_id, is_document=is_document, quality=quality)

    def delete_image(self, image_id: str) -> None:
        path = self.image_path(image_id)
        if path.exists():
            path.unlink()
        self.images.pop(image_id, None)
        self.save()

    def label(self, image_id: str) -> str:
        """Human-friendly label: 'filename (id)'."""
        return f"{self.images[image_id]['filename']} ({image_id[:6]})"

    # ------------------------------------------------------------------ #
    # Multi-image insights
    # ------------------------------------------------------------------ #
    @property
    def insights(self) -> List[dict]:
        return self.data.setdefault("insights", [])

    def add_insight_run(self, run: dict) -> None:
        self.insights.insert(0, run)
        self.save()

    # ------------------------------------------------------------------ #
    # Search / filter
    # ------------------------------------------------------------------ #
    def all_tags(self) -> List[str]:
        tags = set()
        for record in self.images.values():
            tags.update(record.get("tags", []))
        return sorted(tags)

    def filter_images(self, tags: Optional[List[str]] = None, match_all: bool = False,
                      text: str = "", flagged_only: bool = False) -> List[str]:
        """Return image ids matching the tag filter, free-text search and focus flags."""
        text = text.strip().lower()
        result = []
        for image_id, rec in self.images.items():
            image_tags = set(rec.get("tags", []))
            if tags:
                ok = set(tags) <= image_tags if match_all else bool(set(tags) & image_tags)
                if not ok:
                    continue
            if flagged_only and not rec.get("focus_hits"):
                continue
            if text:
                haystack = " ".join([rec["filename"], " ".join(image_tags)]
                                    + [s.get("text", "") for s in rec.get("summaries", {}).values()]
                                    + [(rec.get("ocr") or {}).get("markdown", "")]).lower()
                if text not in haystack:
                    continue
            result.append(image_id)
        return result


class CaseStore:
    """Creates, lists and opens cases under `data/cases`."""

    def __init__(self, root: Path = CASES_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def list_cases(self) -> List[Case]:
        cases = [Case(p) for p in sorted(self.root.iterdir()) if (p / "case.json").exists()]
        return sorted(cases, key=lambda c: c.data.get("created_at", ""), reverse=True)

    def create_case(self, name: str, description: str = "") -> Case:
        if not name.strip():
            raise ValueError("Case name cannot be empty.")
        slug, n = slugify(name), 2
        folder = self.root / slug
        while folder.exists():
            folder = self.root / f"{slug}-{n}"
            n += 1
        (folder / "images").mkdir(parents=True)
        case = Case(folder)
        case.data = {"name": name.strip(), "description": description, "created_at": now_iso(),
                     "focus_tags": list(APP_SETTINGS.default_focus_tags),
                     "images": {}, "insights": []}
        case.save()
        return case

    def open_case(self, case_id: str) -> Case:
        folder = self.root / case_id
        if not (folder / "case.json").exists():
            raise FileNotFoundError(f"Case '{case_id}' not found.")
        return Case(folder)
