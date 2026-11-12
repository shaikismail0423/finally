"""
Classic computer-vision image-quality checks (no AI model needed).

Used to warn the analyst when an image is blurry, dark, low-contrast,
tiny or (for documents) skewed — situations where the VLM is more likely
to make mistakes. Thresholds are deliberately simple and easy to tune.
"""

from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
from PIL import Image

# Tunable thresholds
BLUR_SHARP = 120.0  # Laplacian variance above this = sharp
BLUR_BLURRY = 40.0  # below this = clearly blurry
DARK_MEAN = 50
BRIGHT_MEAN = 230
LOW_CONTRAST_STD = 30
LOW_RES_MIN_SIDE = 400
SKEW_WARN_DEGREES = 3.0


@dataclass
class QualityReport:
    """Result of the quality assessment for one image."""

    score: int  # 0 (unusable) .. 100 (excellent)
    level: str  # "Good" | "Fair" | "Poor"
    flags: List[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"score": self.score, "level": self.level, "flags": self.flags, "metrics": self.metrics}


def _to_gray(image: Image.Image, max_side: int = 1024) -> np.ndarray:
    """Convert to a grayscale numpy array, downscaled for speed/consistency."""
    img = image.copy()
    img.thumbnail((max_side, max_side))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)


def estimate_skew_angle(gray: np.ndarray, max_angle: float = 15.0, step: float = 0.5) -> float:
    """
    Estimate the rotation of a document page in degrees (projection-profile method).

    Text lines produce sharp peaks in the row-sum histogram when they are
    horizontal. We rotate the binarised page over a range of angles and keep
    the angle with the "peakiest" profile. Positive = counter-clockwise.
    """
    small = gray.copy()
    scale = 600 / max(small.shape)
    if scale < 1:
        small = cv2.resize(small, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # Adaptive threshold copes with uneven paper/background colour
    binary = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    if binary.mean() < 1:  # almost no ink -> nothing to measure
        return 0.0
    h, w = binary.shape
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + step, step):
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(binary, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        profile = rotated.sum(axis=1, dtype=np.float64)
        score = float(np.sum(np.diff(profile) ** 2))
        if score > best_score:
            best_angle, best_score = float(angle), score
    # Rotating by `best_angle` straightens the page, so the page is tilted by -best_angle
    return -best_angle + 0.0  # "+ 0.0" avoids showing "-0.0"


def assess_quality(image: Image.Image, is_document: bool = False) -> QualityReport:
    """Compute sharpness, brightness, contrast, resolution (and skew for documents)."""
    gray = _to_gray(image)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    min_side = min(image.size)

    flags, score = [], 100
    if blur_var < BLUR_BLURRY:
        flags.append("Image is blurry — fine details and small text may be misread.")
        score -= 35
    elif blur_var < BLUR_SHARP:
        flags.append("Image is slightly soft — double-check small details.")
        score -= 15
    if brightness < DARK_MEAN:
        flags.append("Image is very dark.")
        score -= 20
    elif brightness > BRIGHT_MEAN and not is_document:  # white paper is normal for documents
        flags.append("Image is over-exposed / washed out.")
        score -= 15
    if contrast < LOW_CONTRAST_STD and not (is_document and brightness > BRIGHT_MEAN):
        flags.append("Low contrast.")
        score -= 15
    if min_side < LOW_RES_MIN_SIDE:
        flags.append(f"Low resolution ({image.size[0]}x{image.size[1]} px).")
        score -= 20

    metrics = {
        "sharpness (Laplacian var)": round(blur_var, 1),
        "brightness (0-255)": round(brightness, 1),
        "contrast (std)": round(contrast, 1),
        "resolution": f"{image.size[0]}x{image.size[1]}",
    }
    if is_document:
        skew = estimate_skew_angle(gray)
        metrics["skew (degrees)"] = round(skew, 1)
        if abs(skew) > SKEW_WARN_DEGREES:
            flags.append(f"Document appears skewed by ~{abs(skew):.0f}°.")
            score -= 10

    score = max(0, min(100, score))
    level = "Good" if score >= 75 else "Fair" if score >= 50 else "Poor"
    return QualityReport(score=score, level=level, flags=flags, metrics=metrics)
