"""
Simple, transparent confidence indicator.

Combines two signals the analyst can understand:
  1. Image quality score (from `quality.py`).
  2. Hedging / uncertainty language in the model's answer
     ("not visible", "unclear", "appears to", ...).

This is a heuristic, not a calibrated probability — the UI says so.
"""

import re
from typing import Optional

_STRONG_UNCERTAINTY = [
    r"not (clearly )?visible", r"cannot (be )?(determine|see|tell|read)", r"can't (determine|see|tell|read)",
    r"unable to", r"illegible", r"not possible to", r"no .* (is|are) visible", r"unclear",
    r"does not (show|appear)", r"insufficient",
]
_SOFT_HEDGES = [r"\bappears?\b", r"\bseems?\b", r"\bpossibly\b", r"\blikely\b", r"\bmight\b", r"\bmay\b",
                r"\bprobably\b", r"\bperhaps\b", r"\bsuggests?\b", r"\bpartially\b"]


def hedging_score(text: str) -> float:
    """Return 0 (assertive) .. 1 (very uncertain) based on wording."""
    if not text:
        return 1.0
    lowered = text.lower()
    strong = sum(bool(re.search(p, lowered)) for p in _STRONG_UNCERTAINTY)
    soft = sum(len(re.findall(p, lowered)) for p in _SOFT_HEDGES)
    words = max(len(lowered.split()), 1)
    soft_density = min(soft / (words / 25), 1.0)  # ~1 hedge per 25 words = high
    return min(1.0, 0.45 * strong + 0.5 * soft_density)


def estimate_confidence(answer: str, quality_score: Optional[int] = None) -> dict:
    """
    Combine image quality and answer wording into High / Medium / Low.

    Returns a dict with `level`, `score` (0..100) and human-readable `reasons`.
    """
    reasons = []
    hedge = hedging_score(answer)
    score = 100 * (1 - hedge)
    if hedge >= 0.45:
        reasons.append("The model expressed uncertainty or said the item is not visible.")
    elif hedge >= 0.2:
        reasons.append("The answer contains hedging language.")
    if quality_score is not None:
        score = 0.6 * score + 0.4 * quality_score
        if quality_score < 75:
            reasons.append(f"Image quality is limited (score {quality_score}/100).")
    level = "High" if score >= 75 else "Medium" if score >= 50 else "Low"
    if not reasons:
        reasons.append("Clear wording and good image quality.")
    return {"level": level, "score": int(round(score)), "reasons": reasons}
