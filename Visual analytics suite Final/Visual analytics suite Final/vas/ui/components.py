"""Reusable UI building blocks shared by several tabs."""

import html
from pathlib import Path
from typing import List, Optional

import streamlit as st
from PIL import Image

from vas.ui.context import AppContext
from vas.utils.image_utils import load_image, make_thumbnail

VERIFY_NOTE = "Decision support only — verify every output against the source image before use in a case."


# --------------------------------------------------------------------------- #
# Cached image loading (keyed by path + modification time)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=256)
def _cached_thumbnail(path: str, mtime: float, max_side: int, letterbox: bool) -> Image.Image:
    return make_thumbnail(load_image(path), max_side, letterbox)


def thumbnail(case, image_id: str, max_side: int = 320, letterbox: bool = True) -> Image.Image:
    """Small cached copy of a case image (for galleries and pickers)."""
    path: Path = case.image_path(image_id)
    return _cached_thumbnail(str(path), path.stat().st_mtime, max_side, letterbox)


# --------------------------------------------------------------------------- #
# Header & banners
# --------------------------------------------------------------------------- #
def render_header(ctx: AppContext, backend_ok: bool, backend_msg: str) -> None:
    case_txt = f"Case: <b>{html.escape(ctx.case.name)}</b>" if ctx.case else "No case open"
    pill_class = "ok" if backend_ok else "bad"
    st.markdown(
        f"""<div class="vas-header">
              <div><h1>Visual Analytics Suite</h1>
                   <p>Talk to your images · ask, locate, tag, summarise, connect and read — powered by Qwen2.5-VL-7B</p></div>
              <div style="text-align:right">
                   <span class="vas-pill {pill_class}">{html.escape(backend_msg)}</span><br>
                   <span class="vas-muted" style="color:#fff">{case_txt}</span>
              </div></div>""",
        unsafe_allow_html=True,
    )


def verify_banner() -> None:
    st.caption(f":material/verified_user: {VERIFY_NOTE}")


# --------------------------------------------------------------------------- #
# Badges
# --------------------------------------------------------------------------- #
def _badge(text: str, kind: str) -> str:
    return f'<span class="vas-badge vas-{kind}">{html.escape(text)}</span>'


def quality_badge_html(quality: dict) -> str:
    level = quality.get("level", "?")
    kind = {"Good": "good", "Fair": "fair", "Poor": "poor"}.get(level, "info")
    return _badge(f"Quality: {level} {quality.get('score', '')}", kind)


def confidence_badge_html(confidence: Optional[dict]) -> str:
    if not confidence:
        return ""
    kind = {"High": "good", "Medium": "fair", "Low": "poor"}[confidence["level"]]
    return _badge(f"Confidence: {confidence['level']}", kind)


def tag_chips_html(tags: List[str], focus: Optional[List[str]] = None, limit: int = 12) -> str:
    focus = set(focus or [])
    chips = [f'<span class="vas-chip{" focus" if t in focus else ""}">{html.escape(t)}</span>' for t in tags[:limit]]
    if len(tags) > limit:
        chips.append(f'<span class="vas-chip">+{len(tags) - limit}</span>')
    return "".join(chips)


def show_confidence(confidence: Optional[dict]) -> None:
    """Badge plus an expandable explanation of why."""
    if not confidence:
        return
    st.markdown(confidence_badge_html(confidence), unsafe_allow_html=True,
                help="Heuristic based on image quality and uncertainty wording in the answer: " + " ".join(confidence["reasons"]))


# --------------------------------------------------------------------------- #
# Image viewer with quality information
# --------------------------------------------------------------------------- #
def image_with_quality(ctx: AppContext, image_id: str, image: Optional[Image.Image] = None, caption: str = "") -> None:
    """Large image plus its quality badge and any quality warnings."""
    rec = ctx.case.images[image_id]
    st.image(image if image is not None else ctx.case.load_image(image_id),
             caption=caption or rec["filename"], width="stretch")
    quality = rec.get("quality", {})
    st.markdown(quality_badge_html(quality) + _badge(f"{rec['width']}×{rec['height']}", "info")
                + (_badge("Document", "info") if rec.get("is_document") else ""), unsafe_allow_html=True)
    for flag in quality.get("flags", []):
        st.caption(f":material/warning: {flag}")


def _set_active(image_id: str) -> None:
    """Callback: make `image_id` the active image everywhere (sidebar + all tabs)."""
    st.session_state["active_image"] = image_id


def image_picker(ctx: AppContext, tab_key: str) -> Optional[str]:
    """
    Image selector shown at the top of every single-image tab.

    All pickers (and the sidebar selector) share one "active image", so
    choosing an image here also switches it in the other tabs. Prev/Next
    buttons make it quick to step through a case.
    """
    image_ids = list(ctx.case.images) if ctx.case else []
    if not image_ids:
        return None
    active = ctx.active_image_id if ctx.active_image_id in image_ids else image_ids[0]
    key = f"pick_{tab_key}"
    st.session_state[key] = active  # keep this picker in sync with the shared active image
    index = image_ids.index(active)
    c1, c2, c3, c4 = st.columns([5, 1, 1, 2], vertical_alignment="bottom")
    c1.selectbox(f"Image ({index + 1} of {len(image_ids)})", image_ids, key=key, format_func=ctx.case.label,
                 on_change=lambda: _set_active(st.session_state[key]))
    c2.button("Prev", key=f"{key}_prev", icon=":material/chevron_left:", width="stretch",
              on_click=_set_active, args=(image_ids[index - 1],), disabled=len(image_ids) < 2)
    c3.button("Next", key=f"{key}_next", icon=":material/chevron_right:", width="stretch",
              on_click=_set_active, args=(image_ids[(index + 1) % len(image_ids)],), disabled=len(image_ids) < 2)
    rec = ctx.case.images[active]
    c4.caption(f"{'Document · ' if rec.get('is_document') else ''}{len(rec.get('tags', []))} tags")
    return active


def no_image_placeholder() -> None:
    st.info("Upload images in the sidebar (or load the sample case) and select an **active image** to begin.",
            icon=":material/add_photo_alternate:")
