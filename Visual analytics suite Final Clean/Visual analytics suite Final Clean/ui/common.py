"""Small UI helpers used by several tabs."""

import io

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from backend.cases import Case, load_image
from backend.model_client import ModelError

VERIFY_NOTE = "Decision support only — verify every output against the source image before use in a case."
BOX_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#FFFFFF"]


def run_model(task, spinner: str = "Analysing…"):
    """Run a model call with a spinner. Problems are shown as messages; returns None if the call failed."""
    try:
        with st.spinner(spinner):
            return task()
    except ValueError as exc:  # something wrong with the input
        st.warning(str(exc))
    except ModelError as exc:  # server down, wrong key, request too large, ...
        st.error(f"Model error: {exc}")
    except Exception as exc:  # last resort, so one bad answer never crashes the app
        st.error(f"Unexpected error ({type(exc).__name__}): {exc}")
    return None


@st.cache_data(show_spinner=False, max_entries=1000)
def _preview(path: str, modified: float, size: int) -> Image.Image:
    image = load_image(path)
    image.thumbnail((size, size))
    return image


def preview(case: Case, image_id: str, size: int = 1600) -> Image.Image:
    """Smaller copy of an image for the screen only (never sent to the model). Cached until the file changes."""
    path = case.image_path(image_id)
    return _preview(str(path), path.stat().st_mtime, size)


def _set_active(image_id: str) -> None:
    st.session_state["active_image"] = image_id


def image_picker(case: Case, key: str):
    """
    Image selector with Prev/Next, shown at the top of each single-image tab. All tabs share one
    "active image", so changing it in one tab changes it everywhere. Returns the image id (None if no images).
    """
    image_ids = list(case.images)
    if not image_ids:
        st.info("Add images in the sidebar (or click **Load sample case**) to begin.")
        return None
    active = st.session_state.get("active_image")
    if active not in image_ids:
        active = image_ids[0]
        st.session_state["active_image"] = active
    widget_key = f"pick_{key}"
    st.session_state[widget_key] = active  # keep this tab's selector in step with the shared active image
    index = image_ids.index(active)
    c1, c2, c3 = st.columns([6, 1, 1], vertical_alignment="bottom")
    c1.selectbox(f"Image ({index + 1} of {len(image_ids)})", image_ids, key=widget_key, format_func=case.label,
                 on_change=lambda: _set_active(st.session_state[widget_key]))
    c2.button("Prev", key=f"{widget_key}_prev", on_click=_set_active, args=(image_ids[index - 1],),
              disabled=len(image_ids) < 2, width="stretch")
    c3.button("Next", key=f"{widget_key}_next", on_click=_set_active, args=(image_ids[(index + 1) % len(image_ids)],),
              disabled=len(image_ids) < 2, width="stretch")
    return active


def draw_boxes(image: Image.Image, detections: list) -> Image.Image:
    """Copy of `image` with numbered, labelled boxes; line and text size grow with the image."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    scale = max(canvas.size) / 1000
    line_width = max(2, int(3 * scale))
    font = ImageFont.load_default(size=max(12, int(18 * scale)))
    for number, detection in enumerate(detections, start=1):
        color = BOX_COLORS[(number - 1) % len(BOX_COLORS)]
        x1, y1, x2, y2 = detection["box"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        text = f"{number}. {detection['label']}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        label_y = y1 - (bottom - top) - 6 if y1 - (bottom - top) - 6 > 0 else y1  # above the box if there is room
        draw.rectangle([x1, label_y, x1 + (right - left) + 8, label_y + (bottom - top) + 6], fill=color)
        draw.text((x1 + 4, label_y + 3 - top), text, fill="black", font=font)
    return canvas


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def csv_bytes(table) -> bytes:
    """CSV with a byte-order mark so Excel shows non-English text correctly."""
    return table.to_csv(index=False).encode("utf-8-sig")
