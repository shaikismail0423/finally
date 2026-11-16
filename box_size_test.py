"""
Bounding-box size test — a separate tool, it does not change the app.

Creates synthetic images of 0.5, 0.7, 1, 2 and 5 megapixels with a red circle and a blue square at known
positions, asks the model server to locate each shape, and compares the model's boxes with the true ones:

  * "Full size" — the image is sent at full size (rounded to 28 px), as the app does now.
  * "Pre-resized" — the image is first shrunk to the server's limit, then sent.

Put this file next to app.py and .env, then run:   streamlit run box_size_test.py
"""

import base64
import io
import math
import os
import re

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

SIZES_MP = [0.5, 0.7, 1, 2, 5]
# Shapes as fractions of the image (left, top, right, bottom), so they sit in the same place at every size
SHAPES = {"red circle": (0.10, 0.15, 0.30, 0.45), "blue square": (0.62, 0.15, 0.82, 0.45)}
PROMPT = ("Locate every {target} in the image, output its bbox coordinates using JSON format. "
          'Return a JSON list like [{{"bbox_2d": [x1, y1, x2, y2], "label": "short description"}}]. '
          "If there is none, return [].")


# --------------------------------------------------------------------------- helpers
def make_image(megapixels: float) -> Image.Image:
    """3:2 image of the given size with the shapes drawn at their known positions."""
    width = round(math.sqrt(megapixels * 1_000_000 * 1.5))
    height = round(width / 1.5)
    image = Image.new("RGB", (width, height), "#F2F2F2")
    draw = ImageDraw.Draw(image)
    side = min(width, height)
    for name in SHAPES:
        x1, y1, x2, y2 = true_box(name, width, height)
        if name == "red circle":  # circle: square box, centred in its area
            cx, cy, r = (x1 + x2) / 2, (y1 + y2) / 2, min(x2 - x1, y2 - y1) / 2
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="#D62828")
        else:
            draw.rectangle([x1, y1, x2, y2], fill="#1D4ED8")
    draw.text((side * 0.02, side * 0.02), f"{width}x{height}", fill="#888888")
    return image


def true_box(name: str, width: int, height: int) -> tuple:
    left, top, right, bottom = SHAPES[name]
    x1, y1, x2, y2 = left * width, top * height, right * width, bottom * height
    if name == "red circle":  # the circle fills a square inside its area
        r = min(x2 - x1, y2 - y1) / 2
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return cx - r, cy - r, cx + r, cy + r
    return x1, y1, x2, y2


def server_size(width: int, height: int, max_pixels: int) -> tuple:
    """Qwen2.5-VL's resize rule: sides rounded to 28; shrunk to fit max_pixels if larger."""
    w, h = max(28, round(width / 28) * 28), max(28, round(height / 28) * 28)
    if w * h > max_pixels:
        shrink = math.sqrt(width * height / max_pixels)
        w, h = max(28, math.floor(width / shrink / 28) * 28), max(28, math.floor(height / shrink / 28) * 28)
    return w, h


def ask(image: Image.Image, target: str, url: str, model: str, key: str) -> tuple:
    """Send one image + prompt. Returns (raw answer, prompt tokens reported by the server)."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    data = base64.b64encode(buffer.getvalue()).decode()
    payload = {"model": model, "temperature": 0, "max_tokens": 300, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
        {"type": "text", "text": PROMPT.format(target=target)}]}]}
    response = requests.post(f"{url}/chat/completions", json=payload, headers={"Authorization": f"Bearer {key}"},
                             timeout=300)
    response.raise_for_status()
    body = response.json()
    return body["choices"][0]["message"]["content"], (body.get("usage") or {}).get("prompt_tokens")


def first_box(raw: str):
    found = re.search(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]", raw or "")
    return tuple(map(float, found.groups())) if found else None


def iou(a: tuple, b: tuple) -> float:
    """Overlap of two boxes: 1.0 = identical, 0 = no overlap."""
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def draw(image: Image.Image, truth: dict, found: dict) -> Image.Image:
    """Green = true position, pink = model's box mapped onto the image."""
    canvas = image.copy()
    pen = ImageDraw.Draw(canvas)
    line = max(3, int(max(canvas.size) / 300))
    for box in truth.values():
        pen.rectangle(box, outline="#16A34A", width=line)
    for box in found.values():
        if box:
            pen.rectangle(box, outline="#FF00FF", width=line)
    canvas.thumbnail((700, 700))
    return canvas


# --------------------------------------------------------------------------- page
st.set_page_config(page_title="Box size test", layout="wide")
st.title("Bounding-box size test")
st.write("Each image has a **red circle** and a **blue square** at known positions. The model is asked to locate each "
         "one. In the pictures, **green** = true position, **pink** = where the box would be drawn.")

with st.sidebar:
    url = st.text_input("Server URL", os.getenv("VAS_API_BASE_URL", "")).rstrip("/")
    model = st.text_input("Model", os.getenv("VAS_API_MODEL", ""))
    key = os.getenv("VAS_API_KEY", "not-needed")
    st.caption("API key: read from .env" if os.getenv("VAS_API_KEY") else "API key: not found in .env")
    limit = st.number_input("Server pixel limit (for 'Pre-resized')", value=1280 * 28 * 28, step=28 * 28)
    sizes = st.multiselect("Image sizes (MP)", SIZES_MP, default=SIZES_MP)

if st.button("Run test", type="primary", disabled=not (url and model and sizes)):
    rows, pictures, text_tokens = [], [], None
    progress = st.progress(0.0)
    steps = len(sizes) * 2 * len(SHAPES)
    step = 0
    for mp in sizes:
        original = make_image(mp)
        W, H = original.size
        truth = {name: true_box(name, W, H) for name in SHAPES}
        for method in ("Full size (as the app does now)", "Pre-resized to the server limit"):
            sent_size = server_size(W, H, 10**12) if method.startswith("Full") else server_size(W, H, int(limit))
            sent = original.resize(sent_size, Image.Resampling.BICUBIC)
            found, scores, tokens, raw_boxes = {}, [], None, []
            for name in SHAPES:
                step += 1
                progress.progress(step / steps, text=f"{mp} MP · {method} · {name}")
                try:
                    raw, tokens = ask(sent, name, url, model, key)
                except Exception as exc:  # noqa: BLE001 - show the problem in the table
                    raw, tokens = f"ERROR: {exc}", None
                box = first_box(raw)
                raw_boxes.append(f"{name}: {[round(v) for v in box] if box else raw[:60]}")
                if box:  # map from the size we sent back to the original image, exactly as the app does
                    sx, sy = W / sent_size[0], H / sent_size[1]
                    found[name] = (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
                    scores.append(iou(found[name], truth[name]))
                else:
                    found[name] = None
                    scores.append(0.0)
            tiles_sent = (sent_size[0] // 28) * (sent_size[1] // 28)
            if method.startswith("Pre") and tokens and text_tokens is None:
                text_tokens = tokens - tiles_sent  # pre-resized images are never changed, so this is exact
            rows.append({"image": f"{mp} MP ({W}x{H})", "method": method, "sent": f"{sent_size[0]}x{sent_size[1]}",
                         "tiles in image sent": tiles_sent, "prompt tokens (server)": tokens,
                         "overlap circle": round(scores[0], 2), "overlap square": round(scores[1], 2),
                         "model boxes (raw)": " | ".join(raw_boxes)})
            pictures.append((f"{mp} MP · {method}", draw(original, truth, found)))
    progress.empty()

    table = pd.DataFrame(rows)
    if text_tokens is not None:  # tiles the model really got = server tokens - text tokens
        table["tiles model really got"] = [int(t - text_tokens) if pd.notna(t) else None
                                           for t in table["prompt tokens (server)"]]
        table["server resized?"] = ["?" if pd.isna(got) else "YES" if got != sent else "no"
                                    for got, sent in zip(table["tiles model really got"], table["tiles in image sent"])]
    first = ["image", "method", "sent", "overlap circle", "overlap square", "server resized?"]
    table = table[[c for c in first if c in table] + [c for c in table if c not in first]]
    st.subheader("Results")
    st.caption("Overlap: 1.0 = box exactly right, above 0.7 = good, below 0.5 = wrong place or size."
               + (f" Text part of each request = {text_tokens} tokens." if text_tokens is not None else ""))
    st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("Pictures (green = true position, pink = model's box)")
    for start in range(0, len(pictures), 2):
        for column, (title, picture) in zip(st.columns(2), pictures[start:start + 2]):
            column.image(picture, caption=title, width="stretch")
    st.info("Take screenshots of the table and the pictures.")
