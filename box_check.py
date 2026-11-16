"""
Box check page: shows whether the model server resizes images before the model sees them.

Sends sample_data/shapes_benchmark.png (900x600, red circle at x 90-270, y 80-260) at several sizes,
asks for the red circle, and works out which image size the model's coordinates refer to.
"""

import math
import re

import pandas as pd
import streamlit as st
from PIL import Image

import config
from backend.model_client import ModelError, chat
from backend.prompts import LOCATE_PROMPT

TRUE_BOX = (90, 80, 270, 260)  # red circle in the 900x600 original


def _size(width: int, height: int, scale: float, round_up: bool = False) -> tuple:
    step = math.ceil if round_up else round
    return max(28, step(width * scale / 28) * 28), max(28, step(height * scale / 28) * 28)


st.title("Box check")
st.caption(f"Server: {config.API_BASE_URL} · Model: {config.API_MODEL}")
st.write("Finds out whether the server resizes images before the model sees them, which is what decides "
         "whether Show Me Where boxes land in the right place. It sends one sample image five times.")

if st.button("Run check", type="primary"):
    original = Image.open(config.SAMPLE_DIR / "shapes_benchmark.png").convert("RGB")
    width, height = original.size
    tests = {
        "Half size": _size(width, height, 0.5),
        "Clean build (rounded to 28 px)": _size(width, height, 1),
        "Final / main build (at least 0.8 MP)": _size(width, height, math.sqrt(1024 * 28 * 28 / (width * height)), True),
        "Large (about 2 MP)": _size(width, height, 1.9),
        "Very large (about 6 MP)": _size(width, height, 3.3),
    }
    rows, answers = [], {}
    progress = st.progress(0.0)
    for number, (name, size) in enumerate(tests.items(), start=1):
        progress.progress(number / len(tests), text=f"Sending {name} ({size[0]}x{size[1]})…")
        row = {"test": name, "sent": f"{size[0]}x{size[1]}"}
        try:
            raw = chat(LOCATE_PROMPT.format(target="red circle"), [original.resize(size, Image.Resampling.BICUBIC)])
        except ModelError as exc:
            rows.append({**row, "result": f"ERROR: {exc}"})
            continue
        answers[name] = raw
        found = re.search(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", raw)
        if not found:
            rows.append({**row, "result": "no box in the answer"})
            continue
        x1, y1, x2, y2 = map(float, found.groups())
        # The red circle is 180 px wide in the 900 px original, so the box width tells us the image width the model saw
        seen_width = (x2 - x1) / (TRUE_BOX[2] - TRUE_BOX[0]) * width
        seen_height = (y2 - y1) / (TRUE_BOX[3] - TRUE_BOX[1]) * height
        ratio = seen_width / size[0]
        rows.append({**row, "model box": f"[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]",
                     "model saw (approx.)": f"{seen_width:.0f}x{seen_height:.0f}",
                     "result": "OK — size kept" if abs(ratio - 1) < 0.04 else f"RESIZED by server (x{ratio:.2f})"})
    progress.empty()
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.info("Please take a screenshot of this table and send it to me.")
    with st.expander("Raw model answers"):
        for name, raw in answers.items():
            st.markdown(f"**{name}**")
            st.code(raw, language="json")
