"""
Box check: finds out how the model server resizes images for Show Me Where.

Put this file in the "Visual analytics suite Final Clean" folder (next to app.py and .env) and run:
    python box_check.py

It sends sample_data/shapes_benchmark.png (900x600, red circle at x 90-270, y 80-260) at several sizes,
asks for the red circle, and prints which image size the model's coordinates actually refer to.
Nothing is saved; the API key is never printed.
"""

import base64
import io
import json
import math
import os
import re

try:
    import requests
    from dotenv import load_dotenv
    from PIL import Image
except ImportError as exc:  # usually: the virtual environment is not activated
    print(f"ERROR: {exc}\nActivate the project's virtual environment first:  .\\.venv\\Scripts\\Activate.ps1")
    input("\nPress Enter to close this window...")
    raise SystemExit(1)

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
URL = os.getenv("VAS_API_BASE_URL", "http://localhost:8000/v1").rstrip("/")
MODEL = os.getenv("VAS_API_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
KEY = os.getenv("VAS_API_KEY", "not-needed")
TRUE_BOX = (90, 80, 270, 260)  # red circle in the 900x600 original

PROMPT = ("Locate every red circle in the image, output its bbox coordinates using JSON format. "
          'Return a JSON list like [{"bbox_2d": [x1, y1, x2, y2], "label": "short description"}]. '
          "If there is none, return [].")


def round28(w, h, scale=1.0, up=False):
    f = math.ceil if up else round
    return max(28, f(w * scale / 28) * 28), max(28, f(h * scale / 28) * 28)


def ask(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    data = base64.b64encode(buffer.getvalue()).decode()
    payload = {"model": MODEL, "temperature": 0, "max_tokens": 300, "messages": [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
                                     {"type": "text", "text": PROMPT}]}]}
    r = requests.post(f"{URL}/chat/completions", json=payload, headers={"Authorization": f"Bearer {KEY}"}, timeout=300)
    r.raise_for_status()
    body = r.json()
    return body["choices"][0]["message"]["content"], (body.get("usage") or {}).get("prompt_tokens")


def main():
    original = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data",
                                       "shapes_benchmark.png")).convert("RGB")
    w, h = original.size
    tests = {
        "original 900x600 (not /28)": (w, h),
        "Clean build (rounded to 28)": round28(w, h),
        "half size": round28(w, h, 0.5),
        "Final/main build (>=0.8 MP)": round28(w, h, math.sqrt(1024 * 28 * 28 / (w * h)), up=True),
        "large (~2 MP)": round28(w, h, 1.9),
    }
    print(f"Server: {URL}   Model: {MODEL}\n")
    print(f"{'test':32} {'sent':>10} {'model box (raw)':>26} {'model saw ~':>13} {'tokens':>7}  verdict")
    for name, size in tests.items():
        sent = original.resize(size, Image.Resampling.BICUBIC)
        try:
            raw, tokens = ask(sent)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:32} {size[0]}x{size[1]:<5} ERROR: {exc}")
            continue
        m = re.search(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", raw)
        if not m:
            print(f"{name:32} {size[0]}x{size[1]:<5} no box in answer: {raw[:80]!r}")
            continue
        x1, y1, x2, y2 = map(float, m.groups())
        # Size of the image the coordinates refer to, estimated from the box width/height
        seen_w = (x2 - x1) / (TRUE_BOX[2] - TRUE_BOX[0]) * w
        seen_h = (y2 - y1) / (TRUE_BOX[3] - TRUE_BOX[1]) * h
        ratio = seen_w / size[0]
        verdict = "OK (server kept the size)" if abs(ratio - 1) < 0.08 else f"server RESIZED (x{ratio:.2f})"
        print(f"{name:32} {size[0]:>4}x{size[1]:<5} {json.dumps([round(v) for v in (x1, y1, x2, y2)]):>26} "
              f"{seen_w:>6.0f}x{seen_h:<6.0f} {str(tokens):>7}  {verdict}")
    print("\nCopy everything above and send it back.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - show the problem instead of closing the window
        print(f"\nERROR: {type(exc).__name__}: {exc}")
    input("\nPress Enter to close this window...")
