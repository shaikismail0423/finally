# Visual Analytics Suite — Office build

A single-page **Streamlit** application that lets an investigator or analyst *talk to their images*:
ask questions, highlight objects, auto-tag, summarise, connect evidence across many images, and read
documents — powered by **Qwen2.5-VL-7B-Instruct** on your organisation's model server
(OpenAI-compatible API, reached with a **base URL + API key**).

> **Decision support only.** Every output is a lead or observation that an analyst must verify.
> The suite performs **no facial recognition, identity matching or judgements about people.**

---

## Contents

1. [Features](#1-features)
2. [Setup](#2-setup)
3. [Using the app (demo script)](#3-using-the-app--suggested-demo-script)
4. [Architecture](#4-architecture)
5. [Folder structure](#5-folder-structure)
6. [How each capability works](#6-how-each-capability-works)
7. [Configuration](#7-configuration)
8. [Troubleshooting](#8-troubleshooting)
9. [Limitations & next steps](#9-limitations--next-steps)

---

## 1. Features

| Capability | Where in the app | What you get |
|---|---|---|
| Free-form questions | **Ask** tab | Chat about an image; follow-ups remember the conversation; "not visible" instead of guessing; confidence badge |
| Visual grounding | **Show Me Where** tab | Labelled boxes drawn on the image, item count, box table, PNG + JSON export |
| Image tagging | **Tags** tab + **Case Gallery** | Categorised tags, case **focus list** (weapons, currency, …) with flagged hits, bulk tagging, edit tags, search/filter by tag |
| Summarisation | **Summary** tab | One-line / paragraph / detailed, optional focus, copy button |
| Multi-image insights | **Connect the Dots** tab | Insights linked to supporting images, "observations to verify", follow-up questions |
| Document OCR & layout | **Documents & OCR** tab | Markdown with headings/tables/fields, table → CSV, key details (names, dates, amounts, phones, …) cross-checked by regex, quality flags |

| Requirement | Implementation |
|---|---|
| Human in the loop | "Verify before use" banners; analyst can edit tags; insights labelled *observations to verify* |
| Evidence traceability | Every result is stored against its source image id in `case.json`; insights list supporting images |
| Confidence indication | Image-quality checks (blur, brightness, contrast, resolution, document skew) + uncertainty-wording heuristic → High/Medium/Low badge |
| Privacy & security | Case files stay on this computer; images are sent only to the organisation's model server |
| Ethical boundaries | Identity/biometric/judgement requests are blocked *before* reaching the model; the system prompt forbids them too |
| Export | PNG, JSON, CSV, Markdown, TXT, and a full Markdown **case report** |

---

## 2. Setup

### 2.1 What you need

| Item | Notes |
|---|---|
| **Python 3.10, 3.11 or 3.12** | https://www.python.org/downloads/ — on Windows tick *"Add python.exe to PATH"* |
| **Office model server details** | Base URL, API key and model name of the Qwen2.5-VL server |
| Internet / network | To install packages once, and to reach the model server |

No GPU is needed on this computer — the model runs on the server.

### 2.2 Install (Windows PowerShell)

```powershell
# 1. Go to the project folder
cd "<path>\Visual analytics suite Final"

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
#   If PowerShell blocks the script, run once:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 3. Install the packages
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux: `python3 -m venv .venv && source .venv/bin/activate`, then the same `pip` commands.

### 2.3 Connect to the model server

Copy `.env.example` to `.env` in the project folder and fill in the three office values:

```ini
VAS_API_BASE_URL=https://<office-qwen-server>/v1     # the part before /chat/completions
VAS_API_MODEL=<model name exactly as the server expects>
VAS_API_KEY=<your api key>
```

Keep `.env` private — it holds your key and is git-ignored.

### 2.4 Run

```powershell
streamlit run app.py
```

Open http://localhost:8501. The header pill should show **Connected · &lt;model&gt;** (or *Server reachable*
if the server has no model list — that is fine). Then sidebar → **Load sample case** → **Ask** tab →
*"What vehicles are visible?"*.

### 2.5 Python packages (installed by `requirements.txt`)

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `pillow` | Image loading, resizing, drawing boxes |
| `numpy` | Numeric arrays |
| `pandas` | Tables and CSV export |
| `opencv-python-headless` | Image-quality checks (blur, skew, contrast) |
| `plotly` | Charts |
| `requests` | Talking to the model server |
| `python-dotenv` | Reads the `.env` file |

---

## 3. Using the app — suggested demo script

1. **Sidebar:** **Load sample case** (6 synthetic images: receipt, invoice, handwritten note, two scenes with the same white van, a shapes benchmark).
   *Choosing the image:* every single-image tab (Ask, Show Me Where, Tags, Summary, Documents) has an **Image**
   picker with **Prev / Next** buttons at the top. The choice is shared — switching it in one tab (or with *Open*
   in the Gallery, or the sidebar) switches it everywhere.
2. **Case Gallery:** KPIs, thumbnails with quality badges, search & filter by tag/text, case report export.
3. **Tags → Tag untagged images:** bulk tagging; then in the Gallery filter by *white van*. Edit tags and **Save**.
4. **Ask** (active image `scene_parking_day.png`): *"What does the sign say?"* → follow-up *"And the number plate?"*. Try *"Who is the driver?"* to show the ethics guard.
5. **Show Me Where:** *license plates* → boxes + count → download marked-up PNG.
6. **Summary:** one-line vs detailed, focus *"vehicles"*.
7. **Connect the Dots:** select the two scenes + receipt + note → *Find connections* (same van & plate, "bring the van" note, duct tape receipt). Follow-up: *"Which images contain the white van?"*.
8. **Documents & OCR** (`receipt_quickmart.png`, `invoice_coastal.png`): structured text, tables → CSV, key details with *model + regex* cross-check; the rotated, blurred note triggers quality warnings.

---

## 4. Architecture

```
                ┌─────────────────────────── Streamlit (app.py) ───────────────────────────┐
                │  sidebar.py   ─ case, upload, active image, model settings               │
   Analyst ───► │  tabs/*.py    ─ one module per tab (UI only)                             │
                │  context.py   ─ AppContext.run_model(): spinner + friendly errors         │
                └──────────────┬──────────────────────────────────────┬────────────────────┘
                               │ calls                                │ reads/writes
                ┌──────────────▼──────────────┐         ┌─────────────▼──────────────┐
                │ services/  (capabilities)    │         │ core/case_store.py         │
                │  qa · grounding · tagging    │         │  data/cases/<id>/case.json │
                │  summarization · multi_image │         │  data/cases/<id>/images/   │
                │  document_ocr · prompts      │         └────────────────────────────┘
                └──────────────┬──────────────┘
                               │ backend.generate(messages)
                ┌──────────────▼──────────────────────────────────────┐
                │ core/vlm/                                            │
                │  OpenAICompatibleBackend ─► office model server      │
                │  (base URL + Bearer API key, /v1/chat/completions)   │
                └──────────────────────────────────────────────────────┘
   utils/ (image prep, JSON parsing, quality, safety, confidence, export)
   visualization/ (box drawing, Plotly charts)
```

Design principles:

* **UI ↔ logic separation** — `services/` never imports Streamlit, so every capability can be reused from a script, notebook or API.
* **One message format** (`core/vlm/base.py`) — services build messages with images as Pillow objects; the backend converts them for the server.
* **Plain files instead of a database** — each case is a folder with `case.json`; easy to inspect, back up or delete.
* **One choke point for model calls** — `AppContext.run_model` handles the spinner and friendly errors.

---

## 5. Folder structure

```
Visual analytics suite Final/
├── app.py                        # Entry point: page setup, sidebar, header, tabs
├── config.py                     # All settings (env-overridable) and paths
├── requirements.txt              # Dependencies
├── .env.example                  # Copy to .env and fill in the office values
├── .streamlit/config.toml        # Theme + upload limit
├── vas/
│   ├── core/
│   │   ├── case_store.py         # Cases, images, results (JSON on disk)
│   │   └── vlm/
│   │       ├── base.py           # Backend interface + message helpers
│   │       └── openai_backend.py # Office server client (base URL + API key)
│   ├── services/                 # The six capabilities
│   │   ├── prompts.py            # Every prompt in one place
│   │   ├── qa.py                 # Questions + follow-ups
│   │   ├── grounding.py          # Boxes: parse, rescale, dedupe
│   │   ├── tagging.py            # Tags + focus list
│   │   ├── summarization.py      # Captions / descriptions
│   │   ├── multi_image.py        # Multi-image insights + follow-ups
│   │   └── document_ocr.py       # Markdown OCR, tables, key fields, regex
│   ├── visualization/
│   │   ├── drawing.py            # Draw labelled boxes
│   │   └── charts.py             # Plotly charts
│   ├── utils/
│   │   ├── image_utils.py        # Load/validate, Qwen smart-resize, base64
│   │   ├── parsing.py            # Robust JSON extraction from model text
│   │   ├── quality.py            # Blur/brightness/contrast/resolution/skew
│   │   ├── safety.py             # Blocks identity/biometric requests
│   │   ├── confidence.py         # High/Medium/Low heuristic
│   │   └── export.py             # CSV + Markdown case report
│   └── ui/
│       ├── context.py            # AppContext + run_model()
│       ├── sidebar.py            # Case/upload/settings sidebar
│       ├── components.py         # Header, badges, image viewer, image picker
│       ├── styles.py             # CSS
│       └── tabs/                 # gallery · ask · grounding · tagging · summary ·
│                                 # connections · document
├── sample_data/                  # Synthetic demo images
└── data/                         # Created at runtime: cases (git-ignored)
```

**Where to change things:** prompts → `vas/services/prompts.py`; quality thresholds → `vas/utils/quality.py`;
blocked request patterns → `vas/utils/safety.py`; default focus list → `config.py`; add a tab → new module in
`vas/ui/tabs/` and one line in `TABS` in `app.py`.

---

## 6. How each capability works

**Image preparation (all capabilities).** Qwen2.5-VL processes images as 28×28-pixel patches. Before sending,
`prepare_for_model` resizes each image with the same `smart_resize` rule as the official `qwen_vl_utils`
(sides multiple of 28, total pixels capped by the *Image detail* setting). Because we control the exact size the
model sees, bounding boxes can be mapped back to the original image precisely.

**Questions.** The image is attached to the first user turn; previous Q&A turns are replayed so follow-ups
have context. The system prompt instructs the model to answer only from what is visible and to say "not visible"
otherwise. `safety.check_request` blocks identity/biometric/judgement questions before any model call.

**Grounding.** Uses Qwen2.5-VL's native grounding prompt (`Locate every {target} … output its bbox
coordinates using JSON format`). The parser accepts `bbox_2d`/`bbox`/`box` keys, nested lists, and the legacy
`<|box_start|>` token format; scales from model-image pixels to original pixels (or from 0–1000 in *relative* mode),
clamps to the image, drops degenerate boxes and near-duplicates (IoU > 0.9). Natural phrasing like
"Highlight all vehicles" is reduced to the target ("vehicles").
Grounding images are always sent at **≥ 0.8 MP** (small images are enlarged): some servers' image processors
silently enlarge smaller images, which would make every box 15–25 % too large. An image of ≥ 0.8 MP is left
unchanged, so the model's coordinates always refer to exactly the image we sent.

**Tagging.** A JSON prompt returns categories (objects, scene, setting, time of day, people, vehicles,
clothing, text present) and an explicit yes/no check for every **case focus item** with evidence. Tags are
normalised and de-duplicated; focus hits are matched to the focus list (e.g. "license plates" → "license plate")
and shown as red flags. Analysts can edit and save tags.
Search/filter in the Gallery supports any/all tag matching, free text (filenames, tags, summaries, OCR text) and
"flagged only".

**Summarisation.** Three length presets with token budgets (80 / 300 / 900) and an optional focus; the
prompt forbids speculation about intent or identity.

**Multi-image insights.**
* *Visual mode* (≤ 6 images by default): images are sent together, labelled *Image 1…N*. They share one pixel
  budget (`VAS_MAX_PIXELS_MULTI_TOTAL`), so adding images makes each smaller instead of overflowing the server's
  context window.
* *Text mode* (larger groups): the model reasons over each image's cached summary + tags (missing captions are
  generated first). Text is cheap, so this scales to many more images.
* The JSON answer is parsed; image numbers are mapped back to image ids (invalid numbers are dropped).

**Documents.** Two model passes: (1) faithful Markdown transcription preserving headings, tables, form
fields and reading order, with `[illegible]` markers; (2) key-field JSON (names, dates, amounts, addresses,
phones, reference numbers, emails, quality issues). A **regex pass** re-extracts emails, phones, dates and amounts
from the transcription; the Key details table shows whether each value was found by *model + regex* (higher trust),
*model* only or *regex only*. Markdown tables are converted to DataFrames for CSV export. Quality warnings combine
the model's issues, `[illegible]` counts and OpenCV checks.

**Image quality.** OpenCV on a downscaled grayscale copy: sharpness = variance of the Laplacian;
brightness = mean; contrast = standard deviation; resolution = shortest side; document skew = projection-profile
method (rotate −15°…+15°, pick the angle with the sharpest row-sum profile). Each problem lowers a 0–100 score.

**Confidence badge.** `0.6 × wording score + 0.4 × quality score`, where the wording score penalises phrases such
as "not visible", "unclear", "appears", "possibly". It is a transparent heuristic, not a calibrated probability.

---

## 7. Configuration

Set in `.env` (copy `.env.example`). Server URL, model name, image detail, temperature and box mode can also be
changed live in the sidebar → **Model settings** (the API key is read from `.env` only).

| Variable | Default | Meaning |
|---|---|---|
| `VAS_API_BASE_URL` | `http://localhost:8000/v1` | **Required.** Office server URL (ends in `/v1`) |
| `VAS_API_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | **Required.** Model name on that server |
| `VAS_API_KEY` | `not-needed` | **Required.** API key (sent as a Bearer token) |
| `VAS_REQUEST_TIMEOUT` | `300` | Seconds to wait for a response |
| `VAS_TEMPERATURE` | `0.1` | Low = consistent, factual |
| `VAS_MAX_NEW_TOKENS` | `1024` | Default answer length limit |
| `VAS_MAX_PIXELS` | `802816` (1024·28·28) | Image detail for single-image tasks |
| `VAS_MAX_PIXELS_MULTI` | `200704` (256·28·28) | Max per-image detail in multi-image mode |
| `VAS_MAX_PIXELS_MULTI_TOTAL` | `802816` (1024·28·28) | Pixel budget shared by all images in one multi-image request |
| `VAS_MAX_MULTI_IMAGES` | `6` | Above this, *Auto* mode in Connect the Dots switches to text mode |
| `VAS_BBOX_MODE` | `absolute` | `relative` for models that output 0–1000 coordinates (e.g. Qwen2-VL) |
| `VAS_DATA_DIR` | `./data` | Where cases are stored |

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| **Model offline** / "Server not reachable" | Check `VAS_API_BASE_URL` and that this computer can reach the server (VPN / office network) |
| "Server rejected the API key" | Check `VAS_API_KEY` in `.env` (no quotes or spaces), then restart the app |
| "model '…' was not found. Available: …" | Copy the exact name from the list into `VAS_API_MODEL` |
| "Model server returned HTTP 404" | `VAS_API_BASE_URL` should end with `/v1`, not `/v1/chat/completions` |
| "larger than the model's context window" | Lower *Image detail*, select fewer images, or ask the server admin for a larger context (vLLM `--max-model-len`) |
| Requests time out | Increase `VAS_REQUEST_TIMEOUT` |
| Changes to `.env` not picked up | Stop the app (Ctrl+C) and run `streamlit run app.py` again |
| `streamlit` not recognised | Activate the virtual environment first (`.\.venv\Scripts\Activate.ps1`) |
| PowerShell won't activate venv | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Start fresh | Stop the app and delete the `data/` folder |

---

## 9. Limitations & next steps

* Confidence is a heuristic; model token log-probabilities (available from vLLM) could make it calibrated.
* There is no authentication or audit log — production use needs SSO/RBAC and audited storage.
* Multi-image *visual* mode is bounded by the server's context window; text mode scales but relies on per-image summaries.
* Out of scope: video, facial recognition, case-management integration, real-time feeds.

Possible next steps: PDF (multi-page) ingestion, vLLM log-prob confidence, per-user access control,
Qwen2.5-VL-32B/72B on a larger server for higher accuracy, and a REST API reusing `vas/services/`.
