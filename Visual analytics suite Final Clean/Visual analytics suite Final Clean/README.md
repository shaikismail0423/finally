# Visual Analytics Suite

A Streamlit app that lets an analyst *talk to their images* using **Qwen2.5-VL** on the office model server
(OpenAI-compatible API, reached with a **base URL + API key**): ask questions, highlight objects, tag,
summarise, connect evidence across images, and read documents.

> **Decision support only.** Every output is an observation for an analyst to verify.
> The model is instructed never to identify people or judge guilt; there is no face recognition.

---

## Setup (Windows PowerShell)

You need **Python 3.10–3.12** and the office server's **base URL, API key and model name**. No GPU is needed
on this computer.

```powershell
cd "<path>\Visual analytics suite Final Clean"
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked, run once: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in:

```ini
VAS_API_BASE_URL=https://<office-qwen-server>/v1     # the part before /chat/completions
VAS_API_MODEL=<model name exactly as the server expects>
VAS_API_KEY=<your api key>
```

Run:

```powershell
streamlit run app.py
```

Open http://localhost:8501. The sidebar shows **Connected · &lt;model&gt;** when the server and key are OK.
Click **Load sample case** to try it with six synthetic images.

---

## What each tab does

| Tab | What you get |
|---|---|
| **Case Gallery** | All images with tags and flags; filter by tag, search text, "flagged only"; download tags (CSV) and a case report (Markdown); remove an image |
| **Ask** | Chat about the active image; follow-up questions remember the conversation |
| **Show Me Where** | Labelled boxes around what you ask for, a count and table, PNG + JSON download |
| **Tags** | Tags by category, the case **focus list** (weapon, currency, …) checked and flagged, edit tags, tag all images |
| **Summary** | One-line caption, short paragraph or detailed description, optional focus |
| **Connect the Dots** | Compare 2 to `VAS_MAX_IMAGES` images; insights with their supporting images; follow-up questions |
| **Documents & OCR** | Text as Markdown (headings, tables, fields), tables as CSV, key details (names, dates, amounts, …) |

Every single-image tab has an image picker with **Prev / Next**; the choice is shared by all tabs.
Everything is saved per case, so results are still there after a restart.

---

## Project structure

```
Visual analytics suite Final Clean/
├── app.py                  # Builds the page: sidebar + one tab per capability
├── config.py               # Settings read from .env
├── backend/                # Everything that talks to the model or the disk (no Streamlit)
│   ├── model_client.py     # chat() and check_connection(): the only code that calls the server
│   ├── prompts.py          # Every prompt in one place
│   ├── analysis.py         # The six capabilities: ask, locate, tag, summarize, connect, read_document
│   ├── parsing.py          # Turns model answers into data: JSON, boxes, tags, insights, fields, tables
│   └── cases.py            # Case folders: case.json + images/
├── ui/                     # Streamlit screens
│   ├── sidebar.py          # Open/create case, sample case, add images, server status
│   ├── common.py           # Shared helpers: error handling, image picker, previews, box drawing
│   └── tab_*.py            # One file per tab
├── sample_data/            # Synthetic demo images
└── data/                   # Created at runtime: your cases (not shared, git-ignored)
```

How a request flows, e.g. *Show Me Where*:

```
ui/tab_locate.py ──► backend/analysis.locate() ──► backend/model_client.chat() ──► office server
       ▲                      │
       └── boxes drawn ◄──────┴── backend/parsing.parse_boxes()  (saved in case.json by backend/cases.py)
```

**Where to change things:** prompts → `backend/prompts.py`; default focus list → `config.py`;
add a tab → a new `ui/tab_*.py` with a `render(case)` function and one line in `TABS` in `app.py`.

---

## How it works

**Images are sent at full size** — no resizing. The only exception is *Show Me Where*: Qwen2.5-VL returns box
coordinates in pixels of the image it sees, and the server rounds both sides to multiples of 28 px. The app does
that rounding itself first, so the model sees exactly the image that was sent and the boxes map back to the
original precisely. (Very large photos — above about 12.8 megapixels — may still be shrunk by the server, which
would shift boxes on those images.)

**Ask** sends the image with the first question and replays the earlier questions and answers, so follow-ups
have context.

**Tags** asks for JSON with tag categories plus a yes/no check, with evidence, for every case focus item.
Tags are lower-cased and de-duplicated ("White_Van" and "white van" become one tag).

**Connect the Dots** sends the selected images together, labelled *Image 1…N*, and asks for insights as JSON;
image numbers in the answer are mapped back to the images. If the model ignores the format, its text is kept as
one low-confidence observation.

**Documents** makes one call for the text as Markdown and, optionally, a second for key details as JSON.
Markdown tables become downloadable CSV files; `[illegible]` words are counted as a quality warning.

**Model answers are parsed tolerantly**: JSON is found even inside ```json fences or with text around it,
and trailing commas are fixed.

---

## Settings (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `VAS_API_BASE_URL` | `http://localhost:8000/v1` | **Required.** Server URL, ending in `/v1` |
| `VAS_API_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | **Required.** Model name on the server |
| `VAS_API_KEY` | `not-needed` | **Required.** API key (sent as a Bearer token) |
| `VAS_MAX_IMAGES` | `20` | Most images compared at once in Connect the Dots — raise it as far as the server handles |
| `VAS_REQUEST_TIMEOUT` | `300` | Seconds to wait for an answer |
| `VAS_TEMPERATURE` | `0.1` | Low = consistent, factual answers |
| `VAS_MAX_NEW_TOKENS` | `1024` | Default answer length limit |
| `VAS_BBOX_MODE` | `absolute` | `relative` only for models that answer in 0–1000 coordinates (e.g. Qwen2-VL) |
| `VAS_DATA_DIR` | `./data` | Where cases are stored |

Restart the app after changing `.env`. The upload limit per file (25 MB) is in `.streamlit/config.toml`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Server not reachable" | Check `VAS_API_BASE_URL` and that this computer can reach the server (VPN / office network) |
| "Server rejected the API key" | Check `VAS_API_KEY` in `.env` (no quotes or spaces), then restart |
| "Model '…' not found. Available: …" | Copy the exact name from the list into `VAS_API_MODEL` |
| "HTTP 404" | `VAS_API_BASE_URL` must end with `/v1`, not `/v1/chat/completions` |
| "request is too large" | Select fewer images in Connect the Dots, or lower `VAS_MAX_IMAGES` |
| Requests time out | Increase `VAS_REQUEST_TIMEOUT` |
| Image files missing | The sidebar offers to remove them from the case (or put the files back in the case's `images/` folder) |
| `streamlit` not recognised | Activate the virtual environment first (`.\.venv\Scripts\Activate.ps1`) |
| Start fresh | Stop the app and delete the `data/` folder |
