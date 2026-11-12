"""
All prompts sent to Qwen2.5-VL, in one place so they are easy to review
and tune. Output formats (JSON schemas) are defined here and parsed in the
matching service module.
"""

from vas.utils.safety import SAFETY_SYSTEM_RULES

BASE_SYSTEM_PROMPT = (
    "You are an image-analysis assistant supporting law-enforcement analysts. "
    "Answer in clear, plain language and base every statement ONLY on what is visible in the image(s). "
    "If something cannot be seen or determined from the image, say so explicitly "
    "(e.g. 'This is not visible in the image') instead of guessing. "
    + SAFETY_SYSTEM_RULES
)

# --------------------------------------------------------------------------- #
# 4.1 Free-form questions
# --------------------------------------------------------------------------- #
QA_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + " Keep answers concise (1-5 sentences) unless asked for detail."

# --------------------------------------------------------------------------- #
# 4.2 Visual grounding — format used by the official Qwen2.5-VL cookbook
# --------------------------------------------------------------------------- #
GROUNDING_PROMPT = (
    "Locate every {target} in the image, output its bbox coordinates using JSON format. "
    'Return a JSON list like [{{"bbox_2d": [x1, y1, x2, y2], "label": "short description"}}]. '
    "If there is none, return []."
)

# --------------------------------------------------------------------------- #
# 4.3 Tagging
# --------------------------------------------------------------------------- #
TAGGING_PROMPT = """Analyse this image and produce descriptive tags for an investigator.
Return ONLY a JSON object with these keys (use short lowercase phrases, empty list if none):
{{
  "objects": [...],          // notable objects, e.g. "cash", "handwritten note", "backpack"
  "scene": [...],            // e.g. "parking lot", "kitchen", "street"
  "setting": [...],          // "indoor" or "outdoor", plus e.g. "urban", "residential"
  "time_of_day": "...",      // "day", "night", "dusk" or "unknown"
  "people": [...],           // counts/visible attributes only, e.g. "two persons" (no identities)
  "vehicles": [...],         // e.g. "white van", "motorcycle"
  "clothing": [...],         // e.g. "red hoodie", "baseball cap"
  "text_present": [...],     // kinds of visible text, e.g. "license plate", "shop sign"
  "focus_items": [           // one entry per case focus item listed below
     {{"item": "...", "present": true/false, "evidence": "where/what you see"}}
  ]
}}
Case focus items to check: {focus_list}.
Only mark a focus item present if it is clearly visible."""

# --------------------------------------------------------------------------- #
# 4.4 Summarisation
# --------------------------------------------------------------------------- #
SUMMARY_LENGTHS = {
    "One-line caption": "Write a single-sentence caption (max 25 words).",
    "Short paragraph": "Write one short paragraph (3-5 sentences).",
    "Detailed description": (
        "Write a detailed, structured description covering scene/setting, people (visible attributes "
        "only), vehicles, objects, visible text, and lighting/time-of-day cues. Use short bullet points."
    ),
}

SUMMARY_PROMPT = (
    "Describe this image for an investigator's case notes. {length_instruction} "
    "{focus_instruction}Describe only what is visible; do not speculate about intent or identity."
)

# --------------------------------------------------------------------------- #
# 4.5 Multi-image insights
# --------------------------------------------------------------------------- #
MULTI_IMAGE_PROMPT = """You are given {n} images labelled Image 1 to Image {n}.
Compare them and find what connects them. Look for:
- the same vehicle or distinctive item in several images
- the same location or landmark
- recurring objects, symbols, logos or text
- a possible timeline/sequence based on visible clues (lighting, clocks, order of events)
- notable differences (an item present in one image and missing in another)
{extra}
Return ONLY a JSON object:
{{
  "insights": [
    {{"type": "recurring_object | same_location | text_or_logo | timeline | difference | other",
      "description": "one or two sentences",
      "images": [list of image numbers that support this],
      "confidence": "high | medium | low"}}
  ],
  "overall_summary": "two sentences on how the images relate"
}}
These are observations for an analyst to verify, not conclusions. Do not identify people."""

MULTI_IMAGE_TEXT_PROMPT = """Below are machine-generated descriptions and tags of {n} images from one case.
Find what connects them (recurring items, vehicles, locations, text, possible timeline, differences).
{descriptions}
{extra}
Return ONLY a JSON object with the same format:
{{"insights": [{{"type": "...", "description": "...", "images": [image numbers], "confidence": "high|medium|low"}}],
  "overall_summary": "..."}}"""

MULTI_IMAGE_FOLLOWUP_PROMPT = (
    "Images are labelled Image 1 to Image {n}. Answer the analyst's question about this group, "
    "always citing image numbers (e.g. 'Images 2 and 5'). Question: {question}"
)

# --------------------------------------------------------------------------- #
# 4.6 Document OCR & layout
# --------------------------------------------------------------------------- #
OCR_MARKDOWN_PROMPT = (
    "Read all text in this document image and reproduce it as Markdown, preserving the layout: "
    "use # headings for titles, Markdown tables for tabular data, '- **Field:** value' for form fields, "
    "and paragraphs for running text. Keep multi-column text in reading order. "
    "Write [illegible] for any word you cannot read. Output only the Markdown."
)

OCR_FIELDS_PROMPT = """Extract key details from this document image.
Return ONLY a JSON object:
{
  "document_type": "e.g. receipt, invoice, ID card, handwritten note, bank statement, chat screenshot",
  "fields": {
    "names": [], "dates": [], "amounts": [], "addresses": [],
    "phone_numbers": [], "reference_numbers": [], "emails": []
  },
  "quality_issues": []   // e.g. "handwritten", "blurry", "skewed", "partially cut off", "low contrast"
}
Copy values exactly as written. Use empty lists when not present."""
