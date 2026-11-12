"""4.4 Image summarisation with selectable length and focus."""

from PIL import Image

from vas.core.vlm.base import VLMBackend, image_part, system_message, text_part, user_message
from vas.utils.image_utils import prepare_for_model

from .prompts import BASE_SYSTEM_PROMPT, SUMMARY_LENGTHS, SUMMARY_PROMPT

# Token budget per length keeps one-line captions short and fast.
_MAX_TOKENS = {"One-line caption": 80, "Short paragraph": 300, "Detailed description": 900}


def summarize_image(backend: VLMBackend, image: Image.Image, length: str, focus: str, max_pixels: int) -> str:
    """Return a written description of the image."""
    if length not in SUMMARY_LENGTHS:
        raise ValueError(f"Unknown summary length '{length}'. Choose one of {list(SUMMARY_LENGTHS)}.")
    focus = (focus or "").strip()
    focus_instruction = f"Focus on: {focus}. " if focus else ""
    prompt = SUMMARY_PROMPT.format(length_instruction=SUMMARY_LENGTHS[length], focus_instruction=focus_instruction)
    model_image = prepare_for_model(image, max_pixels)
    messages = [system_message(BASE_SYSTEM_PROMPT), user_message(image_part(model_image), text_part(prompt))]
    return backend.generate(messages, max_new_tokens=_MAX_TOKENS[length]).strip()
