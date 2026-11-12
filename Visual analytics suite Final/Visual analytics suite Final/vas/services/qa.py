"""4.1 Free-form questions over a single image, with follow-up memory."""

from typing import List

from PIL import Image

from vas.core.vlm.base import VLMBackend, image_part, system_message, text_part, user_message
from vas.utils.image_utils import prepare_for_model
from vas.utils.safety import check_request

from .prompts import QA_SYSTEM_PROMPT


def build_qa_messages(image: Image.Image, history: List[dict], question: str) -> List[dict]:
    """
    Build the conversation sent to the model.

    The image is attached to the *first* user turn; earlier questions and
    answers follow, so the model remembers the conversation about this image.
    `history` is a list of {"role": "user"|"assistant", "content": str}.
    """
    messages = [system_message(QA_SYSTEM_PROMPT)]
    turns = [t for t in history if t.get("role") in ("user", "assistant")] + [{"role": "user", "content": question}]
    for i, turn in enumerate(turns):
        if i == 0:
            messages.append(user_message(image_part(image), text_part(turn["content"])))
        else:
            messages.append({"role": turn["role"], "content": turn["content"]})
    return messages


def ask_question(backend: VLMBackend, image: Image.Image, history: List[dict], question: str,
                 max_pixels: int) -> dict:
    """
    Answer a question about an image.

    Returns {"answer": str, "blocked": bool}. Blocked requests (identity /
    biometric questions) never reach the model.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("Please type a question.")
    policy = check_request(question)
    if policy:
        return {"answer": policy, "blocked": True}
    model_image = prepare_for_model(image, max_pixels)
    answer = backend.generate(build_qa_messages(model_image, history, question))
    return {"answer": answer.strip(), "blocked": False}
