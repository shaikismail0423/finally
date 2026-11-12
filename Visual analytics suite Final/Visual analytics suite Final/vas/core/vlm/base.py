"""
Interface and message format for the vision-language-model backend.

Message format used everywhere in the app (backend-agnostic):

    [
        {"role": "system", "content": "You are ..."},
        {"role": "user", "content": [
            {"type": "image", "image": <PIL.Image>},
            {"type": "text", "text": "What vehicles are visible?"},
        ]},
        {"role": "assistant", "content": "A white van ..."},
    ]

The backend converts this to whatever its server expects.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from PIL import Image


class VLMError(RuntimeError):
    """Raised when the model cannot be reached or returns an error."""


class VLMBackend(ABC):
    """Abstract base class: every backend implements `generate`."""

    @abstractmethod
    def generate(
        self,
        messages: List[dict],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Run the model on a conversation and return the assistant's reply."""

    def health_check(self) -> Tuple[bool, str]:
        """Return (ok, message) describing whether the backend is usable."""
        return True, "Ready"


# --------------------------------------------------------------------------- #
# Small helpers shared by the services
# --------------------------------------------------------------------------- #
def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def image_part(image: Image.Image) -> dict:
    return {"type": "image", "image": image}


def user_message(*parts: dict) -> dict:
    return {"role": "user", "content": list(parts)}


def system_message(text: str) -> dict:
    return {"role": "system", "content": text}
