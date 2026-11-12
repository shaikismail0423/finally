"""
Client for the office model server (any OpenAI-compatible `/v1/chat/completions` API,
e.g. vLLM serving "Qwen/Qwen2.5-VL-7B-Instruct"). The API key is sent as a Bearer token.

Plain `requests` is used (instead of the `openai` SDK) to keep the code
transparent and dependency-light.
"""

from typing import List, Optional, Tuple

import requests

from vas.utils.image_utils import image_to_base64

from .base import VLMBackend, VLMError


class OpenAICompatibleBackend(VLMBackend):
    """Talks to an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed",
                 timeout_s: int = 300, temperature: float = 0.1, max_new_tokens: int = 1024):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.default_temperature = temperature
        self.default_max_tokens = max_new_tokens

    # ------------------------------------------------------------------ #
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @staticmethod
    def _convert_messages(messages: List[dict]) -> List[dict]:
        """Convert internal messages (PIL images) to OpenAI format (base64 data URLs)."""
        converted = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                converted.append({"role": msg["role"], "content": content})
                continue
            parts = []
            for part in content:
                if part["type"] == "text":
                    parts.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image":
                    b64 = image_to_base64(part["image"])
                    parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            converted.append({"role": msg["role"], "content": parts})
        return converted

    # ------------------------------------------------------------------ #
    def generate(self, messages: List[dict], max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": self.default_temperature if temperature is None else temperature,
            "max_tokens": max_new_tokens or self.default_max_tokens,
        }
        try:
            response = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                     headers=self._headers(), timeout=self.timeout_s)
        except requests.exceptions.ConnectionError as exc:
            raise VLMError(
                f"Cannot reach the model server at {self.base_url}. Check VAS_API_BASE_URL and the network."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise VLMError(f"The model server did not answer within {self.timeout_s} s.") from exc

        if response.status_code != 200 and "context" in response.text.lower():
            raise VLMError(
                "The request is larger than the model's context window. Start the server with a larger "
                "context (vLLM: --max-model-len 32768), lower 'Image detail', or select fewer images. "
                f"Server said: {response.text[:300]}"
            )
        if response.status_code in (401, 403):
            raise VLMError("The server rejected the API key — check VAS_API_KEY in .env.")
        if response.status_code != 200:
            raise VLMError(f"Model server returned HTTP {response.status_code}: {response.text[:400]}")
        try:
            return response.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, ValueError) as exc:
            raise VLMError(f"Unexpected response from model server: {response.text[:400]}") from exc

    def health_check(self) -> Tuple[bool, str]:
        """Check the server is up, the API key is accepted and the configured model is available."""
        try:
            response = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=5)
        except Exception as exc:  # noqa: BLE001 - any failure means "not ready"
            return False, f"Server not reachable at {self.base_url} ({type(exc).__name__})."
        if response.status_code in (401, 403):
            return False, "Server rejected the API key — check VAS_API_KEY in .env."
        if response.status_code != 200:
            # Some gateways don't offer /models; chat requests can still work.
            return True, f"Server reachable · {self.model}"
        try:
            model_ids = [m.get("id", "") for m in response.json().get("data", [])]
        except ValueError:
            model_ids = []
        if model_ids and self.model not in model_ids:
            return False, f"Server is up but model '{self.model}' was not found. Available: {', '.join(model_ids[:5])}"
        return True, f"Connected · {self.model}"
