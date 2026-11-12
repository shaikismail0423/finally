"""Model backend: builds the client for the office Qwen2.5-VL server from the settings."""

from config import ModelSettings

from .base import VLMBackend, VLMError
from .openai_backend import OpenAICompatibleBackend


def create_backend(settings: ModelSettings) -> VLMBackend:
    """Create the client for the OpenAI-compatible model server in `settings`."""
    return OpenAICompatibleBackend(settings.api_base_url, settings.api_model_name, settings.api_key,
                                   settings.request_timeout_s, settings.temperature, settings.max_new_tokens)


__all__ = ["VLMBackend", "VLMError", "create_backend", "OpenAICompatibleBackend"]
