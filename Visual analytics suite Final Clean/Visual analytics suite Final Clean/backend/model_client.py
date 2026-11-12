"""Talks to the office Qwen2.5-VL server through its OpenAI-compatible API (base URL + API key)."""

import base64
import io

import requests
from PIL import Image

import config
from backend.prompts import SYSTEM_PROMPT


class ModelError(Exception):
    """The model server could not be reached or returned an error."""


def _headers() -> dict:
    return {"Authorization": f"Bearer {config.API_KEY}"}


def _image_part(image: Image.Image) -> dict:
    """Embed an image in the request as a base64 JPEG (the server cannot read files on this computer)."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


def chat(prompt: str, images: list = (), history: list = (), system: str = SYSTEM_PROMPT,
         max_tokens: int = None) -> str:
    """
    Send a prompt with images to the model and return its answer.

    images:  attached to the first question; when there are several, each is labelled "Image 1:", "Image 2:", ...
    history: earlier turns [{"role": "user" | "assistant", "content": text}] so follow-up questions have context.
    """
    image_parts = []
    for number, image in enumerate(images, start=1):
        if len(images) > 1:
            image_parts.append({"type": "text", "text": f"Image {number}:"})
        image_parts.append(_image_part(image))

    turns = list(history) + [{"role": "user", "content": prompt}]
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": image_parts + [{"type": "text", "text": turns[0]["content"]}]}]
    messages += [{"role": turn["role"], "content": turn["content"]} for turn in turns[1:]]

    payload = {"model": config.API_MODEL, "messages": messages, "temperature": config.TEMPERATURE,
               "max_tokens": max_tokens or config.MAX_NEW_TOKENS}
    try:
        response = requests.post(f"{config.API_BASE_URL}/chat/completions", json=payload, headers=_headers(),
                                 timeout=config.REQUEST_TIMEOUT)
    except requests.exceptions.Timeout:
        raise ModelError(f"The model server did not answer within {config.REQUEST_TIMEOUT} s.") from None
    except requests.exceptions.RequestException as exc:
        raise ModelError(f"Cannot reach the model server at {config.API_BASE_URL}. "
                         "Check VAS_API_BASE_URL and the network.") from exc

    if response.status_code in (401, 403):
        raise ModelError("The server rejected the API key — check VAS_API_KEY in .env.")
    if response.status_code != 200:
        hint = " The request is too large for the server: select fewer images." if "context" in response.text.lower() else ""
        raise ModelError(f"Model server returned HTTP {response.status_code}: {response.text[:400]}{hint}")
    try:
        return response.json()["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError, ValueError):
        raise ModelError(f"Unexpected response from the model server: {response.text[:400]}") from None


def check_connection() -> tuple:
    """Return (ok, message): is the server reachable, is the API key accepted, does it serve our model?"""
    try:
        response = requests.get(f"{config.API_BASE_URL}/models", headers=_headers(), timeout=5)
    except requests.exceptions.RequestException as exc:
        return False, f"Server not reachable at {config.API_BASE_URL} ({type(exc).__name__})."
    if response.status_code in (401, 403):
        return False, "Server rejected the API key — check VAS_API_KEY in .env."
    if response.status_code != 200:  # some gateways have no model list; chat can still work
        return True, f"Server reachable · {config.API_MODEL}"
    try:
        model_ids = [model["id"] for model in response.json()["data"]]
    except (KeyError, TypeError, ValueError):
        model_ids = []
    if model_ids and config.API_MODEL not in model_ids:
        return False, f"Model '{config.API_MODEL}' not found on the server. Available: {', '.join(model_ids[:5])}"
    return True, f"Connected · {config.API_MODEL}"
