"""
elevenlabs_tts.py
------------------
Plain, framework-agnostic ElevenLabs text-to-speech helper — used directly
by the Flask routes in ollama_service2.py (no FastAPI dependency).

Env vars:
    ELEVENLABS_API_KEY   (required)
    ELEVENLABS_VOICE_ID  (default: ElevenLabs' stock "Rachel" voice)
"""

from __future__ import annotations

import os

import requests

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # "Rachel"


class TTSError(Exception):
    def __init__(self, message, status_code=502):
        super().__init__(message)
        self.status_code = status_code


def synthesize(text: str, voice_id: str | None = None,
               stability: float = 0.5, similarity_boost: float = 0.75) -> bytes:
    """Returns MP3 audio bytes for `text`. Raises TTSError on failure."""
    if not ELEVENLABS_API_KEY:
        raise TTSError("ELEVENLABS_API_KEY is not configured", status_code=503)
    if not text or not text.strip():
        raise TTSError("'text' is required", status_code=400)

    voice_id = voice_id or ELEVENLABS_VOICE_ID
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    try:
        resp = requests.post(
            url,
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": text,
                "voice_settings": {"stability": stability, "similarity_boost": similarity_boost},
            },
            timeout=60,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise TTSError(f"ElevenLabs request failed: {exc}") from exc

    return resp.content


def list_voices() -> dict:
    if not ELEVENLABS_API_KEY:
        raise TTSError("ELEVENLABS_API_KEY is not configured", status_code=503)
    try:
        resp = requests.get(
            "https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise TTSError(f"ElevenLabs request failed: {exc}") from exc
    return resp.json()
