from __future__ import annotations

import base64
import json
import os
from urllib.parse import urlencode

from .base import Event, SessionContext, WebSocketProvider, loads
from .batch import BatchHTTPProvider


class ElevenLabsProvider(WebSocketProvider):
    family = "elevenlabs"
    sample_rates = (8000, 16000)
    has_endpoint_ack = False
    wait_for_ready = True
    api_key_env = "ELEVENLABS_API_KEY"
    hosts = ("api.elevenlabs.io", "api.in.residency.elevenlabs.io")

    def url(self, ctx: SessionContext) -> str:
        lp = self.lang_params(ctx.lang)
        fmt = "ulaw_8000" if ctx.condition == "telephony" and ctx.sample_rate == 8000 else f"pcm_{ctx.sample_rate}"
        pairs: list[tuple[str, str]] = [
            ("model_id", self.params.get("model_id", "scribe_v2_realtime")),
            ("audio_format", fmt),
            ("language_code", lp.get("language_code", "en")),
            ("commit_strategy", self.params.get("commit_strategy", "manual")),
        ]
        for lang in lp.get("secondary_languages") or []:
            pairs.append(("secondary_languages", lang))
        base = os.environ.get("ELEVENLABS_WS_BASE", self.cfg["url"])
        return f"{base}?{urlencode(pairs)}"

    def headers(self, ctx: SessionContext) -> dict[str, str]:
        return {"xi-api-key": self.api_key()}

    def audio_message(self, chunk: bytes) -> str:
        return json.dumps({
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(chunk).decode("ascii"),
            "commit": False,
            "sample_rate": self.ctx.sample_rate if self.ctx else 16000,
        })

    def endpoint_message(self) -> str:
        # Live Scribe v2 rejects {"message_type":"commit"}. Commit on a silent audio chunk.
        sr = self.ctx.sample_rate if self.ctx else 16000
        silence = b"\x00\x00" * max(1, sr // 100)
        return json.dumps({
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(silence).decode("ascii"),
            "commit": True,
            "sample_rate": sr,
        })

    def parse(self, raw) -> list[Event]:
        msg = loads(raw)
        mt = msg.get("message_type")
        if mt == "session_started":
            return [Event("ready")]
        if mt == "partial_transcript":
            return [Event("partial", msg.get("text") or "")]
        if mt == "committed_transcript":
            text = msg.get("text") or ""
            return [Event("final", text), Event("turn_end", text)]
        if mt and "error" in mt:
            return [Event("error", msg.get("error") or mt, meta=msg)]
        return []


# NON-STREAMING. Not registered. Scribe v2 on the Voice Arena board is this file upload.
# The streaming model is scribe_v2_realtime, registered as `elevenlabs` above.
class ElevenLabsBatchProvider(BatchHTTPProvider):
    """NON-STREAMING file upload (scribe_v2). Not registered in configs/providers.yaml."""

    family = "elevenlabs"
    sample_rates = (16000,)
    api_key_env = "ELEVENLABS_API_KEY"
    hosts = ("api.elevenlabs.io",)

    async def transcribe(self, wav: bytes, ctx: SessionContext) -> str:
        lp = self.lang_params(ctx.lang)
        data = {
            "model_id": self.params.get("model_id", "scribe_v2"),
            "tag_audio_events": str(self.params.get("tag_audio_events", False)).lower(),
            "diarize": "false",
        }
        if lp.get("language_code"):
            data["language_code"] = lp["language_code"]
        r = await self._client.post(
            self.cfg["url"],
            headers={"xi-api-key": self.api_key()},
            data=data,
            files={"file": ("audio.wav", wav, "audio/wav")},
        )
        r.raise_for_status()
        return r.json().get("text") or ""
