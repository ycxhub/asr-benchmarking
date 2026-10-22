from __future__ import annotations

import base64
import json
import os
from urllib.parse import urlencode

from .base import Event, SessionContext, WebSocketProvider, loads


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
        return json.dumps({"message_type": "commit"})

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
