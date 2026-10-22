from __future__ import annotations

import base64
import json
from urllib.parse import urlencode

from .base import Event, SessionContext, WebSocketProvider, loads


class SarvamProvider(WebSocketProvider):
    family = "sarvam"
    sample_rates = (8000, 16000)
    has_endpoint_ack = False  # transcript.final is the stop signal
    wait_for_ready = True
    api_key_env = "SARVAM_API_KEY"
    hosts = ("api.sarvam.ai",)

    def url(self, ctx: SessionContext) -> str:
        lp = self.lang_params(ctx.lang)
        q = {
            "model": self.params.get("model", "saaras:v3-realtime"),
            "language_code": lp["language_code"],
            "mode": lp.get("mode", "transcribe"),
            "endpointing": self.params.get("endpointing", "manual"),
            "encoding": "mulaw" if ctx.condition == "telephony" and ctx.sample_rate == 8000 else self.params.get("encoding", "linear16"),
            "sample_rate": str(ctx.sample_rate),
            "stream_type": self.params.get("stream_type", "fast"),
        }
        return f"{self.cfg['url']}?{urlencode(q)}"

    def headers(self, ctx: SessionContext) -> dict[str, str]:
        return {"API-SUBSCRIPTION-KEY": self.api_key()}

    def audio_message(self, chunk: bytes) -> str:
        return json.dumps({"event": "audio_input", "audio": base64.b64encode(chunk).decode("ascii")})

    async def begin_turn(self) -> None:
        await self.send_json({"event": "speech_start"})

    def endpoint_message(self) -> str:
        return json.dumps({"event": "speech_end"})

    async def force_endpoint(self) -> None:
        await self.send_json({"event": "speech_end"})
        await self.send_json({"event": "flush"})

    def finish_message(self) -> str:
        return json.dumps({"event": "end"})

    def parse(self, raw) -> list[Event]:
        msg = loads(raw)
        ev = msg.get("event") or msg.get("type")
        if ev == "session.begin":
            return [Event("ready")]
        if ev == "transcript.partial":
            return [Event("partial", msg.get("text") or "")]
        if ev == "transcript.final":
            text = msg.get("text") or ""
            return [Event("final", text), Event("turn_end", text)]
        if ev == "error":
            return [Event("error", msg.get("message", "sarvam error"), meta=msg)]
        return []
