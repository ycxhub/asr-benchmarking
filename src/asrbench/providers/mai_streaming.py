from __future__ import annotations

import asyncio
import base64
import json
from urllib.parse import urlsplit

from .. import settings
from .base import Event, ProviderError, SessionContext, WebSocketProvider, loads

_TX = "conversation.item.input_audio_transcription."


class MAIStreamingProvider(WebSocketProvider):
    """Microsoft MAI-Transcribe-2-Streaming over the Foundry Realtime WebSocket.

    The service has no server-side turn detection: a turn is finalized only by
    input_audio_buffer.commit, so native mode commits once at the end of the session.
    """

    family = "microsoft"
    sample_rates = (16000,)
    has_endpoint_ack = True
    wait_for_ready = True
    api_key_env = "AZURE_MAI_API_KEY"
    hosts = ()

    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        self._created = asyncio.Event()
        self._committed_text = ""
        self._uncommitted = False

    def url(self, ctx: SessionContext) -> str:
        ep = settings.env("AZURE_MAI_ENDPOINT")
        if not ep:
            raise ProviderError("AZURE_MAI_ENDPOINT is not set (add it to .env)")
        host = urlsplit(ep if "://" in ep else f"https://{ep}").hostname
        return f"wss://{host}/mai/v1/realtime?intent=transcription"

    def headers(self, ctx: SessionContext) -> dict[str, str]:
        return {"api-key": self.api_key()}

    async def on_open(self, ctx: SessionContext) -> None:
        deployment = settings.env("AZURE_MAI_DEPLOYMENT") or self.params.get("deployment")
        if not deployment:
            raise ProviderError("AZURE_MAI_DEPLOYMENT is not set (add it to .env)")
        await asyncio.wait_for(self._created.wait(), self.ready_timeout_s)
        lp = self.lang_params(ctx.lang)
        transcription: dict = {"model": deployment}
        if lp.get("language"):
            transcription["language"] = lp["language"]
        await self.send_json({
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {"input": {
                    "format": {"type": "audio/pcm", "rate": ctx.sample_rate},
                    "transcription": transcription,
                    "turn_detection": None,
                    "noise_reduction": None,
                }},
            },
        })

    def audio_message(self, chunk: bytes) -> str:
        self._uncommitted = True
        return json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii")})

    def endpoint_message(self) -> str:
        self._uncommitted = False
        return json.dumps({"type": "input_audio_buffer.commit"})

    def finish_message(self) -> str | None:
        return self.endpoint_message() if self._uncommitted else None

    def parse(self, raw) -> list[Event]:
        msg = loads(raw)
        typ = msg.get("type", "")
        if typ == "session.created":
            self._created.set()
            return []
        if typ == "session.updated":
            return [Event("ready")]
        if typ == "error" or typ == _TX + "failed":
            err = msg.get("error") or msg
            return [Event("error", err.get("message", str(err)) if isinstance(err, dict) else str(err), meta=msg)]
        if typ == _TX + "delta":
            self._committed_text += msg.get("delta") or ""
            return [Event("partial", self._committed_text)] if self._committed_text.strip() else []
        if typ == _TX + "intermediate":
            text = self._committed_text + (msg.get("intermediate") or "")
            return [Event("partial", text)] if text.strip() else []
        if typ == _TX + "completed":
            text = msg.get("transcript") or self._committed_text
            self._committed_text = ""
            return [Event("final", text), Event("endpoint", text), Event("turn_end", text)]
        return []
