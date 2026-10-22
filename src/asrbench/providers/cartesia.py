from __future__ import annotations

import json
from urllib.parse import urlencode

from .base import Event, SessionContext, WebSocketProvider, loads


class CartesiaManualProvider(WebSocketProvider):
    family = "cartesia"
    sample_rates = (8000, 16000)
    has_endpoint_ack = True
    api_key_env = "CARTESIA_API_KEY"
    hosts = ("api.cartesia.ai",)

    def url(self, ctx: SessionContext) -> str:
        enc = "pcm_mulaw" if ctx.condition == "telephony" and ctx.sample_rate == 8000 else self.params.get("encoding", "pcm_s16le")
        q = urlencode({
            "model": self.params.get("model", "ink-2"),
            "encoding": enc,
            "sample_rate": str(ctx.sample_rate),
            "cartesia_version": self.params.get("cartesia_version", "2026-08-14"),
        })
        return f"{self.cfg['url']}?{q}"

    def headers(self, ctx: SessionContext) -> dict[str, str]:
        return {"X-API-Key": self.api_key(), "Cartesia-Version": str(self.params.get("cartesia_version", "2026-08-14"))}

    def endpoint_message(self) -> str:
        return "finalize"

    def finish_message(self) -> str:
        return "close"

    def parse(self, raw) -> list[Event]:
        if isinstance(raw, str) and raw in ("finalize", "close"):
            return []
        try:
            msg = loads(raw)
        except Exception:
            return []
        typ = msg.get("type")
        if typ == "transcript":
            text = msg.get("text") or ""
            if msg.get("is_final"):
                return [Event("final", text)]
            return [Event("partial", text)] if text else []
        if typ == "flush_done":
            return [Event("endpoint", meta=msg)]
        if typ == "error":
            return [Event("error", msg.get("message") or msg.get("title") or "cartesia error", meta=msg)]
        return []


class CartesiaTurnsProvider(WebSocketProvider):
    """Auto turn-detection endpoint. No force-endpoint; native mode only."""

    family = "cartesia"
    sample_rates = (8000, 16000)
    supports_forced = False
    has_endpoint_ack = False
    api_key_env = "CARTESIA_API_KEY"
    hosts = ("api.cartesia.ai",)

    def url(self, ctx: SessionContext) -> str:
        enc = "pcm_mulaw" if ctx.condition == "telephony" and ctx.sample_rate == 8000 else self.params.get("encoding", "pcm_s16le")
        q = urlencode({
            "model": self.params.get("model", "ink-2"),
            "encoding": enc,
            "sample_rate": str(ctx.sample_rate),
            "cartesia_version": self.params.get("cartesia_version", "2026-08-14"),
        })
        return f"{self.cfg['url']}?{q}"

    def headers(self, ctx: SessionContext) -> dict[str, str]:
        return {"X-API-Key": self.api_key(), "Cartesia-Version": str(self.params.get("cartesia_version", "2026-08-14"))}

    def endpoint_message(self) -> None:
        return None

    def finish_message(self) -> str:
        return json.dumps({"type": "close"})

    def parse(self, raw) -> list[Event]:
        try:
            msg = loads(raw)
        except Exception:
            return []
        typ = msg.get("type")
        text = msg.get("transcript") or msg.get("text") or ""
        if typ in ("turn.update", "turn.start"):
            return [Event("partial", text, meta={"turn": typ})] if text else []
        if typ == "turn.end":
            return [Event("final", text), Event("turn_end", text)]
        if typ == "error":
            return [Event("error", msg.get("message") or "cartesia error", meta=msg)]
        return []
