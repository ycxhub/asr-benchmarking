from __future__ import annotations

import json
from urllib.parse import urlencode

from .base import Event, SessionContext, WebSocketProvider, loads


class MurfProvider(WebSocketProvider):
    family = "murf"
    sample_rates = (8000, 16000)
    final_only = False
    has_endpoint_ack = True
    api_key_env = "MURF_API_KEY"
    hosts = ("in.api.murf.ai",)

    def url(self, ctx: SessionContext) -> str:
        lp = self.lang_params(ctx.lang)
        q = urlencode({"api_key": self.api_key(), "language": lp["language"]})
        return f"{self.cfg['url']}?{q}"

    async def on_open(self, ctx: SessionContext) -> None:
        codec = "mulaw" if ctx.condition == "telephony" and ctx.sample_rate == 8000 else "pcm_s16le"
        await self.send_json({"type": "AudioConfig", "codec": codec, "sample_rate": str(ctx.sample_rate), "channels": "1"})

    def endpoint_message(self) -> str:
        return json.dumps({"type": "Finalize"})

    def finish_message(self) -> str:
        return json.dumps({"type": "CloseStream"})

    def parse(self, raw) -> list[Event]:
        msg = loads(raw)
        kind = msg.get("type")
        inner = msg.get("rawEvent") if isinstance(msg.get("rawEvent"), dict) else {}
        text = msg.get("transcript") or inner.get("transcript") or ""
        event = inner.get("event")
        if kind in ("Error", "error"):
            return [Event("error", msg.get("message") or inner.get("description") or "murf error", meta=msg)]
        if kind == "QuotaWarning":
            return [Event("info", f"quota {msg.get('remaining_funds')}", meta=msg)]
        if kind == "Results":
            evs = [Event("final", text, meta={k: msg.get(k) for k in ("chunk_id", "speech_final", "session_id")})]
            if msg.get("speech_final"):
                evs.append(Event("endpoint", text, meta={"speech_final": True}))
                evs.append(Event("turn_end", text, meta={"speech_final": True}))
            return evs
        if kind == "partial" or event in ("Update", "StartOfTurn", "EagerEndOfTurn", "TurnResumed"):
            return [Event("partial", text, meta={"event": event})] if text else []
        if kind == "final" or event == "EndOfTurn":
            evs = [Event("final", text, meta={"trigger": inner.get("trigger"), "event": event})]
            evs.append(Event("turn_end", text))
            if inner.get("trigger") == "manual" or kind == "final":
                evs.append(Event("endpoint", text, meta={"trigger": inner.get("trigger") or "manual"}))
            return evs
        return []
