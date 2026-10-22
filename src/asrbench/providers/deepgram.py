from __future__ import annotations

import json
from urllib.parse import urlencode

from .base import Event, SessionContext, WebSocketProvider, loads


def _qs(pairs: list[tuple[str, str]]) -> str:
    return urlencode(pairs, doseq=True)


class DeepgramFluxProvider(WebSocketProvider):
    family = "deepgram"
    sample_rates = (8000, 16000)
    has_endpoint_ack = True
    wait_for_ready = True
    api_key_env = "DEEPGRAM_API_KEY"
    hosts = ("api.deepgram.com",)

    def url(self, ctx: SessionContext) -> str:
        lp = self.lang_params(ctx.lang)
        enc = "mulaw" if ctx.condition == "telephony" and ctx.sample_rate == 8000 else self.params.get("encoding", "linear16")
        pairs: list[tuple[str, str]] = [
            ("model", lp.get("model", "flux-general-en")),
            ("encoding", enc),
            ("sample_rate", str(ctx.sample_rate)),
        ]
        for hint in lp.get("language_hint") or []:
            pairs.append(("language_hint", hint))
        return f"{self.cfg['url']}?{_qs(pairs)}"

    def headers(self, ctx: SessionContext) -> dict[str, str]:
        return {"Authorization": f"Token {self.api_key()}"}

    def endpoint_message(self) -> str:
        return json.dumps({"type": "ForceEndTurn"})

    def finish_message(self) -> str:
        return json.dumps({"type": "CloseStream"})

    def parse(self, raw) -> list[Event]:
        msg = loads(raw)
        typ = msg.get("type")
        if typ == "Connected":
            return [Event("ready")]
        if typ == "TurnInfo":
            event = msg.get("event")
            text = msg.get("transcript") or ""
            if event in ("Update", "StartOfTurn", "EagerEndOfTurn", "TurnResumed"):
                return [Event("partial", text, meta={"event": event})] if text else []
            if event == "EndOfTurn":
                evs = [Event("final", text, meta={"trigger": msg.get("trigger")}), Event("turn_end", text)]
                if msg.get("trigger") == "manual":
                    evs.append(Event("endpoint", text, meta={"trigger": "manual"}))
                return evs
        if typ in ("FatalError", "Warning"):
            return [Event("error" if typ == "FatalError" else "info", msg.get("message") or typ, meta=msg)]
        return []


class DeepgramNova3Provider(WebSocketProvider):
    family = "deepgram"
    sample_rates = (8000, 16000)
    has_endpoint_ack = True
    api_key_env = "DEEPGRAM_API_KEY"
    hosts = ("api.deepgram.com",)

    def url(self, ctx: SessionContext) -> str:
        lp = self.lang_params(ctx.lang)
        enc = "mulaw" if ctx.condition == "telephony" and ctx.sample_rate == 8000 else self.params.get("encoding", "linear16")
        pairs = [
            ("model", self.params.get("model", "nova-3")),
            ("language", lp.get("language", "en-US")),
            ("encoding", enc),
            ("sample_rate", str(ctx.sample_rate)),
            ("punctuate", str(self.params.get("punctuate", "true"))),
            ("endpointing", str(self.params.get("endpointing", "false"))),
            ("interim_results", "true"),
        ]
        return f"{self.cfg['url']}?{_qs(pairs)}"

    def headers(self, ctx: SessionContext) -> dict[str, str]:
        return {"Authorization": f"Token {self.api_key()}"}

    def endpoint_message(self) -> str:
        return json.dumps({"type": "Finalize"})

    def finish_message(self) -> str:
        return json.dumps({"type": "CloseStream"})

    def parse(self, raw) -> list[Event]:
        msg = loads(raw)
        if "channel" not in msg and msg.get("type") not in ("Results", None):
            if msg.get("type") in ("Metadata",):
                return []
            if msg.get("type") == "Error" or msg.get("error"):
                return [Event("error", str(msg.get("message") or msg.get("error") or msg), meta=msg)]
        alt = ((msg.get("channel") or {}).get("alternatives") or [{}])[0]
        text = alt.get("transcript") or ""
        if msg.get("from_finalize"):
            return [Event("final", text), Event("endpoint", text, meta={"from_finalize": True})]
        if msg.get("is_final") or msg.get("speech_final"):
            evs = [Event("final", text)]
            if msg.get("speech_final"):
                evs.append(Event("turn_end", text))
            return evs
        return [Event("partial", text)] if text else []
