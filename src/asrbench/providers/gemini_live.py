from __future__ import annotations

import base64
import json
from urllib.parse import urlencode

from .base import Event, SessionContext, WebSocketProvider, loads


class GeminiLiveProvider(WebSocketProvider):
    """Gemini 3.5 Transcribe Live over the Live API (BidiGenerateContent WebSocket).

    Forced mode disables server VAD and brackets each turn with activityStart/activityEnd.
    Native mode leaves server VAD on and lets Gemini finalize on its own pauses.
    """

    family = "google"
    sample_rates = (16000,)
    has_endpoint_ack = False
    wait_for_ready = True
    api_key_env = "GEMINI_API_KEY"
    hosts = ("generativelanguage.googleapis.com",)

    def url(self, ctx: SessionContext) -> str:
        return f"{self.cfg['url']}?{urlencode({'key': self.api_key()})}"

    def _manual(self) -> bool:
        return bool(self.ctx and self.ctx.mode == "forced")

    async def on_open(self, ctx: SessionContext) -> None:
        lp = self.lang_params(ctx.lang)
        transcription: dict = {"languageCodes": list(lp.get("language_codes") or [])}
        if self.params.get("mode"):
            transcription["mode"] = self.params["mode"]
        setup: dict = {
            "model": f"models/{self.params.get('model', 'gemini-3.5-transcribe-live')}",
            "generationConfig": {"responseModalities": ["TEXT"]},
            "inputAudioTranscription": transcription,
        }
        if self._manual():
            setup["realtimeInputConfig"] = {"automaticActivityDetection": {"disabled": True}}
        await self.send_json({"setup": setup})

    async def begin_turn(self) -> None:
        if self._manual():
            await self.send_json({"realtimeInput": {"activityStart": {}}})

    def audio_message(self, chunk: bytes) -> str:
        sr = self.ctx.sample_rate if self.ctx else 16000
        return json.dumps({"realtimeInput": {"audio": {
            "data": base64.b64encode(chunk).decode("ascii"),
            "mimeType": f"audio/pcm;rate={sr}",
        }}})

    def endpoint_message(self) -> str:
        return json.dumps({"realtimeInput": {"activityEnd": {}}})

    def finish_message(self) -> str:
        return json.dumps({"realtimeInput": {"audioStreamEnd": True}})

    def parse(self, raw) -> list[Event]:
        msg = loads(raw)
        if "setupComplete" in msg:
            return [Event("ready")]
        if "error" in msg:
            err = msg["error"]
            return [Event("error", err.get("message", str(err)) if isinstance(err, dict) else str(err), meta=msg)]
        if "goAway" in msg:
            return [Event("info", "goAway", meta=msg)]
        sc = msg.get("serverContent") or {}
        evs: list[Event] = []
        interim = (sc.get("interimInputTranscription") or {}).get("text") or ""
        if interim:
            evs.append(Event("partial", interim))
        final = sc.get("inputTranscription")
        if final is not None:
            text = final.get("text") or ""
            evs.append(Event("final", text))
            evs.append(Event("turn_end", text))
        return evs
