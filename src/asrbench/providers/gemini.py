"""NON-STREAMING. Not registered in the harness.

Gemini 3 Pro takes a whole clip in one generateContent request. It is not a live
transcription API (the Live API is a different product, and Gemini 3 Pro does not
support it). Voice Arena's Hindi board lists this as a batch model.
"""

from __future__ import annotations

import base64

from .base import ProviderError, SessionContext
from .batch import BatchHTTPProvider

DEFAULT_PROMPT = (
    "Transcribe the speech in this audio exactly as spoken, in {language}. "
    "Write Hindi words in Devanagari and keep English words as spoken. "
    "Output only the transcript with no commentary, labels, timestamps or translation. "
    "If there is no speech, output nothing."
)


class GeminiProvider(BatchHTTPProvider):
    """NON-STREAMING one-shot generateContent. Not registered in configs/providers.yaml."""

    family = "google"
    sample_rates = (16000,)
    api_key_env = "GEMINI_API_KEY"
    hosts = ("generativelanguage.googleapis.com",)

    async def transcribe(self, wav: bytes, ctx: SessionContext) -> str:
        lp = self.lang_params(ctx.lang)
        model = self.params.get("model", "gemini-3.1-pro-preview")
        prompt = (self.params.get("prompt") or DEFAULT_PROMPT).format(language=lp.get("language", "Hindi"))
        gen_cfg: dict = {}
        if self.params.get("thinking_level"):
            gen_cfg["thinkingConfig"] = {"thinkingLevel": self.params["thinking_level"]}
        if self.params.get("temperature") is not None:
            gen_cfg["temperature"] = float(self.params["temperature"])
        body = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(wav).decode("ascii")}},
                ],
            }],
            "generationConfig": gen_cfg,
        }
        r = await self._client.post(
            f"{self.cfg['url']}/{model}:generateContent",
            headers={"x-goog-api-key": self.api_key(), "Content-Type": "application/json"},
            json=body,
        )
        r.raise_for_status()
        msg = r.json()
        cands = msg.get("candidates") or []
        if not cands:
            raise ProviderError(f"no candidates: {str(msg.get('promptFeedback') or msg)[:300]}")
        parts = (cands[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
