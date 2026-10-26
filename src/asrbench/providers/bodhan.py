"""NON-STREAMING. Not registered in the harness.

Bodhan indic-transcribe is a file upload (POST /v1/audio/transcriptions, 30 s max).
Voice Arena's Hindi board lists it, and Bodhan publishes no live streaming API.
"""

from __future__ import annotations

from .base import SessionContext
from .batch import BatchHTTPProvider


class BodhanProvider(BatchHTTPProvider):
    """NON-STREAMING file upload. Not registered in configs/providers.yaml."""

    family = "bodhan"
    sample_rates = (16000,)
    api_key_env = "BODHAN_API_KEY"
    hosts = ("api.bodhan.ai",)
    max_segment_s = 29.5

    async def transcribe(self, wav: bytes, ctx: SessionContext) -> str:
        lp = self.lang_params(ctx.lang)
        data = {"model": self.params.get("model", "indic-transcribe")}
        if lp.get("language"):
            data["language"] = lp["language"]
        r = await self._client.post(
            self.cfg["url"],
            headers={"Authorization": f"Bearer {self.api_key()}"},
            data=data,
            files={"file": ("audio.wav", wav, "audio/wav")},
        )
        r.raise_for_status()
        return r.json().get("text") or ""
