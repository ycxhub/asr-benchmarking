"""NON-STREAMING. Not registered in the harness.

MAI-Transcribe-2 on the Voice Arena Hindi board is Azure fast transcription: upload a
file, get one transcript. The live model is a different name, MAI-Transcribe-2-Streaming,
and is not this class.
"""

from __future__ import annotations

import json

from .. import settings
from .base import ProviderError, SessionContext
from .batch import BatchHTTPProvider


class MAITranscribeProvider(BatchHTTPProvider):
    """NON-STREAMING file upload (MAI-Transcribe-2). Not registered in configs/providers.yaml."""

    family = "microsoft"
    sample_rates = (16000,)
    api_key_env = "AZURE_SPEECH_KEY"
    hosts = ()

    def _endpoint(self) -> str:
        ep = settings.env("AZURE_SPEECH_ENDPOINT")
        if ep:
            return ep.rstrip("/")
        region = settings.env("AZURE_SPEECH_REGION")
        if not region:
            raise ProviderError("set AZURE_SPEECH_ENDPOINT or AZURE_SPEECH_REGION in .env")
        return f"https://{region}.api.cognitive.microsoft.com"

    async def connect(self, ctx: SessionContext) -> None:
        self._endpoint()
        await super().connect(ctx)

    async def transcribe(self, wav: bytes, ctx: SessionContext) -> str:
        lp = self.lang_params(ctx.lang)
        definition: dict = {"enhancedMode": {"enabled": True, "model": self.params.get("model", "MAI-Transcribe-2")}}
        if lp.get("locales"):
            definition["locales"] = list(lp["locales"])
        api_version = self.params.get("api_version", "2025-10-15")
        r = await self._client.post(
            f"{self._endpoint()}/speechtotext/transcriptions:transcribe?api-version={api_version}",
            headers={"Ocp-Apim-Subscription-Key": self.api_key()},
            data={"definition": json.dumps(definition)},
            files={"audio": ("audio.wav", wav, "audio/wav")},
        )
        r.raise_for_status()
        phrases = r.json().get("combinedPhrases") or []
        return " ".join(p.get("text", "") for p in phrases).strip()
