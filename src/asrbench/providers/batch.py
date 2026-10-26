"""NON-STREAMING helper. Not used by any registered provider.

File-upload adapters (Bodhan, Saaras v3, Gemini 3 Pro, MAI-Transcribe-2, Scribe v2)
buffer paced audio and POST it when a turn ends. They are kept here and called out
as NON-STREAMING on each class. The harness registers streaming WebSocket providers only.
"""

from __future__ import annotations

import asyncio
import io
import time
import wave

import httpx
import numpy as np

from .base import ProviderError, SessionContext, StreamingProvider


def wav_bytes(pcm16: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return buf.getvalue()


def split_pcm16(pcm16: bytes, sample_rate: int, max_s: float, search_s: float = 5.0) -> list[bytes]:
    """Split into pieces <= max_s, cutting at the quietest 20 ms frame in the last search_s of each window."""
    max_n = int(max_s * sample_rate)
    samples = np.frombuffer(pcm16, dtype=np.int16)
    if len(samples) <= max_n:
        return [pcm16]
    frame = max(1, sample_rate // 50)
    out: list[bytes] = []
    start = 0
    while len(samples) - start > max_n:
        lo = start + max(frame, max_n - int(search_s * sample_rate))
        hi = start + max_n
        win = samples[lo:hi].astype(np.float32)
        n_frames = len(win) // frame
        if n_frames:
            energy = (win[: n_frames * frame].reshape(n_frames, frame) ** 2).mean(axis=1)
            cut = lo + int(np.argmin(energy)) * frame
        else:
            cut = hi
        out.append(samples[start:cut].tobytes())
        start = cut
    out.append(samples[start:].tobytes())
    return out


class BatchHTTPProvider(StreamingProvider):
    """Adapter for file-based (non-streaming) STT APIs.

    Audio is buffered while the session paces it in real time. On each forced endpoint the
    turn's audio is uploaded as one request; in native mode the whole session is uploaded on
    finish(). EOS-to-final latency therefore measures request round-trip, not streaming.
    """

    final_only = True
    has_endpoint_ack = True
    supports_forced = True
    supports_native = True
    max_segment_s: float | None = None
    request_timeout_s = 120.0

    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        self._buf = bytearray()
        self._tasks: list[asyncio.Task] = []
        self._client: httpx.AsyncClient | None = None
        self._uploaded = False

    async def transcribe(self, wav: bytes, ctx: SessionContext) -> str:
        raise NotImplementedError

    async def connect(self, ctx: SessionContext) -> None:
        self.ctx = ctx
        self.api_key()
        timeout = float(self.params.get("request_timeout_s", self.request_timeout_s))
        self._client = httpx.AsyncClient(timeout=timeout)

    async def send_audio(self, chunk: bytes) -> None:
        self._buf.extend(chunk)

    def _flush(self) -> None:
        pcm, self._buf = bytes(self._buf), bytearray()
        self._uploaded = True
        self._tasks.append(asyncio.create_task(self._run(pcm)))

    async def _run(self, pcm: bytes) -> None:
        assert self.ctx is not None
        max_s = self.params.get("max_segment_s", self.max_segment_s)
        pieces = split_pcm16(pcm, self.ctx.sample_rate, float(max_s)) if max_s else [pcm]
        try:
            texts = [await self.transcribe(wav_bytes(p, self.ctx.sample_rate), self.ctx) for p in pieces]
        except httpx.HTTPStatusError as exc:
            self.emit("error", f"{self.name}: HTTP {exc.response.status_code}: {exc.response.text[:300]}")
            return
        except (httpx.HTTPError, ProviderError) as exc:
            self.emit("error", f"{self.name}: {exc!r}")
            return
        text = " ".join(t.strip() for t in texts if t and t.strip())
        t = time.perf_counter()
        self.emit("final", text, t=t, n_pieces=len(pieces))
        self.emit("endpoint", text, t=t)
        self.emit("turn_end", text, t=t)

    async def force_endpoint(self) -> None:
        self._flush()

    async def finish(self) -> None:
        if not self._uploaded and self._buf:
            self._flush()

    async def close(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._client is not None:
            await self._client.aclose()
        self.emit("closed")
        self.closed.set()
