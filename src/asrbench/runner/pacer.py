from __future__ import annotations

import asyncio
import time

import numpy as np

from ..audio import to_pcm16


class Pacer:
    """Drift-corrected real-time audio pacer. Frame i is due at t0 + i * frame_s."""

    def __init__(self, pcm16: bytes, sample_rate: int, frame_ms: int = 20):
        self.sr = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # pcm16 mono
        if self.frame_bytes <= 0:
            raise ValueError("frame too small")
        pad = (-len(pcm16)) % self.frame_bytes
        self.data = pcm16 + (b"\x00" * pad)
        self.n_frames = len(self.data) // self.frame_bytes
        self.i = 0
        self.t0: float | None = None
        self.lateness_ms: list[float] = []

    def remaining(self) -> int:
        return self.n_frames - self.i

    def audio_time_s(self) -> float:
        return self.i * self.frame_ms / 1000.0

    def silence_frame(self) -> bytes:
        return b"\x00" * self.frame_bytes

    async def next_frame(self, silence: bool = False) -> tuple[bytes, float]:
        """Sleep until the next deadline and return (bytes, send_time)."""
        now = time.perf_counter()
        if self.t0 is None:
            self.t0 = now
        due = self.t0 + self.i * (self.frame_ms / 1000.0)
        wait = due - time.perf_counter()
        if wait > 0:
            await asyncio.sleep(wait)
        sent = time.perf_counter()
        self.lateness_ms.append((sent - due) * 1000.0)
        if silence or self.i >= self.n_frames:
            chunk = self.silence_frame()
        else:
            start = self.i * self.frame_bytes
            chunk = self.data[start : start + self.frame_bytes]
        self.i += 1
        return chunk, sent

    def p99_lateness_ms(self) -> float:
        if not self.lateness_ms:
            return 0.0
        return float(np.percentile(self.lateness_ms, 99))
