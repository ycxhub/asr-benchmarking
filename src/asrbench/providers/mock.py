from __future__ import annotations

import asyncio
import time

from .base import SessionContext, StreamingProvider


class MockProvider(StreamingProvider):
    """Injects known delays. Used to prove the harness measures latency correctly."""

    family = "mock"
    sample_rates = (16000, 8000)
    final_only = False
    has_endpoint_ack = True
    api_key_env = None
    hosts = ()

    async def connect(self, ctx: SessionContext) -> None:
        self.ctx = ctx
        self._finals: list[str] = []
        await asyncio.sleep(0.005)  # tiny "handshake" so setup_ms is non-zero

    async def send_audio(self, chunk: bytes) -> None:
        pass

    async def force_endpoint(self) -> None:
        delay_p = float(self.params.get("delay_partial_ms", 80)) / 1000.0
        delay_f = float(self.params.get("delay_final_ms", 300)) / 1000.0
        text = (self.ctx.oracle_text if self.ctx and self.ctx.oracle_text else "hello world")
        asyncio.create_task(self._emit_after(delay_p, "partial", text.split()[0] if text else "hello"))
        asyncio.create_task(self._emit_after(delay_f, "final", text, endpoint=True))

    async def _emit_after(self, delay: float, kind: str, text: str, endpoint: bool = False) -> None:
        await asyncio.sleep(delay)
        t = time.perf_counter()
        self.emit(kind, text, t=t)
        if endpoint:
            self.emit("endpoint", text, t=t)
            self.emit("turn_end", text, t=t)

    async def finish(self) -> None:
        pass

    async def close(self) -> None:
        self.emit("closed")
        self.closed.set()
