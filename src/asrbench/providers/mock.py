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
        if self.params.get("fail_connect"):
            raise ConnectionError("mock connect refused")
        self.ctx = ctx
        self._audio_s = 0.0
        self._turn = 0
        self._native_fired: set[int] = set()
        self._tasks: set[asyncio.Task] = set()
        await asyncio.sleep(0.005)

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _ms(self, *names: str, default: float) -> float:
        for name in names:
            if name in self.params and self.params[name] is not None:
                return float(self.params[name]) / 1000.0
        return default / 1000.0

    def _turn_text(self, index: int) -> str:
        text = (self.ctx.oracle_text if self.ctx and self.ctx.oracle_text else "hello world").strip()
        turns = (self.ctx.oracle_turns if self.ctx and self.ctx.oracle_turns else [[0.0, 1.0]])
        words = text.split()
        n = max(1, len(turns))
        if n == 1 or not words:
            return text
        base, rem = divmod(len(words), n)
        start = index * base + min(index, rem)
        size = base + (1 if index < rem else 0)
        return " ".join(words[start : start + size]) or text

    def _final_only(self) -> bool:
        return bool(self.params.get("final_only", self.final_only))

    async def _emit_turn(self, index: int, delay: float) -> None:
        text = self._turn_text(index)
        if not self._final_only():
            partial_delay = self._ms("delay_partial_ms", "partial_delay_ms", default=80)
            await asyncio.sleep(min(partial_delay, delay))
            word = text.split()[0] if text else "hello"
            self.emit("partial", word, t=time.perf_counter())
            remain = delay - min(partial_delay, delay)
            if remain > 0:
                await asyncio.sleep(remain)
        else:
            await asyncio.sleep(delay)
        t = time.perf_counter()
        self.emit("final", text, t=t)
        self.emit("endpoint", text, t=t)
        self.emit("turn_end", text, t=t)

    async def send_audio(self, chunk: bytes) -> None:
        if not self.ctx:
            return
        self._audio_s += (len(chunk) // 2) / float(self.ctx.sample_rate or 16000)
        if self.ctx.mode == "forced" or not self.supports_forced:
            return
        silence = self._ms("native_silence_ms", default=500)
        delay = self._ms("native_delay_ms", "delay_final_ms", "finalize_delay_ms", default=300)
        for i, (_start, end) in enumerate(self.ctx.oracle_turns or []):
            if i in self._native_fired:
                continue
            if self._audio_s + 1e-4 >= float(end) + silence:
                self._native_fired.add(i)
                self._spawn(self._emit_turn(i, delay))

    async def force_endpoint(self) -> None:
        index = self._turn
        self._turn += 1
        delay = self._ms("delay_final_ms", "finalize_delay_ms", default=300)
        self._spawn(self._emit_turn(index, delay))

    async def finish(self) -> None:
        return None

    async def close(self) -> None:
        self.emit("closed")
        self.closed.set()
