from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from .. import settings

log = logging.getLogger(__name__)

# Event kinds emitted by adapters:
#   partial   interim hypothesis for the segment in progress (replaces the previous partial)
#   final     finalized text segment; the session transcript is the concatenation of finals
#   endpoint  provider confirmed a forced endpoint is complete (forced mode)
#   turn_end  provider's own end-of-turn decision (native mode)
#   error     provider-reported error
#   closed    socket closed (emitted by the base class)
EVENT_KINDS = ("partial", "final", "endpoint", "turn_end", "error", "closed", "info")


@dataclass
class Event:
    kind: str
    text: str = ""
    t: float = 0.0
    meta: dict = field(default_factory=dict)


@dataclass
class SessionContext:
    lang: str
    sample_rate: int
    mode: str
    condition: str = "wideband"
    oracle_text: str | None = None
    oracle_turns: list | None = None


class ProviderError(RuntimeError):
    pass


class StreamingProvider:
    """One streaming session. A new instance is created per session.

    Call order used by the session runner:
      connect(ctx) -> begin_turn() -> send_audio()* -> force_endpoint() [forced mode]
      -> begin_turn() for the next turn ... -> finish() -> close()
    """

    family = "base"
    sample_rates: tuple[int, ...] = (16000,)
    frame_ms = 20
    final_only = False
    has_endpoint_ack = True
    supports_forced = True
    supports_native = True
    api_key_env: str | None = None
    hosts: tuple[str, ...] = ()

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg
        self.params: dict = dict(cfg.get("params") or {})
        self.events: asyncio.Queue[Event] = asyncio.Queue()
        self.closed = asyncio.Event()
        self.ctx: SessionContext | None = None

    def lang_params(self, lang: str) -> dict:
        lm = (self.cfg.get("lang_map") or {}).get(lang)
        if lm is None:
            raise ProviderError(f"{self.name} has no configuration for language {lang}")
        return dict(lm)

    def emit(self, kind: str, text: str = "", t: float | None = None, **meta) -> None:
        self.events.put_nowait(Event(kind, text, time.perf_counter() if t is None else t, meta))

    def api_key(self) -> str:
        if not self.api_key_env:
            return ""
        key = settings.env(self.api_key_env)
        if not key:
            raise ProviderError(f"{self.api_key_env} is not set (add it to .env)")
        return key

    async def connect(self, ctx: SessionContext) -> None:
        raise NotImplementedError

    async def begin_turn(self) -> None:
        pass

    async def send_audio(self, chunk: bytes) -> None:
        raise NotImplementedError

    async def force_endpoint(self) -> None:
        raise NotImplementedError

    async def finish(self) -> None:
        pass

    async def close(self) -> None:
        pass


class WebSocketProvider(StreamingProvider):
    wait_for_ready = False
    ready_timeout_s = 10.0
    open_timeout_s = 15.0

    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        self.ws = None
        self._rx: asyncio.Task | None = None
        self._ready = asyncio.Event()

    # --- hooks for subclasses -------------------------------------------------
    def url(self, ctx: SessionContext) -> str:
        raise NotImplementedError

    def headers(self, ctx: SessionContext) -> dict[str, str]:
        return {}

    async def on_open(self, ctx: SessionContext) -> None:
        pass

    def audio_message(self, chunk: bytes) -> bytes | str:
        return chunk

    def endpoint_message(self) -> bytes | str | None:
        return None

    def finish_message(self) -> bytes | str | None:
        return None

    def parse(self, raw: bytes | str) -> list[Event]:
        raise NotImplementedError

    # --- plumbing -------------------------------------------------------------
    async def send_json(self, obj: Any) -> None:
        await self.ws.send(json.dumps(obj, ensure_ascii=False))

    async def connect(self, ctx: SessionContext) -> None:
        self.ctx = ctx
        self.ws = await ws_connect(
            self.url(ctx),
            additional_headers=self.headers(ctx),
            compression=None,
            open_timeout=self.open_timeout_s,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
            close_timeout=3,
        )
        self._rx = asyncio.create_task(self._recv_loop())
        await self.on_open(ctx)
        if self.wait_for_ready:
            try:
                await asyncio.wait_for(self._ready.wait(), self.ready_timeout_s)
            except asyncio.TimeoutError as exc:
                raise ProviderError(f"{self.name}: no session-ready message within {self.ready_timeout_s}s") from exc

    async def _recv_loop(self) -> None:
        code, reason = None, ""
        try:
            async for raw in self.ws:
                t = time.perf_counter()
                try:
                    evs = self.parse(raw)
                except Exception as exc:  # noqa: BLE001
                    evs = [Event("error", f"parse error: {exc}", meta={"raw": str(raw)[:500]})]
                for ev in evs:
                    ev.t = t
                    if ev.kind == "ready":
                        self._ready.set()
                        continue
                    self.events.put_nowait(ev)
        except ConnectionClosed as exc:
            code, reason = exc.rcvd.code if exc.rcvd else None, exc.rcvd.reason if exc.rcvd else ""
        except Exception as exc:  # noqa: BLE001
            reason = repr(exc)
        finally:
            if self.ws is not None and self.ws.close_code is not None and code is None:
                code, reason = self.ws.close_code, self.ws.close_reason or reason
            self.events.put_nowait(Event("closed", t=time.perf_counter(), meta={"code": code, "reason": reason}))
            self.closed.set()

    async def send_audio(self, chunk: bytes) -> None:
        await self.ws.send(self.audio_message(chunk))

    async def force_endpoint(self) -> None:
        msg = self.endpoint_message()
        if msg is not None:
            await self.ws.send(msg)

    async def finish(self) -> None:
        msg = self.finish_message()
        if msg is not None and self.ws is not None and not self.closed.is_set():
            try:
                await self.ws.send(msg)
            except ConnectionClosed:
                pass

    async def close(self) -> None:
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:  # noqa: BLE001
                pass
        if self._rx is not None:
            try:
                await asyncio.wait_for(self._rx, 3)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._rx.cancel()


def loads(raw: bytes | str) -> dict:
    return json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
