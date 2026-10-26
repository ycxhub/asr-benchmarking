from __future__ import annotations

import asyncio
import math
import time
from dataclasses import asdict, dataclass, field

from ..audio import prepare_stream_audio, read_pcm16
from ..datasets.manifest import Dataset, Item
from ..providers.base import Event, ProviderError, SessionContext, StreamingProvider
from .pacer import Pacer

FALLBACK_S = 2.0  # Artificial Analysis fallback when there is no force-endpoint ack


@dataclass(frozen=True)
class TurnFrames:
    start_frame: int
    end_frame: int


def plan_turn_frames(turns: list[list[float]], frame_s: float, frame_samples: int = 0) -> list[TurnFrames]:
    """Inclusive frame indexes whose audio covers each [start, end] interval."""
    del frame_samples
    out: list[TurnFrames] = []
    for start, end in turns:
        start_frame = int(start / frame_s)
        end_frame = math.ceil(end / frame_s - 1e-9) - 1
        out.append(TurnFrames(start_frame, max(start_frame, end_frame)))
    return out


@dataclass
class TurnTiming:
    index: int
    speech_start: float
    speech_end: float
    t_speech_start_sent: float | None = None
    t_endpoint: float | None = None
    t_first_token: float | None = None
    t_first_partial_after_eos: float | None = None
    t_final: float | None = None
    t_endpoint_ack: float | None = None
    timeout: bool = False
    pre_endpoint_final: bool = False
    premature_turn_end: bool = False
    eos_to_final_ms: float | None = None
    ttft_ms: float | None = None
    eos_to_first_partial_ms: float | None = None


@dataclass
class SessionResult:
    item_id: str
    provider: str
    lang: str
    mode: str
    condition: str
    ok: bool
    hypothesis: str
    error: str | None = None
    setup_ms: float = 0.0
    p99_lateness_ms: float = 0.0
    events: list[dict] = field(default_factory=list)
    turns: list[TurnTiming] = field(default_factory=list)
    n_finals: int = 0
    empty: bool = False


def _event_dict(ev: Event) -> dict:
    return {"kind": ev.kind, "text": ev.text, "t": ev.t, "meta": ev.meta}


async def _drain(provider: StreamingProvider, log: list[Event], stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            ev = await asyncio.wait_for(provider.events.get(), 0.05)
        except asyncio.TimeoutError:
            continue
        log.append(ev)
        if ev.kind == "closed":
            break


def _assign_turn_metrics(turns: list[TurnTiming], events: list[Event], mode: str, supports_forced: bool, has_ack: bool, timeout_s: float) -> None:
    for i, turn in enumerate(turns):
        turn.timeout = False
        t_ss = turn.t_speech_start_sent
        t_ep = turn.t_endpoint
        t_next = turns[i + 1].t_endpoint if i + 1 < len(turns) else None
        window = [e for e in events if (t_ss is None or e.t >= t_ss - 1e-6) and (t_next is None or e.t < t_next)]

        first_tok = next((e for e in window if e.kind in ("partial", "final") and e.text.strip()), None)
        if first_tok and t_ss is not None:
            turn.t_first_token = first_tok.t
            turn.ttft_ms = (first_tok.t - t_ss) * 1000.0

        if any(e.kind == "turn_end" and t_ep is not None and e.t < t_ep for e in window):
            turn.premature_turn_end = True

        if t_ep is None:
            continue
        after = [e for e in events if e.t >= t_ep - 1e-6]
        pre_finals = [
            e for e in events
            if e.kind == "final" and e.text.strip() and (t_ss is None or e.t >= t_ss - 1e-6) and e.t < t_ep
        ]
        ack = next((e for e in after if e.kind == "endpoint"), None)
        first_final = next((e for e in after if e.kind == "final"), None)
        first_partial = next((e for e in after if e.kind == "partial" and e.text.strip()), None)
        native_end = next((e for e in after if e.kind == "turn_end"), None)

        if first_partial:
            turn.t_first_partial_after_eos = first_partial.t
            turn.eos_to_first_partial_ms = (first_partial.t - t_ep) * 1000.0

        stop_ev = None
        if mode == "forced" and supports_forced and has_ack and ack:
            stop_ev = ack
            turn.t_endpoint_ack = ack.t
        elif mode == "forced" and supports_forced and first_final:
            stop_ev = first_final
        elif mode == "native" and native_end:
            stop_ev = native_end
        elif not supports_forced:
            # AA fallback: first natural final within 2s; else first partial after 2s; else latest partial before 2s
            within = [e for e in after if e.t <= t_ep + FALLBACK_S and e.kind in ("final", "turn_end")]
            if within:
                stop_ev = within[0]
            else:
                after2 = [e for e in after if e.t > t_ep + FALLBACK_S and e.kind in ("partial", "final") and e.text.strip()]
                before2 = [e for e in after if e.t <= t_ep + FALLBACK_S and e.kind == "partial" and e.text.strip()]
                stop_ev = after2[0] if after2 else (before2[-1] if before2 else None)

        if stop_ev:
            turn.t_final = stop_ev.t
            turn.eos_to_final_ms = max(0.0, (stop_ev.t - t_ep) * 1000.0)
        elif pre_finals:
            turn.pre_endpoint_final = True
            turn.t_final = pre_finals[-1].t
            turn.eos_to_final_ms = 0.0
        else:
            last_t = events[-1].t if events else t_ep
            if last_t - t_ep >= timeout_s - 0.05:
                turn.timeout = True


async def run_item(
    provider: StreamingProvider,
    ds: Dataset,
    item: Item,
    *,
    mode: str,
    condition: str,
    timeout_s: float,
    silence_tail_s: float,
    frame_ms: int,
) -> SessionResult:
    pcm16, file_sr = read_pcm16(ds.audio_path(item))
    rates = list(provider.sample_rates)
    stream_bytes, stream_sr = prepare_stream_audio(pcm16, condition, rates)
    ctx = SessionContext(
        lang=item.lang,
        sample_rate=stream_sr,
        mode=mode,
        condition=condition,
        oracle_text=item.reference,
        oracle_turns=item.turns,
    )
    turns = [TurnTiming(i, a, b) for i, (a, b) in enumerate(item.turns or [[item.speech_start, item.speech_end]])]
    events: list[Event] = []
    stop = asyncio.Event()
    t_connect = time.perf_counter()
    try:
        await provider.connect(ctx)
    except Exception as exc:  # noqa: BLE001
        return SessionResult(
            item_id=item.id, provider=provider.name, lang=item.lang, mode=mode, condition=condition,
            ok=False, hypothesis="", error=f"connect: {exc}", setup_ms=(time.perf_counter() - t_connect) * 1000.0,
        )
    setup_ms = (time.perf_counter() - t_connect) * 1000.0
    collector = asyncio.create_task(_drain(provider, events, stop))
    pacer = Pacer(stream_bytes, stream_sr, frame_ms=frame_ms)
    error: str | None = None
    def _refresh(now: float) -> None:
        _assign_turn_metrics(turns, events, mode, provider.supports_forced, provider.has_endpoint_ack, timeout_s)
        for tr in turns:
            if tr.t_endpoint is not None and tr.t_final is None and now - tr.t_endpoint > timeout_s:
                tr.timeout = True

    try:
        turn_i = 0
        begun = False
        last_speech_end = turns[-1].speech_end if turns else 0.0
        max_audio_s = last_speech_end + silence_tail_s
        while pacer.audio_time_s() < max_audio_s + 1e-6:
            _refresh(time.perf_counter())
            at = pacer.audio_time_s()
            if turns and all(t.t_final is not None or t.timeout for t in turns) and at >= last_speech_end:
                break
            if turn_i < len(turns) and not begun and at >= turns[turn_i].speech_start - 1e-4:
                await provider.begin_turn()
                begun = True
            chunk, sent = await pacer.next_frame(silence=pacer.i >= pacer.n_frames)
            await provider.send_audio(chunk)
            at = pacer.audio_time_s()
            if turn_i < len(turns):
                tr = turns[turn_i]
                if tr.t_speech_start_sent is None and at >= tr.speech_start:
                    tr.t_speech_start_sent = sent
                if tr.t_endpoint is None and at >= tr.speech_end:
                    if mode == "forced" and provider.supports_forced:
                        await provider.force_endpoint()
                        tr.t_endpoint = time.perf_counter()
                    else:
                        tr.t_endpoint = sent
                    begun = False
                    turn_i += 1

        await provider.finish()
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            _refresh(time.perf_counter())
            if all(t.t_final is not None or t.timeout for t in turns):
                await asyncio.sleep(0.02)
                break
            await asyncio.sleep(0.02)
    except ProviderError as exc:
        error = str(exc)
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    finally:
        try:
            await provider.close()
        except Exception:  # noqa: BLE001
            pass
        stop.set()
        try:
            await asyncio.wait_for(collector, 2)
        except Exception:  # noqa: BLE001
            collector.cancel()

    now = time.perf_counter()
    _assign_turn_metrics(turns, events, mode, provider.supports_forced, provider.has_endpoint_ack, timeout_s)
    for tr in turns:
        if tr.t_endpoint is not None and tr.t_final is None and now - tr.t_endpoint > timeout_s:
            tr.timeout = True
    finals = [e.text.strip() for e in events if e.kind == "final" and e.text.strip()]
    hypothesis = " ".join(finals)
    err_ev = next((e for e in events if e.kind == "error"), None)
    if err_ev and not error:
        error = err_ev.text
    ok = error is None and bool(hypothesis) and not all(t.timeout for t in turns)
    return SessionResult(
        item_id=item.id,
        provider=provider.name,
        lang=item.lang,
        mode=mode,
        condition=condition,
        ok=ok,
        hypothesis=hypothesis,
        error=error,
        setup_ms=setup_ms,
        p99_lateness_ms=pacer.p99_lateness_ms(),
        events=[_event_dict(e) for e in events],
        turns=turns,
        n_finals=len(finals),
        empty=not bool(hypothesis),
    )


def result_to_dict(r: SessionResult) -> dict:
    d = asdict(r)
    return d
