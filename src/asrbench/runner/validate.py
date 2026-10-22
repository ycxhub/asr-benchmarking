"""Harness self-test: inject known server delays with the mock provider and check the measured latency recovers them."""
from __future__ import annotations

import asyncio
import statistics

import numpy as np

from .. import audio as A
from ..providers.base import SessionContext
from ..providers.mock import MockProvider
from .session import SessionConfig, run_session

TOLERANCE_MS = 5.0


def _synthetic(turns: list[list[float]], duration: float, sr: int = 16000) -> bytes:
    rng = np.random.default_rng(0)
    x = np.zeros(int(duration * sr), dtype=np.float32)
    for s, e in turns:
        n = int((e - s) * sr)
        t = np.arange(n) / sr
        x[int(s * sr):int(s * sr) + n] = 0.3 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)) + 0.02 * rng.standard_normal(n)
    return A.to_pcm16(x)


async def _one(delay_ms: float, mode: str, turns, duration, final_only=False) -> list[dict]:
    params = {"finalize_delay_ms": delay_ms, "native_delay_ms": delay_ms, "partial_delay_ms": 100, "first_token_ms": 300,
              "native_silence_ms": 500, "jitter_ms": 0, "wer_noise": 0, "final_only": final_only}
    prov = MockProvider("mock-validate", {"params": params, "lang_map": {"en-US": {}}})
    ctx = SessionContext(lang="en-US", sample_rate=16000, mode=mode, oracle_text="one two three four five six seven eight", oracle_turns=turns)
    res = await run_session(prov, _synthetic(turns, duration), 16000, turns, ctx, SessionConfig(mode=mode, keep_events=False), "synthetic")
    if res.status != "ok":
        raise RuntimeError(res.error)
    return [dict(t, pacing_p99_ms=res.pacing_ms.get("p99")) for t in res.turns]


async def validate(delays_ms=(0, 100, 300, 1000), repeats: int = 3, concurrency: int = 4) -> dict:
    """Returns per-delay measured medians and errors for forced and native mode."""
    turns_single = [[0.5, 2.5]]
    turns_multi = [[0.5, 2.0], [3.0, 4.6]]
    rows = []
    for mode in ("forced", "native"):
        for d in delays_ms:
            jobs = []
            for r in range(repeats):
                turns, dur = (turns_single, 3.2) if r % 2 == 0 else (turns_multi, 5.2)
                jobs.append(_one(d, mode, turns, dur))
            # run several sessions concurrently to include event-loop contention in the check
            results = []
            for i in range(0, len(jobs), concurrency):
                results += await asyncio.gather(*jobs[i:i + concurrency])
            turns_out = [t for res in results for t in res]
            measured = [t["eos_to_final_ms"] for t in turns_out if t["eos_to_final_ms"] is not None]
            # native: provider waits native_silence (500 ms of audio after speech end) before deciding
            expected = d if mode == "forced" else d + 500
            frame_slack = 0 if mode == "forced" else 20
            med = statistics.median(measured) if measured else None
            err = None if med is None else med - expected
            ttft = [t["ttft_ms"] for t in turns_out if t["ttft_ms"] is not None]
            rows.append({
                "mode": mode, "injected_ms": d, "expected_ms": expected, "measured_p50_ms": None if med is None else round(med, 2),
                "error_ms": None if err is None else round(err, 2),
                "max_abs_error_ms": None if not measured else round(max(abs(m - expected) for m in measured), 2),
                "ttft_p50_ms": round(statistics.median(ttft), 1) if ttft else None, "ttft_expected_ms": 400,
                "n_turns": len(turns_out), "resolved": len(measured),
                "pass": err is not None and abs(err) <= TOLERANCE_MS + frame_slack and len(measured) == len(turns_out),
            })
    return {"tolerance_ms": TOLERANCE_MS, "rows": rows, "pass": all(r["pass"] for r in rows)}


def validate_sync(**kw) -> dict:
    return asyncio.run(validate(**kw))
