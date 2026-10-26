"""Harness self-test: inject known server delays with the mock provider and check the measured latency recovers them."""
from __future__ import annotations

import asyncio
import statistics
import tempfile
from pathlib import Path

import numpy as np

from .. import audio as A
from ..datasets.manifest import Dataset, Item
from ..providers.mock import MockProvider
from .session import run_item

TOLERANCE_MS = 5.0


def _synthetic(turns: list[list[float]], duration: float, sr: int = 16000) -> bytes:
    rng = np.random.default_rng(0)
    x = np.zeros(int(duration * sr), dtype=np.float32)
    for s, e in turns:
        n = int((e - s) * sr)
        t = np.arange(n) / sr
        x[int(s * sr) : int(s * sr) + n] = 0.3 * np.sin(2 * np.pi * 220 * t)
    return A.to_pcm16(x)


def _dataset(root: Path, turns: list[list[float]], duration: float) -> tuple[Dataset, Item]:
    audio = root / "audio" / "clip.wav"
    A.write_wav(audio, A.from_pcm16(_synthetic(turns, duration)))
    item = Item(
        id="clip",
        audio="audio/clip.wav",
        reference="one two three four five six seven eight",
        lang="en-US",
        duration_s=duration,
        sha256=A.sha256_file(audio),
        speech=turns,
        turns=turns,
    )
    return Dataset(name="synthetic", path=root, meta={"lang": "en-US"}, items=[item]), item


async def _one(delay_ms: float, mode: str, turns, duration) -> list[dict]:
    params = {
        "finalize_delay_ms": delay_ms,
        "native_delay_ms": delay_ms,
        "partial_delay_ms": 40,
        "native_silence_ms": 200,
        "final_only": False,
    }
    prov = MockProvider("mock-validate", {"params": params, "lang_map": {"en-US": {}}})
    with tempfile.TemporaryDirectory() as raw:
        ds, item = _dataset(Path(raw), turns, duration)
        res = await run_item(
            prov,
            ds,
            item,
            mode=mode,
            condition="wideband",
            timeout_s=max(2.0, delay_ms / 1000.0 + 1.5),
            silence_tail_s=0.35 if mode == "forced" else 0.45,
            frame_ms=20,
        )
    if not res.ok:
        raise RuntimeError(res.error or "session failed")
    return [
        {
            "eos_to_final_ms": t.eos_to_final_ms,
            "ttft_ms": t.ttft_ms,
            "timeout": t.timeout,
            "pacing_p99_ms": res.p99_lateness_ms,
        }
        for t in res.turns
    ]


async def validate(delays_ms=(0, 300), repeats: int = 2, concurrency: int = 1) -> dict:
    """Return per-delay measured medians for forced and native mode.

    Native mode adds the mock's 200 ms of trailing silence before it finalizes,
    which is how a vendor turn detector behaves. Forced mode must recover the
    injected delay within 5 ms.
    """
    turns_single = [[0.15, 0.45]]
    turns_multi = [[0.10, 0.35], [0.60, 0.90]]
    rows = []
    native_silence_ms = 200
    for mode in ("forced", "native"):
        for d in delays_ms:
            jobs = []
            for r in range(repeats):
                turns, dur = (turns_single, 0.7) if r % 2 == 0 else (turns_multi, 1.15)
                jobs.append(_one(d, mode, turns, dur))
            results = []
            for i in range(0, len(jobs), concurrency):
                results += await asyncio.gather(*jobs[i : i + concurrency])
            turns_out = [t for res in results for t in res]
            measured = [t["eos_to_final_ms"] for t in turns_out if t["eos_to_final_ms"] is not None]
            expected = d if mode == "forced" else d + native_silence_ms
            frame_slack = 0 if mode == "forced" else 20
            med = statistics.median(measured) if measured else None
            err = None if med is None else med - expected
            rows.append({
                "mode": mode,
                "injected_ms": d,
                "expected_ms": expected,
                "measured_p50_ms": None if med is None else round(med, 2),
                "error_ms": None if err is None else round(err, 2),
                "max_abs_error_ms": None if not measured else round(max(abs(m - expected) for m in measured), 2),
                "n_turns": len(turns_out),
                "resolved": len(measured),
                "pass": err is not None and abs(err) <= TOLERANCE_MS + frame_slack and len(measured) == len(turns_out),
            })
    return {"tolerance_ms": TOLERANCE_MS, "rows": rows, "pass": all(r["pass"] for r in rows)}


def validate_sync(**kw) -> dict:
    return asyncio.run(validate(**kw))
