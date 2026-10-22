from __future__ import annotations

import importlib.util
import logging
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import settings

log = logging.getLogger(__name__)

SILERO_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
SR = 16000
WINDOW = 512
CONTEXT = 64

THRESHOLD = 0.5
NEG_THRESHOLD = 0.35
MIN_SPEECH_MS = 250
MIN_SILENCE_MS = 100
SPEECH_PAD_MS = 30

MAX_TURN_S = 30.0
MIN_TURN_PAUSE_S = 0.5


@dataclass
class Segment:
    start: float
    end: float


def _silero_model_path() -> Path | None:
    cached = settings.models_dir() / "silero_vad.onnx"
    if cached.exists():
        return cached
    spec = importlib.util.find_spec("silero_vad")
    if spec and spec.submodule_search_locations:
        candidate = Path(list(spec.submodule_search_locations)[0]) / "data" / "silero_vad.onnx"
        if candidate.exists():
            return candidate
    try:
        log.info("Downloading Silero VAD model to %s", cached)
        urllib.request.urlretrieve(SILERO_URL, cached)
        return cached
    except Exception as exc:  # noqa: BLE001
        log.warning("Silero VAD unavailable (%s); falling back to energy VAD", exc)
        return None


@lru_cache(maxsize=1)
def _silero_session():
    path = _silero_model_path()
    if path is None:
        return None
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def silero_probs(audio: np.ndarray) -> np.ndarray | None:
    sess = _silero_session()
    if sess is None:
        return None
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, CONTEXT), dtype=np.float32)
    sr = np.array(SR, dtype=np.int64)
    n = int(np.ceil(len(audio) / WINDOW))
    padded = np.zeros(n * WINDOW, dtype=np.float32)
    padded[: len(audio)] = audio
    probs = np.empty(n, dtype=np.float32)
    for i in range(n):
        chunk = padded[i * WINDOW : (i + 1) * WINDOW][None, :]
        x = np.concatenate([context, chunk], axis=1)
        out, state = sess.run(None, {"input": x, "state": state, "sr": sr})
        probs[i] = float(out[0][0])
        context = x[:, -CONTEXT:]
    return probs


def _segments_from_probs(probs: np.ndarray, n_samples: int, window: int = WINDOW) -> list[Segment]:
    min_speech = MIN_SPEECH_MS * SR / 1000
    min_silence = MIN_SILENCE_MS * SR / 1000
    pad = SPEECH_PAD_MS * SR / 1000
    triggered, temp_end, start = False, 0, 0
    raw: list[tuple[int, int]] = []
    for i, p in enumerate(probs):
        pos = i * window
        if p >= THRESHOLD and temp_end:
            temp_end = 0
        if p >= THRESHOLD and not triggered:
            triggered, start = True, pos
            continue
        if p < NEG_THRESHOLD and triggered:
            if not temp_end:
                temp_end = pos
            if pos - temp_end < min_silence:
                continue
            if temp_end - start > min_speech:
                raw.append((start, temp_end))
            triggered, temp_end = False, 0
    if triggered and n_samples - start > min_speech:
        raw.append((start, n_samples))
    segs = [Segment(max(0.0, (s - pad) / SR), min(n_samples / SR, (e + pad) / SR)) for s, e in raw]
    merged: list[Segment] = []
    for s in segs:
        if merged and s.start <= merged[-1].end:
            merged[-1].end = max(merged[-1].end, s.end)
        else:
            merged.append(s)
    return merged


def energy_segments(audio: np.ndarray) -> list[Segment]:
    win = int(0.03 * SR)
    n = len(audio) // win
    if n == 0:
        return []
    frames = audio[: n * win].reshape(n, win)
    db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-9)
    floor = np.percentile(db, 10)
    active = db > max(floor + 12.0, -50.0)
    probs = np.where(active, 1.0, 0.0).astype(np.float32)
    return _segments_from_probs(probs, len(audio), window=win)


def detect_speech(audio16k: np.ndarray) -> tuple[list[Segment], str]:
    probs = silero_probs(audio16k)
    if probs is not None:
        return _segments_from_probs(probs, len(audio16k)), "silero-v5"
    return energy_segments(audio16k), "energy"


def plan_turns(segments: list[Segment], duration: float) -> list[Segment]:
    """One turn per clip; clips with >MAX_TURN_S of speech are split at pauses >= MIN_TURN_PAUSE_S."""
    if not segments:
        return [Segment(0.0, duration)]
    turns: list[Segment] = []
    cur = Segment(segments[0].start, segments[0].end)
    for prev, seg in zip(segments, segments[1:]):
        gap = seg.start - prev.end
        if seg.end - cur.start > MAX_TURN_S and gap >= MIN_TURN_PAUSE_S:
            turns.append(cur)
            cur = Segment(seg.start, seg.end)
        else:
            cur.end = seg.end
    turns.append(cur)
    return turns
