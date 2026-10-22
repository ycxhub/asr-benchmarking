from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

CANONICAL_SR = 16000


def decode_bytes(data: bytes) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr


def decode_file(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32, copy=False)
    return soxr.resample(audio, sr_in, sr_out, quality="HQ").astype(np.float32)


def canonicalize(audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
    """Mono float32 at 16 kHz. Returns audio and quality info (no loudness normalization)."""
    out = resample(audio, sr, CANONICAL_SR)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    clipped = float(np.mean(np.abs(out) >= 0.999)) if out.size else 0.0
    if peak > 1.0:
        out = out / peak
    info = {"source_sample_rate": sr, "peak": round(peak, 4), "clipped_fraction": round(clipped, 6)}
    return out, info


def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def from_pcm16(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def write_wav(path: Path, audio: np.ndarray, sr: int = CANONICAL_SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def read_pcm16(path: str | Path) -> tuple[bytes, int]:
    data, sr = sf.read(str(path), dtype="int16", always_2d=True)
    return data[:, 0].astype("<i2").tobytes(), sr


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --- G.711 mu-law (audioop was removed in Python 3.13) ---

_MU = 255.0


def mulaw_encode(audio: np.ndarray) -> np.ndarray:
    x = np.clip(audio, -1.0, 1.0)
    y = np.sign(x) * np.log1p(_MU * np.abs(x)) / np.log1p(_MU)
    return np.round((y + 1.0) / 2.0 * _MU).astype(np.uint8)


def mulaw_decode(codes: np.ndarray) -> np.ndarray:
    y = codes.astype(np.float32) / _MU * 2.0 - 1.0
    return (np.sign(y) * ((1.0 + _MU) ** np.abs(y) - 1.0) / _MU).astype(np.float32)


def telephony_degrade(audio16k: np.ndarray) -> np.ndarray:
    """16 kHz -> 8 kHz -> mu-law round trip. Returns 8 kHz float32."""
    a8 = resample(audio16k, CANONICAL_SR, 8000)
    return mulaw_decode(mulaw_encode(a8))


def prepare_stream_audio(pcm16_16k: bytes, condition: str, provider_rates: list[int]) -> tuple[bytes, int]:
    """Return (pcm16 bytes, sample_rate) to stream for a condition and provider."""
    if condition == "wideband":
        return pcm16_16k, CANONICAL_SR
    if condition == "telephony":
        a8 = telephony_degrade(from_pcm16(pcm16_16k))
        if 8000 in provider_rates:
            return to_pcm16(a8), 8000
        return to_pcm16(resample(a8, 8000, CANONICAL_SR)), CANONICAL_SR
    raise ValueError(f"unknown condition: {condition}")
