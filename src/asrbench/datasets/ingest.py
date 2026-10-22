from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import audio as A
from .. import settings, vad
from .manifest import Item, dataset_dir, manifest_sha256, write_manifest

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_id(raw: str) -> str:
    s = _SAFE.sub("_", str(raw)).strip("._")
    return s[:120] or "item"


class DatasetWriter:
    """Validates, canonicalizes (16 kHz mono PCM16), VAD-segments and records items."""

    def __init__(self, name: str, lang: str, meta: dict, min_s: float = 0.5, max_s: float = 60.0, overwrite: bool = True):
        if lang not in settings.LANGUAGES:
            raise ValueError(f"lang must be one of {settings.LANGUAGES}, got {lang!r}")
        self.name, self.lang, self.meta = name, lang, dict(meta)
        self.min_s, self.max_s = min_s, max_s
        self.dir = dataset_dir(name)
        if overwrite and self.dir.exists():
            shutil.rmtree(self.dir)
        (self.dir / "audio").mkdir(parents=True, exist_ok=True)
        self.items: list[Item] = []
        self.rejects: list[dict] = []
        self.vad_methods: set[str] = set()
        self._ids: set[str] = set()

    def reject(self, raw_id: str, reason: str) -> None:
        self.rejects.append({"id": str(raw_id), "reason": reason})

    def add(self, raw_id: str, samples: np.ndarray, sr: int, reference: str, speaker: str | None = None, tags: dict | None = None) -> bool:
        reference = " ".join(str(reference or "").split())
        if not reference:
            self.reject(raw_id, "empty reference")
            return False
        if samples is None or len(samples) == 0:
            self.reject(raw_id, "empty audio")
            return False
        audio16, qinfo = A.canonicalize(samples, sr)
        duration = len(audio16) / A.CANONICAL_SR
        if duration < self.min_s or duration > self.max_s:
            self.reject(raw_id, f"duration {duration:.2f}s outside [{self.min_s}, {self.max_s}]")
            return False
        item_id = safe_id(raw_id)
        base, n = item_id, 1
        while item_id in self._ids:
            n += 1
            item_id = f"{base}_{n}"
        self._ids.add(item_id)

        segments, method = vad.detect_speech(audio16)
        self.vad_methods.add(method)
        speech_found = bool(segments)
        turns = vad.plan_turns(segments, duration)
        rel = f"audio/{item_id}.wav"
        path = self.dir / rel
        A.write_wav(path, audio16)
        all_tags = dict(tags or {})
        all_tags.update({k: v for k, v in qinfo.items() if k != "source_sample_rate"})
        all_tags["source_sample_rate"] = qinfo["source_sample_rate"]
        if not speech_found:
            all_tags["vad_no_speech"] = True
        self.items.append(
            Item(
                id=item_id,
                audio=rel,
                reference=reference,
                lang=self.lang,
                duration_s=round(duration, 3),
                sha256=A.sha256_file(path),
                speech=[[round(s.start, 3), round(s.end, 3)] for s in segments],
                turns=[[round(t.start, 3), round(t.end, 3)] for t in turns],
                speaker=str(speaker) if speaker not in (None, "") else None,
                tags=all_tags,
            )
        )
        return True

    def finalize(self) -> dict:
        if not self.items:
            raise ValueError(f"no valid items for dataset '{self.name}' ({len(self.rejects)} rejected)")
        manifest = self.dir / "manifest.jsonl"
        write_manifest(manifest, self.items)
        if self.rejects:
            with open(self.dir / "rejects.jsonl", "w", encoding="utf-8") as f:
                for r in self.rejects:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        total = sum(i.duration_s for i in self.items)
        speech = sum(e - s for i in self.items for s, e in i.speech)
        meta = {
            **self.meta,
            "name": self.name,
            "lang": self.lang,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_items": len(self.items),
            "n_rejected": len(self.rejects),
            "total_audio_s": round(total, 1),
            "total_speech_s": round(speech, 1),
            "n_multi_turn": sum(1 for i in self.items if len(i.turns) > 1),
            "vad_method": ",".join(sorted(self.vad_methods)),
            "manifest_sha256": manifest_sha256(manifest),
            "audio_format": "wav pcm_s16le 16000 Hz mono",
        }
        (self.dir / "dataset.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return meta


def verify_dataset_audio(ds_dir: Path, items: list[Item]) -> list[str]:
    """Return ids whose audio is missing or whose SHA-256 does not match the manifest."""
    bad = []
    for it in items:
        p = ds_dir / it.audio
        if not p.exists() or A.sha256_file(p) != it.sha256:
            bad.append(it.id)
    return bad
