from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import settings


@dataclass
class Item:
    id: str
    audio: str
    reference: str
    lang: str
    duration_s: float
    sha256: str
    speech: list[list[float]]
    turns: list[list[float]]
    speaker: str | None = None
    tags: dict = field(default_factory=dict)

    @property
    def speech_start(self) -> float:
        return self.turns[0][0]

    @property
    def speech_end(self) -> float:
        return self.turns[-1][1]


@dataclass
class Dataset:
    name: str
    path: Path
    meta: dict
    items: list[Item]

    @property
    def lang(self) -> str:
        return self.meta["lang"]

    def audio_path(self, item: Item) -> Path:
        return self.path / item.audio


def dataset_dir(name: str) -> Path:
    return settings.datasets_dir() / name


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(name: str) -> Dataset:
    d = dataset_dir(name)
    meta_path = d / "dataset.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"dataset '{name}' not found in {settings.datasets_dir()}")
    meta = json.loads(meta_path.read_text())
    items = []
    with open(d / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(Item(**json.loads(line)))
    return Dataset(name=name, path=d, meta=meta, items=items)


def list_datasets() -> list[dict]:
    out = []
    for meta_path in sorted(settings.datasets_dir().glob("*/dataset.json")):
        try:
            out.append(json.loads(meta_path.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def write_manifest(path: Path, items: list[Item]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
