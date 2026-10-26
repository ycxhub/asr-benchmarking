from __future__ import annotations

import csv
import json
import math
import random
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import GatedRepoError

from .. import audio as A
from .. import settings
from .ingest import DatasetWriter

Progress = Callable[[str], None]

TEXT_COLS = ["text", "transcript", "transcription", "sentence", "reference", "normalized", "verbatim"]
AUDIO_COLS = ["audio", "audio_filepath", "audio_path", "file", "path"]
ID_COLS = ["id", "sample_id", "segment_id", "utt_id", "file_name", "filename"]
SPEAKER_COLS = ["speaker_id", "speaker", "client_id"]


def _noop(_: str) -> None:
    pass


def _require_token(spec: dict) -> None:
    access = str(spec.get("access") or "")
    if "login" in access.lower() and not settings.env("HF_TOKEN"):
        raise RuntimeError(
            f"{spec.get('name')} requires HF_TOKEN. Open the dataset on Hugging Face, accept its terms, "
            "then add a read token to .env."
        )


def _download(repo_id: str, filename: str, *, revision: str | None = None, token: str | None = None) -> str:
    try:
        return hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision, token=token)
    except GatedRepoError as exc:
        raise RuntimeError(
            f"Hugging Face denied {repo_id}. Logged in as the token owner, open "
            f"https://huggingface.co/datasets/{repo_id} and click Agree, then retry."
        ) from exc


def _as_list(v) -> list:
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _first_present(row: dict, candidates: list[str]):
    for c in candidates:
        if c in row and row[c] not in (None, ""):
            return c, row[c]
    return None, None


def _meta(spec: dict, source_type: str) -> dict:
    return {
        "source": {"type": source_type, **{k: spec[k] for k in ("hf_id", "config", "split", "revision") if spec.get(k)}},
        "license": spec.get("license", "unknown"),
        "public": bool(spec.get("public", True)),
        "description": spec.get("description", ""),
        "selection": {"seed": spec.get("seed", 1234), "sample": spec.get("sample"), "min_s": spec.get("min_s", 1.0), "max_s": spec.get("max_s", 60.0)},
        "cmi_col": spec.get("cmi_col"),
    }


def _row_tags(row: dict, spec: dict) -> dict:
    tags = {}
    for col in _as_list(spec.get("tag_cols")):
        if col in row and row[col] is not None and not isinstance(row[col], (bytes, dict, list)):
            tags[col] = row[col]
    if spec.get("cmi_col") and row.get(spec["cmi_col"]) is not None:
        try:
            tags["cmi"] = float(row[spec["cmi_col"]])
        except (TypeError, ValueError):
            pass
    return tags


def _decode_audio_cell(cell, repo_id: str, revision: str | None, token: str | None):
    if isinstance(cell, dict):
        if cell.get("bytes"):
            return A.decode_bytes(cell["bytes"])
        if cell.get("array") is not None:
            import numpy as np

            return np.asarray(cell["array"], dtype="float32"), int(cell["sampling_rate"])
        cell = cell.get("path")
    if isinstance(cell, str) and cell:
        local = _download(repo_id, cell, revision=revision, token=token)
        return A.decode_file(local)
    raise ValueError("unsupported audio cell")


def _find_parquet_files(api: HfApi, repo_id: str, config: str | None, split: str, token: str | None) -> tuple[list[str], str | None]:
    split_re = re.compile(rf"(^|[/_-]){re.escape(split)}([/_.-]|$)")
    for revision in (None, "refs/convert/parquet"):
        try:
            files = api.list_repo_files(repo_id, repo_type="dataset", revision=revision, token=token)
        except Exception:  # noqa: BLE001
            continue
        parquet = [f for f in files if f.endswith(".parquet")]
        if config:
            scoped = [f for f in parquet if f.split("/")[0] == config or f"/{config}/" in f"/{f}"]
            parquet = scoped or parquet
        matched = sorted(f for f in parquet if split_re.search(f))
        if matched:
            return matched, revision
    raise FileNotFoundError(f"no parquet files for {repo_id} config={config} split={split}. Is the dataset gated (set HF_TOKEN and accept its terms)?")


def import_hf_parquet(spec: dict, progress: Progress = _noop) -> dict:
    """Seeded sample of a Hugging Face dataset stored as parquet with an audio column."""
    _require_token(spec)
    token = settings.env("HF_TOKEN")
    repo_id, split = spec["hf_id"], spec.get("split", "test")
    seed, n_target = int(spec.get("seed", 1234)), spec.get("sample")
    api = HfApi(token=token)
    files, revision = _find_parquet_files(api, repo_id, spec.get("config"), split, token)
    rng = random.Random(seed)
    rng.shuffle(files)
    max_shards = int(spec.get("max_shards", 8))
    per_shard = math.ceil(n_target / min(len(files), max_shards)) if n_target else None

    writer = DatasetWriter(spec["name"], spec["lang"], _meta(spec, "huggingface-parquet"), min_s=float(spec.get("min_s", 1.0)), max_s=float(spec.get("max_s", 60.0)))
    text_cols = _as_list(spec.get("text_col")) or TEXT_COLS
    audio_cols = _as_list(spec.get("audio_col")) or AUDIO_COLS
    id_cols = _as_list(spec.get("id_col")) or ID_COLS
    spk_cols = _as_list(spec.get("speaker_col")) or SPEAKER_COLS

    for shard_idx, fname in enumerate(files):
        if n_target and len(writer.items) >= n_target:
            break
        progress(f"downloading {repo_id}:{fname}")
        local = _download(repo_id, fname, revision=revision, token=token)
        pf = pq.ParquetFile(local)
        table = pf.read()
        rows = list(range(table.num_rows))
        rng.shuffle(rows)
        taken = 0
        cols = table.column_names
        for r in rows:
            if n_target and (len(writer.items) >= n_target or (per_shard and taken >= per_shard and shard_idx < max_shards - 1)):
                break
            row = {c: table.column(c)[r].as_py() for c in cols}
            _, text = _first_present(row, text_cols)
            acol, cell = _first_present(row, audio_cols)
            raw_id = _first_present(row, id_cols)[1] or f"{Path(fname).stem}_{r}"
            if cell is None:
                writer.reject(str(raw_id), "no audio column")
                continue
            try:
                samples, sr = _decode_audio_cell(cell, repo_id, revision, token)
            except Exception as exc:  # noqa: BLE001
                writer.reject(str(raw_id), f"decode error: {exc}")
                continue
            if writer.add(str(raw_id), samples, sr, text or "", speaker=_first_present(row, spk_cols)[1], tags=_row_tags(row, spec)):
                taken += 1
                if len(writer.items) % 25 == 0:
                    progress(f"{spec['name']}: {len(writer.items)} items")
    return writer.finalize()


def import_hf_jsonl_files(spec: dict, progress: Progress = _noop) -> dict:
    """Datasets published as a JSONL manifest plus audio files in the repo (e.g. Artificial Analysis sets)."""
    _require_token(spec)
    token = settings.env("HF_TOKEN")
    repo_id = spec["hf_id"]
    local_manifest = _download(repo_id, spec["manifest_file"], token=token)
    rows = [json.loads(line) for line in Path(local_manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(int(spec.get("seed", 1234)))
    rng.shuffle(rows)
    n_target = spec.get("sample")
    writer = DatasetWriter(spec["name"], spec["lang"], _meta(spec, "huggingface-files"), min_s=float(spec.get("min_s", 1.0)), max_s=float(spec.get("max_s", 60.0)))
    audio_key, text_key = spec.get("audio_path_col", "url"), spec.get("text_col", "transcript")
    for row in rows:
        if n_target and len(writer.items) >= n_target:
            break
        raw_id = row.get(spec.get("id_col", "id")) or row.get(audio_key)
        try:
            local = _download(repo_id, row[audio_key], token=token)
            samples, sr = A.decode_file(local)
        except Exception as exc:  # noqa: BLE001
            writer.reject(str(raw_id), f"download/decode error: {exc}")
            continue
        writer.add(str(raw_id), samples, sr, row.get(text_key, ""), speaker=row.get(spec.get("speaker_col", "speaker")), tags=_row_tags(row, spec))
        progress(f"{spec['name']}: {len(writer.items)} items")
    return writer.finalize()


def _read_local_manifest(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    delim = "\t" if path.suffix == ".tsv" else ","
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=delim))


def find_local_manifest(root: Path) -> Path:
    for name in ("manifest.jsonl", "manifest.csv", "manifest.tsv", "metadata.csv", "metadata.jsonl"):
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    raise FileNotFoundError("upload must contain manifest.jsonl / manifest.csv / manifest.tsv with columns: audio, text[, id, speaker, ...]")


def import_folder(source: str | Path, name: str, lang: str, license: str = "user-provided", public: bool = False,
                  min_s: float = 0.5, max_s: float = 60.0, progress: Progress = _noop) -> dict:
    """Import a local folder or .zip containing a manifest and audio files."""
    source = Path(source).expanduser()
    tmp = None
    if source.suffix == ".zip":
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(source) as zf:
            zf.extractall(tmp.name)
        root = Path(tmp.name)
    else:
        root = source
    try:
        manifest = find_local_manifest(root)
        rows = _read_local_manifest(manifest)
        meta = {"source": {"type": "upload", "path": str(source)}, "license": license, "public": public,
                "selection": {"sample": None, "min_s": min_s, "max_s": max_s}}
        writer = DatasetWriter(name, lang, meta, min_s=min_s, max_s=max_s)
        reserved = set(AUDIO_COLS + TEXT_COLS + ID_COLS + SPEAKER_COLS + ["lang", "language"])
        for i, row in enumerate(rows):
            acol, rel = _first_present(row, AUDIO_COLS)
            _, text = _first_present(row, TEXT_COLS)
            raw_id = _first_present(row, ID_COLS)[1] or (Path(rel).stem if rel else f"row{i}")
            row_lang = row.get("lang") or row.get("language")
            if row_lang and row_lang != lang:
                writer.reject(str(raw_id), f"language {row_lang} != dataset language {lang}")
                continue
            if not rel:
                writer.reject(str(raw_id), "missing audio path")
                continue
            apath = (manifest.parent / rel) if not Path(rel).is_absolute() else Path(rel)
            try:
                samples, sr = A.decode_file(apath)
            except Exception as exc:  # noqa: BLE001
                writer.reject(str(raw_id), f"decode error: {exc}")
                continue
            tags = {k: v for k, v in row.items() if k not in reserved and v not in (None, "")}
            if "cmi" in tags:
                try:
                    tags["cmi"] = float(tags["cmi"])
                except ValueError:
                    tags.pop("cmi")
            writer.add(str(raw_id), samples, sr, text or "", speaker=_first_present(row, SPEAKER_COLS)[1], tags=tags)
            if i % 25 == 0:
                progress(f"{name}: {len(writer.items)} items")
        return writer.finalize()
    finally:
        if tmp is not None:
            tmp.cleanup()


_PHRASES = [
    "transfer five lakh to the savings account",
    "the payment failed please try again",
    "what is my account balance today",
    "schedule a call with the insurance desk",
]


def import_synthetic(spec: dict, progress: Progress = _noop) -> dict:
    """Local tones with known transcripts. No network and no API key."""
    import numpy as np

    n = int(spec.get("sample") or 4)
    sr = 16000
    writer = DatasetWriter(
        spec["name"],
        spec["lang"],
        {"source": {"type": "synthetic"}, "license": spec.get("license", "generated"), "public": False,
         "description": spec.get("description", ""), "selection": {"sample": n}},
        min_s=float(spec.get("min_s", 0.5)),
        max_s=float(spec.get("max_s", 5.0)),
    )
    for i in range(n):
        dur = 1.2
        audio = np.zeros(int(dur * sr), dtype=np.float32)
        start, end = int(0.25 * sr), int(0.95 * sr)
        t = np.arange(end - start) / sr
        audio[start:end] = 0.45 * np.sin(2 * np.pi * (180 + 30 * (i % 4)) * t)
        writer.add(f"smoke-{i:03d}", audio, sr, _PHRASES[i % len(_PHRASES)], speaker=f"spk-{i % 2}")
        progress(f"{spec['name']}: {len(writer.items)} items")
    return writer.finalize()


IMPORTERS = {
    "hf_parquet": import_hf_parquet,
    "hf_jsonl_files": import_hf_jsonl_files,
    "synthetic": import_synthetic,
}


def import_from_spec(spec: dict, progress: Progress = _noop) -> dict:
    kind = spec.get("importer", "hf_parquet")
    if kind not in IMPORTERS:
        raise ValueError(f"unknown importer {kind}")
    return IMPORTERS[kind](spec, progress)
