from __future__ import annotations

import asyncio
import hashlib
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from typing import Callable

from .. import settings
from ..datasets.ingest import verify_dataset_audio
from ..datasets.manifest import load_dataset
from ..metrics.normalize import NORM_VERSION
from ..providers import load_provider, snapshot_configs
from ..providers.base import ProviderError
from ..store import upsert_run
from .network import baseline_for_hosts
from .session import SessionResult, result_to_dict, run_item

Progress = Callable[[str], None]


def _provenance() -> dict:
    root = settings.PROJECT_ROOT
    commit = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    lock = root / "uv.lock"
    lock_hash = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None
    return {
        "norm_version": NORM_VERSION,
        "git_commit": commit,
        "lockfile_sha256": lock_hash,
        "package_version": _package_version(),
    }


def _package_version() -> str:
    pyproject = settings.PROJECT_ROOT / "pyproject.toml"
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        if line.startswith("version"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "unknown"


def _noop(_: str) -> None:
    return None


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + f"{random.randint(0, 9999):04d}"


async def _warmup(name: str, lang: str, mode: str) -> None:
    from ..audio import to_pcm16
    import numpy as np
    from ..datasets.manifest import Dataset, Item

    silence = to_pcm16(np.zeros(16000, dtype=np.float32))
    tmp = settings.cache_dir() / "_warmup.wav"
    if not tmp.exists():
        from ..audio import write_wav
        write_wav(tmp, np.zeros(16000, dtype=np.float32))
    item = Item(
        id="warmup", audio=tmp.name, reference="warmup", lang=lang, duration_s=1.0,
        sha256="0", speech=[[0.1, 0.8]], turns=[[0.1, 0.8]],
    )
    ds = Dataset(name="_warmup", path=tmp.parent, meta={"lang": lang}, items=[item])
    try:
        p = load_provider(name)
        await run_item(p, ds, item, mode=mode, condition="wideband", timeout_s=3.0, silence_tail_s=0.3, frame_ms=20)
    except Exception:  # noqa: BLE001
        pass


async def run_benchmark(
    *,
    dataset_name: str,
    providers: list[str],
    mode: str = "forced",
    condition: str = "wideband",
    max_items: int | None = None,
    seed: int = 1234,
    concurrency: int = 2,
    warmup: bool = True,
    run_id: str | None = None,
    progress: Progress = _noop,
) -> dict:
    ds = load_dataset(dataset_name)
    bad = verify_dataset_audio(ds.path, ds.items)
    if bad:
        raise RuntimeError(f"{len(bad)} audio files failed SHA-256 verification (e.g. {bad[:3]})")
    items = list(ds.items)
    if max_items:
        items = items[: int(max_items)]
    cfg_snap = snapshot_configs(providers)
    defaults = cfg_snap.get("defaults") or {}
    timeout_s = float(defaults.get("forced_timeout_s" if mode == "forced" else "native_timeout_s", 5.0))
    silence_tail_s = float(defaults.get("silence_tail_s", 2.0))
    hosts = []
    for n in providers:
        hosts.extend(cfg_snap["providers"][n].get("hosts") or [])
    progress("measuring network baseline")
    net = baseline_for_hosts(hosts)

    run_id = run_id or new_run_id()
    run_dir = settings.runs_dir() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "lang": ds.lang,
        "manifest_sha256": ds.meta.get("manifest_sha256"),
        "n_items": len(items),
        "providers": providers,
        "mode": mode,
        "condition": condition,
        "timeout_s": timeout_s,
        "concurrency": concurrency,
        "seed": seed,
        "methodology": "MSB v1.0",
        "provenance": _provenance(),
        "network": net,
        "config": cfg_snap,
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2, default=str))

    if warmup:
        for n in providers:
            if n == "mock":
                continue
            progress(f"warmup {n}")
            await _warmup(n, ds.lang, mode)

    rng = random.Random(seed)
    order = list(providers)
    sem = asyncio.Semaphore(max(1, concurrency))
    n_done = 0
    lock = asyncio.Lock()
    out_path = run_dir / "results.jsonl"

    async def one(item, provider_name: str):
        nonlocal n_done
        async with sem:
            t0 = time.perf_counter()
            res: SessionResult | None = None
            attempts = 0
            for attempts in range(1, 3):
                try:
                    prov = load_provider(provider_name)
                    res = await run_item(
                        prov, ds, item, mode=mode, condition=condition,
                        timeout_s=timeout_s, silence_tail_s=silence_tail_s,
                        frame_ms=int(cfg_snap["providers"][provider_name].get("frame_ms") or defaults.get("frame_ms") or 20),
                    )
                except ProviderError as exc:
                    res = SessionResult(
                        item_id=item.id, provider=provider_name, lang=item.lang, mode=mode, condition=condition,
                        ok=False, hypothesis="", error=str(exc),
                    )
                if res.ok or attempts == 2 or not (res.error or "").startswith("connect:"):
                    break
            assert res is not None
            row = result_to_dict(res)
            row["attempts"] = attempts
            row["wall_s"] = round(time.perf_counter() - t0, 3)
            row["reference"] = item.reference
            row["duration_s"] = item.duration_s
            row["speaker"] = item.speaker
            row["tags"] = item.tags
            async with lock:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                n_done += 1
                if n_done % 5 == 0 or n_done == len(items) * len(providers):
                    progress(f"{run_id}: {n_done}/{len(items) * len(providers)} sessions")
            return row

    tasks = []
    for item in items:
        rng.shuffle(order)
        for name in order:
            tasks.append(asyncio.create_task(one(item, name)))
    await asyncio.gather(*tasks)
    meta["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2, default=str))
    upsert_run(meta, str(run_dir))
    progress(f"run {run_id} written to {run_dir}")
    return meta
