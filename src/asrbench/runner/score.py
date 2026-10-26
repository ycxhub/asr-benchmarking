from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .. import settings
from ..metrics.latency import summarize_latency
from ..metrics.stats import bootstrap_ci, bootstrap_percentile, paired_bootstrap
from ..metrics.wer import corpus_wer, score_pair
from ..suites import weights_by_dataset
from .report import render_report


def run_dir(run_id: str) -> Path:
    path = settings.runs_dir() / run_id
    if not (path / "run.json").exists():
        raise FileNotFoundError(f"run '{run_id}' not found in {settings.runs_dir()}")
    return path


def load_run(run_id: str) -> tuple[dict, list[dict]]:
    path = run_dir(run_id)
    meta = json.loads((path / "run.json").read_text())
    rows = []
    results = path / "results.jsonl"
    if results.exists():
        for line in results.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return meta, rows


def _failed(row: dict) -> bool:
    if row.get("error"):
        return True
    if not (row.get("hypothesis") or "").strip():
        return True
    return row.get("ok") is False


def _duration_bucket(seconds: float) -> str:
    if seconds < 5:
        return "0-5s"
    if seconds < 15:
        return "5-15s"
    if seconds < 30:
        return "15-30s"
    return "30s+"


def _cmi_bucket(value) -> str | None:
    if value is None:
        return None
    try:
        cmi = float(value)
    except (TypeError, ValueError):
        return None
    if cmi <= 0:
        return "0"
    if cmi <= 0.1:
        return "(0, 0.1]"
    if cmi <= 0.25:
        return "(0.1, 0.25]"
    if cmi <= 0.5:
        return "(0.25, 0.5]"
    return ">0.5"


def score_rows(meta: dict, rows: list[dict]) -> tuple[dict, list[dict]]:
    lang = meta.get("lang") or "en-US"
    utterances: list[dict] = []
    for row in rows:
        failed = _failed(row)
        pair = score_pair(row.get("reference") or "", row.get("hypothesis") or "", row.get("lang") or lang, failed=failed)
        turns = row.get("turns") or []
        eos = [t.get("eos_to_final_ms") for t in turns if t.get("eos_to_final_ms") is not None and not t.get("timeout")]
        tags = row.get("tags") or {}
        utterances.append({
            "item_id": row.get("item_id"),
            "provider": row.get("provider"),
            "reference": row.get("reference") or "",
            "hypothesis": row.get("hypothesis") or "",
            "ref_norm": pair.ref_norm,
            "hyp_norm": pair.hyp_norm,
            "wer": pair.wer,
            "cer": pair.cer,
            "sn_wer": pair.sn_wer,
            "substitutions": pair.substitutions,
            "deletions": pair.deletions,
            "insertions": pair.insertions,
            "ref_words": pair.ref_words,
            "failed": pair.failed,
            "error": row.get("error"),
            "duration_s": row.get("duration_s") or 0,
            "speaker": row.get("speaker"),
            "tags": tags,
            "duration_bucket": _duration_bucket(float(row.get("duration_s") or 0)),
            "cmi_bucket": _cmi_bucket(tags.get("cmi")),
            "eos_to_final_ms": None if not eos else sum(eos) / len(eos),
        })

    by_provider: dict[str, list[dict]] = defaultdict(list)
    raw_by_provider: dict[str, list[dict]] = defaultdict(list)
    for utt, row in zip(utterances, rows):
        by_provider[utt["provider"]].append(utt)
        raw_by_provider[utt["provider"]].append(row)

    providers = []
    for name in meta.get("providers") or sorted(by_provider):
        group = by_provider.get(name) or []
        from ..metrics.wer import PairScore

        pairs = [
            PairScore(
                u["substitutions"], u["deletions"], u["insertions"], 0, u["ref_words"],
                len((u["hyp_norm"] or "").split()), u["wer"], u["cer"], u["sn_wer"],
                u["ref_norm"], u["hyp_norm"], u["failed"],
            )
            for u in group
        ]
        wer = corpus_wer(pairs)
        weighted = corpus_wer(pairs, duration_weighted=True, durations=[float(u["duration_s"] or 0) for u in group])
        lat = summarize_latency(raw_by_provider.get(name) or [])
        eos_samples = lat["eos_to_final_ms"].pop("samples", [])
        lat["ttft_ms"].pop("samples", [])
        lat["eos_to_final_ms"]["ci95"] = bootstrap_percentile(eos_samples, 50)
        providers.append({
            "provider": name,
            "wer": wer,
            "wer_duration_weighted": weighted.get("wer"),
            "wer_ci95": bootstrap_ci(group),
            "latency": lat,
        })

    murf = next((p for p in providers if p["provider"] == "murf" or str(p["provider"]).startswith("murf")), None)
    paired = []
    if murf:
        base = by_provider[murf["provider"]]
        for other in providers:
            if other["provider"] == murf["provider"]:
                continue
            paired.append({
                "baseline": murf["provider"],
                "other": other["provider"],
                **paired_bootstrap(base, by_provider[other["provider"]]),
            })

    weights = weights_by_dataset()
    dataset = meta.get("dataset")
    suite_weight = weights.get(dataset)

    def _slice(key: str) -> list[dict]:
        out = []
        buckets: dict[tuple, list] = defaultdict(list)
        for u in utterances:
            label = u.get(key)
            if not label:
                continue
            buckets[(u["provider"], label)].append(u)
        for (provider, label), group in sorted(buckets.items()):
            out.append({
                "provider": provider,
                "bucket": label,
                "n": len(group),
                "wer": pooled_from(group),
            })
        return out

    summary = {
        "run_id": meta.get("run_id"),
        "dataset": dataset,
        "lang": lang,
        "mode": meta.get("mode"),
        "condition": meta.get("condition"),
        "n_rows": len(rows),
        "suite_lang": None if not suite_weight else suite_weight[0],
        "suite_weight": None if not suite_weight else suite_weight[1],
        "providers": providers,
        "paired_vs_murf": paired,
        "slices": {"duration": _slice("duration_bucket"), "cmi": _slice("cmi_bucket")},
        "worst": sorted(utterances, key=lambda u: u["wer"], reverse=True)[:50],
    }
    return summary, utterances


def pooled_from(group: list[dict]) -> float | None:
    from ..metrics.stats import pooled_wer

    return pooled_wer(group)


def score_run(run_id: str) -> dict:
    meta, rows = load_run(run_id)
    summary, utterances = score_rows(meta, rows)
    path = run_dir(run_id)
    (path / "scores.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(path / "utterances.jsonl", "w", encoding="utf-8") as f:
        for utt in utterances:
            f.write(json.dumps(utt, ensure_ascii=False) + "\n")
    (path / "report.html").write_text(render_report(summary, utterances), encoding="utf-8")
    return summary
