from __future__ import annotations

import numpy as np


def pooled_wer(rows: list[dict]) -> float | None:
    n = sum(int(r.get("ref_words") or 0) for r in rows)
    if not n:
        return None
    err = sum(int(r.get("substitutions") or 0) + int(r.get("deletions") or 0) + int(r.get("insertions") or 0) for r in rows)
    return err / n


def _percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    return float(np.percentile(np.asarray(xs, dtype=float), p))


def bootstrap_ci(
    rows: list[dict],
    *,
    cluster_key: str = "speaker",
    n_boot: int = 1000,
    seed: int = 1234,
    alpha: float = 0.05,
) -> dict:
    """95% CI for pooled WER. Resamples speakers when more than one speaker is present."""
    point = pooled_wer(rows)
    if not rows or point is None:
        return {"estimate": point, "low": None, "high": None, "n": len(rows), "kind": "empty", "n_boot": n_boot}
    groups: dict[str, list[dict]] = {}
    for i, row in enumerate(rows):
        key = row.get(cluster_key) or f"row-{i}"
        groups.setdefault(str(key), []).append(row)
    keys = list(groups)
    clustered = len(keys) >= 2 and len(keys) < len(rows)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        if clustered:
            draw = rng.integers(0, len(keys), size=len(keys))
            picked = [row for j in draw for row in groups[keys[int(j)]]]
        else:
            draw = rng.integers(0, len(rows), size=len(rows))
            picked = [rows[int(j)] for j in draw]
        wer = pooled_wer(picked)
        if wer is not None:
            samples.append(wer)
    lo = _percentile(samples, 100 * alpha / 2)
    hi = _percentile(samples, 100 * (1 - alpha / 2))
    return {
        "estimate": point,
        "low": lo,
        "high": hi,
        "n": len(rows),
        "n_clusters": len(keys) if clustered else len(rows),
        "kind": "cluster" if clustered else "clip",
        "n_boot": n_boot,
    }


def paired_bootstrap(
    baseline: list[dict],
    other: list[dict],
    *,
    n_boot: int = 1000,
    seed: int = 1234,
    alpha: float = 0.05,
) -> dict:
    """Paired clip bootstrap of (other WER - baseline WER). Positive means the other model is worse."""
    base = {r["item_id"]: r for r in baseline}
    pairs = [(base[r["item_id"]], r) for r in other if r.get("item_id") in base]
    if not pairs:
        return {"n": 0, "diff": None, "low": None, "high": None, "p_two_sided": None}
    point_b = pooled_wer([a for a, _ in pairs])
    point_o = pooled_wer([b for _, b in pairs])
    point = None if point_b is None or point_o is None else point_o - point_b
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    n = len(pairs)
    for _ in range(n_boot):
        draw = rng.integers(0, n, size=n)
        wb = pooled_wer([pairs[int(j)][0] for j in draw])
        wo = pooled_wer([pairs[int(j)][1] for j in draw])
        if wb is None or wo is None:
            continue
        diffs.append(wo - wb)
    if not diffs or point is None:
        return {"n": n, "diff": point, "low": None, "high": None, "p_two_sided": None}
    # Two-sided p: share of resamples on the other side of 0 from the point estimate, doubled.
    if point >= 0:
        p = 2 * sum(d <= 0 for d in diffs) / len(diffs)
    else:
        p = 2 * sum(d >= 0 for d in diffs) / len(diffs)
    return {
        "n": n,
        "diff": point,
        "low": _percentile(diffs, 100 * alpha / 2),
        "high": _percentile(diffs, 100 * (1 - alpha / 2)),
        "p_two_sided": min(1.0, p),
        "n_boot": n_boot,
    }


def bootstrap_percentile(values: list[float], percentile: float, *, n_boot: int = 1000, seed: int = 1234, alpha: float = 0.05) -> dict:
    if not values:
        return {"estimate": None, "low": None, "high": None, "n": 0}
    arr = np.asarray(values, dtype=float)
    point = float(np.percentile(arr, percentile))
    rng = np.random.default_rng(seed)
    draws = [float(np.percentile(arr[rng.integers(0, len(arr), size=len(arr))], percentile)) for _ in range(n_boot)]
    return {
        "estimate": point,
        "low": _percentile(draws, 100 * alpha / 2),
        "high": _percentile(draws, 100 * (1 - alpha / 2)),
        "n": len(values),
        "n_boot": n_boot,
    }
