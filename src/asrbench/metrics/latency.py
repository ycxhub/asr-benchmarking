from __future__ import annotations

import numpy as np


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    return float(np.percentile(xs, p))


def summarize_latency(rows: list[dict]) -> dict:
    eos, ttft, partial = [], [], []
    n_to = n_pre = n_premature = n_turns = 0
    setups = []
    for row in rows:
        setups.append(row.get("setup_ms") or 0)
        for t in row.get("turns") or []:
            n_turns += 1
            if t.get("timeout"):
                n_to += 1
            if t.get("pre_endpoint_final"):
                n_pre += 1
            if t.get("premature_turn_end"):
                n_premature += 1
            if t.get("eos_to_final_ms") is not None and not t.get("timeout"):
                eos.append(float(t["eos_to_final_ms"]))
            if t.get("ttft_ms") is not None:
                ttft.append(float(t["ttft_ms"]))
            if t.get("eos_to_first_partial_ms") is not None:
                partial.append(float(t["eos_to_first_partial_ms"]))
    return {
        "n_turns": n_turns,
        "n_timeout": n_to,
        "timeout_rate": n_to / n_turns if n_turns else None,
        "n_pre_endpoint_final": n_pre,
        "n_premature_turn_end": n_premature,
        "eos_to_final_ms": {
            "n": len(eos),
            "p50": _pct(eos, 50),
            "p90": _pct(eos, 90),
            "p95": _pct(eos, 95),
            "mean": float(np.mean(eos)) if eos else None,
            "samples": eos,
        },
        "ttft_ms": {"n": len(ttft), "p50": _pct(ttft, 50), "p90": _pct(ttft, 90), "samples": ttft},
        "eos_to_first_partial_ms": {
            "n": len(partial),
            "p50": _pct(partial, 50) if partial else None,
            "p90": _pct(partial, 90) if partial else None,
            "note": "n/a for final-only providers" if not partial else None,
        },
        "setup_ms_p50": _pct(setups, 50),
        "empty_rate": sum(1 for r in rows if r.get("empty")) / len(rows) if rows else None,
        "error_rate": sum(1 for r in rows if r.get("error")) / len(rows) if rows else None,
    }
