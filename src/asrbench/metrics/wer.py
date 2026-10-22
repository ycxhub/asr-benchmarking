from __future__ import annotations

from dataclasses import dataclass

from jiwer import process_words

from .normalize import normalize


@dataclass
class PairScore:
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_words: int
    hyp_words: int
    wer: float
    cer: float
    sn_wer: float | None
    ref_norm: str
    hyp_norm: str
    failed: bool = False


def _align(ref: str, hyp: str) -> tuple[int, int, int, int, int]:
    if not ref.strip():
        n = 0
        ins = len(hyp.split())
        return 0, 0, ins, 0, n
    out = process_words(ref, hyp)
    s, d, i, h = out.substitutions, out.deletions, out.insertions, out.hits
    n = s + d + h
    return s, d, i, h, n


def _cer(ref: str, hyp: str) -> float:
    a, b = ref.replace(" ", ""), hyp.replace(" ", "")
    if not a:
        return 0.0 if not b else 1.0
    # simple Levenshtein on characters
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(a)


def score_pair(reference: str, hypothesis: str, lang: str, *, failed: bool = False) -> PairScore:
    ref_n = normalize(reference, lang, script_normalize=False)
    if failed or hypothesis is None:
        s, d, i, h, n = _align(ref_n, "")
        if n == 0:
            n = max(1, len(ref_n.split()))
            d = n
        return PairScore(s, d, i, h, n, 0, 1.0 if n else 0.0, 1.0, 1.0 if lang == "hi" else None, ref_n, "", True)
    hyp_n = normalize(hypothesis, lang, script_normalize=False)
    s, d, i, h, n = _align(ref_n, hyp_n)
    wer = (s + d + i) / n if n else (0.0 if not hyp_n else 1.0)
    cer = _cer(ref_n, hyp_n)
    sn = None
    if lang == "hi":
        rs = normalize(reference, lang, script_normalize=True)
        hs = normalize(hypothesis, lang, script_normalize=True)
        ss, dd, ii, hh, nn = _align(rs, hs)
        sn = (ss + dd + ii) / nn if nn else (0.0 if not hs else 1.0)
    return PairScore(s, d, i, h, n, len(hyp_n.split()), wer, cer, sn, ref_n, hyp_n, False)


def corpus_wer(rows: list[PairScore], *, duration_weighted: bool = False, durations: list[float] | None = None) -> dict:
    if duration_weighted and durations and len(durations) == len(rows):
        num = den = 0.0
        for r, dur in zip(rows, durations):
            w = (r.substitutions + r.deletions + r.insertions) / r.ref_words if r.ref_words else r.wer
            num += w * dur
            den += dur
        return {"wer": num / den if den else None, "kind": "duration-weighted"}
    s = sum(r.substitutions for r in rows)
    d = sum(r.deletions for r in rows)
    i = sum(r.insertions for r in rows)
    n = sum(r.ref_words for r in rows)
    sn_rows = [r.sn_wer for r in rows if r.sn_wer is not None]
    # pooled SN-WER approximated from per-clip if we didn't store SN counts; recompute from sn_wer * n
    sn = None
    if any(r.sn_wer is not None for r in rows):
        sn_err = sum((r.sn_wer or 0) * r.ref_words for r in rows)
        sn = sn_err / n if n else None
    return {
        "wer": (s + d + i) / n if n else None,
        "sn_wer": sn,
        "cer": sum(r.cer * r.ref_words for r in rows) / n if n else None,
        "substitutions": s,
        "deletions": d,
        "insertions": i,
        "ref_words": n,
        "n_clips": len(rows),
        "n_failed": sum(1 for r in rows if r.failed),
        "kind": "pooled",
    }
