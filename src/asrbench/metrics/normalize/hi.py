from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from .en import normalize_en

NORM_DIR = Path(__file__).resolve().parents[4] / "configs" / "normalization"

# Combining nukta. NFC decomposes the precomposed nukta letters (they are composition
# exclusions), so dropping U+093C folds क़/क़ to the base letter.
_NUKTA = "\u093c"
_CHANDRA = str.maketrans({"ॉ": "ो", "ॅ": "े", "ऑ": "ओ", "ऍ": "ए", "ँ": "ं"})
_DEV_DIGIT = str.maketrans("०१२३४५६७८९", "0123456789")
_PUNCT = re.compile(r"[।॥॰\-–—.,!?;:\"'“”‘’()\[\]{}<>/\\|*~`@#^&+=_]+")
_ZW = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_BRACKET = re.compile(r"\[.*?\]|<.*?>|\(noise\)|\(laugh\)|\(music\)", re.I)
_FILLERS = {"हम्म", "हम्मम", "उम्म", "आह", "अह", "hmm", "hm", "um", "uh", "uhh", "aah", "huh"}
_HALF_NASAL = re.compile(r"([ङञणनम])्(?=[\u0915-\u0939])")

_ONES = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, "छह": 6, "छे": 6,
    "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14,
    "पंद्रह": 15, "पन्द्रह": 15, "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20,
    "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24, "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27,
    "अट्ठाईस": 28, "उनतीस": 29, "तीस": 30, "चालीस": 40, "पचास": 50, "साठ": 60, "सत्तर": 70,
    "अस्सी": 80, "नब्बे": 90, "सौ": 100,
}
_SCALE = {"सौ": 100, "हजार": 1000, "हज़ार": 1000, "लाख": 100000, "करोड": 10_000_000, "करोड़": 10_000_000, "अरब": 10**9}
_FRAC = {"डेढ़": 1.5, "देढ़": 1.5, "ढाई": 2.5, "साढ़े": 0.5, "सवा": 0.25, "पौने": -0.25}

_LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z']*")
_DEV_TOKEN = re.compile(r"[\u0900-\u097F]+")

# Very small ITRANS-ish fallback for leftover romanized Hindi (not a neural transliterator).
_ITRANS = [
    ("ksh", "क्ष"), ("gy", "ज्ञ"), ("sh", "श"), ("chh", "छ"), ("ch", "च"), ("th", "थ"),
    ("dh", "ध"), ("kh", "ख"), ("gh", "घ"), ("ph", "फ"), ("bh", "भ"), ("ng", "ङ"),
    ("aa", "ा"), ("ee", "ी"), ("ii", "ी"), ("oo", "ू"), ("uu", "ू"), ("ai", "ै"), ("au", "ौ"),
    ("ka", "क"), ("ki", "कि"), ("ku", "कु"), ("ke", "के"), ("ko", "को"),
    ("ga", "ग"), ("gi", "गि"), ("gu", "गु"), ("ge", "गे"), ("go", "गो"),
    ("ha", "ह"), ("hi", "हि"), ("hu", "हु"), ("he", "हे"), ("ho", "हो"),
    ("na", "न"), ("ni", "नि"), ("nu", "नु"), ("ne", "ने"), ("no", "नो"),
    ("ma", "म"), ("mi", "मि"), ("mu", "मु"), ("me", "मे"), ("mo", "मो"),
    ("ra", "र"), ("ri", "रि"), ("ru", "रु"), ("re", "रे"), ("ro", "रो"),
    ("la", "ल"), ("li", "लि"), ("lu", "लु"), ("le", "ले"), ("lo", "लो"),
    ("ta", "त"), ("ti", "ति"), ("tu", "तु"), ("te", "ते"), ("to", "तो"),
    ("da", "द"), ("di", "दि"), ("du", "दु"), ("de", "दे"), ("do", "दो"),
    ("pa", "प"), ("pi", "पि"), ("pu", "पु"), ("pe", "पे"), ("po", "पो"),
    ("ba", "ब"), ("bi", "बि"), ("bu", "बु"), ("be", "बे"), ("bo", "बो"),
    ("sa", "स"), ("si", "सि"), ("su", "सु"), ("se", "से"), ("so", "सो"),
    ("ya", "य"), ("ye", "ये"), ("yo", "यो"),
    ("wa", "व"), ("va", "व"),
    ("a", "अ"), ("i", "इ"), ("u", "उ"), ("e", "ए"), ("o", "ओ"),
    ("k", "क"), ("g", "ग"), ("h", "ह"), ("n", "न"), ("m", "म"), ("r", "र"),
    ("l", "ल"), ("t", "त"), ("d", "द"), ("p", "प"), ("b", "ब"), ("s", "स"),
    ("y", "य"), ("v", "व"), ("w", "व"), ("j", "ज"), ("c", "च"),
]


@lru_cache(maxsize=1)
def lexicon() -> dict[str, str]:
    path = NORM_DIR / "hi_lexicon.tsv"
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if len(parts) < 2:
            continue
        canon = parts[0]
        mapping[canon] = canon
        for alt in parts[1:]:
            mapping[alt.casefold()] = canon
            mapping[alt] = canon
    return mapping


def _translit_fallback(token: str) -> str:
    t = token.casefold()
    out = []
    i = 0
    while i < len(t):
        hit = None
        for src, dst in _ITRANS:
            if t.startswith(src, i):
                hit = (len(src), dst)
                break
        if hit:
            out.append(hit[1])
            i += hit[0]
        else:
            i += 1
    return "".join(out) or token


def script_normalize_token(tok: str) -> str:
    lex = lexicon()
    if tok in lex:
        return lex[tok]
    low = tok.casefold()
    if low in lex:
        return lex[low]
    if _LATIN_TOKEN.fullmatch(tok):
        # Keep unambiguous English words as lowercase latin; transliterate the rest.
        if low in lex:
            return lex[low]
        if len(tok) <= 2:
            return low
        return _translit_fallback(tok) or low
    return tok


def script_normalize_hi(text: str) -> str:
    return " ".join(script_normalize_token(t) for t in text.split() if t)


def _hi_numbers(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in _FRAC and i + 1 < len(tokens) and tokens[i + 1] in _SCALE:
            scale = _SCALE[tokens[i + 1]]
            factor = _FRAC[t]
            if factor in (0.5, 0.25, -0.25) and i + 2 < len(tokens) and tokens[i + 1] not in ("लाख", "करोड", "करोड़"):
                # साढ़े तीन सौ
                pass
            if factor in (1.5, 2.5):
                out.append(str(int(factor * scale)))
                i += 2
                continue
        if t in _ONES:
            val = _ONES[t]
            j = i + 1
            while j < len(tokens) and tokens[j] in _SCALE:
                val *= _SCALE[tokens[j]]
                j += 1
            out.append(str(int(val)))
            i = j
            continue
        out.append(t)
        i += 1
    return out


def normalize_hi(text: str, *, script_normalize: bool = False) -> str:
    t = unicodedata.normalize("NFC", text or "")
    t = t.replace(_NUKTA, "")
    t = _BRACKET.sub(" ", t)
    t = _ZW.sub("", t)
    t = t.translate(_CHANDRA).translate(_DEV_DIGIT)
    t = _HALF_NASAL.sub("ं", t)
    t = t.replace("₹", " रुपये ").replace("%", " प्रतिशत ")
    t = re.sub(r"\b(?:rs|inr)\b", " रुपये ", t, flags=re.I)
    t = _PUNCT.sub(" ", t)
    # Latin runs: apply English number folding
    def _lat(m: re.Match) -> str:
        return normalize_en(m.group(0))

    t = re.sub(r"[A-Za-z0-9][A-Za-z0-9'.,]*", _lat, t)
    tokens = [tok for tok in t.split() if tok and tok.casefold() not in _FILLERS and tok not in _FILLERS]
    tokens = _hi_numbers(tokens)
    t = " ".join(tokens)
    if script_normalize:
        t = script_normalize_hi(t)
    return " ".join(t.split())
