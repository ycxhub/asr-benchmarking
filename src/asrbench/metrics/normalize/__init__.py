from __future__ import annotations

from .en import normalize_en
from .hi import normalize_hi, script_normalize_hi

NORM_VERSION = "en-1.0/hi-1.0"


def normalize(text: str, lang: str, *, script_normalize: bool = False) -> str:
    if lang == "hi":
        return normalize_hi(text, script_normalize=script_normalize)
    return normalize_en(text)
