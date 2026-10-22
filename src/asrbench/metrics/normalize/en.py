from __future__ import annotations

import re
import unicodedata

from whisper_normalizer.english import EnglishTextNormalizer

_BASE = EnglishTextNormalizer()

_APOS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
_DROPPED = {
    "dont": "don't", "doesnt": "doesn't", "didnt": "didn't", "wont": "won't",
    "cant": "can't", "isnt": "isn't", "arent": "aren't", "wasnt": "wasn't",
    "werent": "weren't", "havent": "haven't", "hasnt": "hasn't", "hadnt": "hadn't",
    "wouldnt": "wouldn't", "couldnt": "couldn't", "shouldnt": "shouldn't",
    "im": "i'm", "youre": "you're", "hes": "he's", "shes": "she's",
    "its": "it's", "were": "we're", "theyre": "they're", "ive": "i've",
    "youve": "you've", "weve": "we've", "theyve": "they've", "ill": "i'll",
    "youll": "you'll", "well": "we'll", "theyll": "they'll", "id": "i'd",
    "youd": "you'd", "wed": "we'd", "theyd": "they'd", "lets": "let's",
    "thats": "that's", "whats": "what's", "whos": "who's", "theres": "there's",
}
_INFORMAL = {
    "okay": "ok", "alright": "all right", "yep": "yeah", "yup": "yeah",
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
    "em": "them", "cause": "because", "cuz": "because",
}
_COMPOUNDS = {"e mail": "email", "web site": "website", "log in": "login", "sign up": "signup"}
_LAKH = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:lakh|lac)s?\b")
_CRORE = re.compile(r"\b(\d+(?:\.\d+)?)\b\s*crores?\b")
_WORD_LAKH = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|half|\d+(?:\.\d+)?)\s+(?:lakh|lac)s?\b")
_WORD_CRORE = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|half|\d+(?:\.\d+)?)\s+crores?\b")
_NUMWORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "half": 0.5}
_GROUP_COMMA = re.compile(r"(?<=\d),(?=\d)")
_SPACED_DIGITS = re.compile(r"(?<=\d)\s+(?=\d)")
_ACRONYM = re.compile(r"\b([a-z])(?:\s+([a-z])){1,8}\b")
_STRETCH = re.compile(r"\b([a-z])\1{2,}\b")
_LAUGH = re.compile(r"\b(?:ha)+h?\b")


def _numword(tok: str) -> float:
    return float(tok) if tok.replace(".", "", 1).isdigit() else float(_NUMWORD.get(tok, 0))


def normalize_en(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").translate(_APOS).lower()
    t = _GROUP_COMMA.sub("", t)
    t = re.sub(r"\b(" + "|".join(_DROPPED) + r")\b", lambda m: _DROPPED[m.group(1)], t)
    t = _WORD_LAKH.sub(lambda m: str(int(_numword(m.group(1)) * 100000)), t)
    t = _WORD_CRORE.sub(lambda m: str(int(_numword(m.group(1)) * 10_000_000)), t)
    t = _LAKH.sub(lambda m: str(int(float(m.group(1)) * 100000)), t)
    t = _CRORE.sub(lambda m: str(int(float(m.group(1)) * 10_000_000)), t)
    t = _BASE(t)
    t = _SPACED_DIGITS.sub("", t)
    t = _ACRONYM.sub(lambda m: m.group(0).replace(" ", ""), t)
    t = _STRETCH.sub(lambda m: m.group(1) * 2, t)
    t = _LAUGH.sub("", t)
    for a, b in _INFORMAL.items():
        t = re.sub(rf"\b{a}\b", b, t)
    for a, b in _COMPOUNDS.items():
        t = t.replace(a, b)
    return " ".join(t.split())
