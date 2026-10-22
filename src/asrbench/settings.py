from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

load_dotenv(PROJECT_ROOT / ".env", override=False)

LANGUAGES = ("hi", "en-IN", "en-US")


@lru_cache(maxsize=1)
def home() -> Path:
    raw = os.environ.get("ASRBENCH_HOME") or "~/asrbench-data"
    path = Path(os.path.expandvars(raw)).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def datasets_dir() -> Path:
    p = home() / "datasets"
    p.mkdir(parents=True, exist_ok=True)
    return p


def runs_dir() -> Path:
    p = home() / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = home() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    p = home() / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None
