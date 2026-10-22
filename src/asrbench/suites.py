from __future__ import annotations

from pathlib import Path

import yaml

from . import settings


def suites_dir() -> Path:
    return settings.CONFIG_DIR / "suites"


def list_suites() -> list[dict]:
    return [load_suite(p.stem) for p in sorted(suites_dir().glob("*.yaml"))]


def load_suite(name: str) -> dict:
    path = suites_dir() / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"suite '{name}' not found in {suites_dir()}")
    suite = yaml.safe_load(path.read_text())
    for ds in suite["datasets"]:
        ds.setdefault("lang", suite["lang"])
    return suite


def dataset_spec(suite_name: str, dataset_name: str) -> dict:
    for ds in load_suite(suite_name)["datasets"]:
        if ds["name"] == dataset_name:
            return ds
    raise KeyError(f"dataset {dataset_name} not in suite {suite_name}")


def weights_by_dataset() -> dict[str, tuple[str, float]]:
    """dataset name -> (suite lang, weight) across all suites."""
    out = {}
    for suite in list_suites():
        for ds in suite["datasets"]:
            out[ds["name"]] = (suite["lang"], float(ds.get("weight", 1.0)))
    return out
