from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from .. import settings
from .base import Event, ProviderError, SessionContext, StreamingProvider, WebSocketProvider

_CACHE: dict | None = None


def providers_yaml() -> dict:
    global _CACHE
    if _CACHE is None:
        _CACHE = yaml.safe_load((settings.CONFIG_DIR / "providers.yaml").read_text())
    return _CACHE


def all_provider_cfgs() -> dict[str, dict]:
    return dict(providers_yaml()["providers"])


def enabled_providers() -> list[str]:
    return [n for n, c in all_provider_cfgs().items() if c.get("enabled", True)]


def load_provider(name: str) -> StreamingProvider:
    cfgs = all_provider_cfgs()
    if name not in cfgs:
        raise KeyError(f"unknown provider {name!r}. known: {sorted(cfgs)}")
    cfg = cfgs[name]
    mod_name, cls_name = cfg["class"].split(".")
    mod = importlib.import_module(f"asrbench.providers.{mod_name}")
    cls = getattr(mod, cls_name)
    return cls(name, cfg)


def snapshot_configs(names: list[str]) -> dict:
    y = providers_yaml()
    return {
        "defaults": y.get("defaults") or {},
        "providers": {n: all_provider_cfgs()[n] for n in names},
    }


__all__ = [
    "Event",
    "ProviderError",
    "SessionContext",
    "StreamingProvider",
    "WebSocketProvider",
    "all_provider_cfgs",
    "enabled_providers",
    "load_provider",
    "snapshot_configs",
    "providers_yaml",
]
