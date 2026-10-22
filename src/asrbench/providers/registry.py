from __future__ import annotations

import importlib
from functools import lru_cache

import yaml

from .. import settings
from .base import StreamingProvider

ADAPTERS = {
    "mock": "asrbench.providers.mock:MockProvider",
    "murf": "asrbench.providers.murf:MurfProvider",
    "deepgram": "asrbench.providers.deepgram:DeepgramNova3Provider",
    "deepgram_flux": "asrbench.providers.deepgram:DeepgramFluxProvider",
    "elevenlabs": "asrbench.providers.elevenlabs:ElevenLabsProvider",
    "cartesia": "asrbench.providers.cartesia:CartesiaProvider",
    "cartesia_turns": "asrbench.providers.cartesia:CartesiaTurnsProvider",
    "sarvam": "asrbench.providers.sarvam:SarvamProvider",
}


def _adapter_class(adapter: str) -> type[StreamingProvider]:
    mod, cls = ADAPTERS[adapter].split(":")
    return getattr(importlib.import_module(mod), cls)


@lru_cache(maxsize=1)
def load_config() -> dict:
    return yaml.safe_load((settings.CONFIG_DIR / "providers.yaml").read_text())["providers"]


def provider_names(include_mock: bool = True) -> list[str]:
    return [n for n, c in load_config().items() if include_mock or c["adapter"] != "mock"]


def provider_config(name: str) -> dict:
    cfg = load_config().get(name)
    if cfg is None:
        raise KeyError(f"unknown provider '{name}'. Known: {', '.join(load_config())}")
    return cfg


def provider_class(name: str) -> type[StreamingProvider]:
    return _adapter_class(provider_config(name)["adapter"])


def make_provider(name: str) -> StreamingProvider:
    return provider_class(name)(name, provider_config(name))


def describe(name: str) -> dict:
    cfg = provider_config(name)
    cls = provider_class(name)
    key_env = cls.api_key_env
    return {
        "name": name,
        "label": cfg.get("label", name),
        "adapter": cfg["adapter"],
        "model": cfg.get("model", ""),
        "modes": [m for m, ok in (("forced", cls.supports_forced), ("native", cls.supports_native)) if ok],
        "languages": sorted((cfg.get("lang_map") or {}).keys()),
        "sample_rates": list(cls.sample_rates),
        "final_only": bool((cfg.get("params") or {}).get("final_only", cls.final_only)),
        "api_key_env": key_env,
        "key_present": (not key_env) or bool(settings.env(key_env)),
        "hosts": list(cls.hosts),
    }
