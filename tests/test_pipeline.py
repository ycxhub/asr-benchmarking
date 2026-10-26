import asyncio

import pytest

from asrbench import settings
from asrbench.datasets.importers import import_from_spec
from asrbench.runner.orchestrator import run_benchmark
from asrbench.runner.score import score_run
from asrbench.suites import dataset_spec


def test_gated_dataset_stops_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    spec = dataset_spec("hi", "indicvoices-hi")
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        import_from_spec(spec)


def test_smoke_mock_run_scores_zero_wer(tmp_path, monkeypatch):
    monkeypatch.setenv("ASRBENCH_HOME", str(tmp_path))
    settings.home.cache_clear()
    spec = dict(dataset_spec("smoke", "smoke-en"))
    spec["sample"] = 1
    meta = import_from_spec(spec)
    assert meta["n_items"] == 1
    run = asyncio.run(
        run_benchmark(
            dataset_name="smoke-en",
            providers=["mock"],
            max_items=1,
            warmup=False,
            concurrency=1,
        )
    )
    summary = score_run(run["run_id"])
    scored = summary["providers"][0]
    assert scored["wer"]["wer"] == 0
    assert scored["latency"]["eos_to_final_ms"]["p50"] == pytest.approx(300, abs=40)
    report = settings.runs_dir() / run["run_id"] / "report.html"
    assert "Leaderboard" in report.read_text(encoding="utf-8")
    assert (settings.home() / "index.sqlite").exists()
