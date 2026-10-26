from __future__ import annotations

import asyncio
import re
from collections import Counter

import typer

from . import settings
from .datasets.importers import import_folder, import_from_spec
from .datasets.manifest import list_datasets
from .metrics.normalize.hi import lexicon
from .providers import all_provider_cfgs
from .runner.orchestrator import run_benchmark
from .runner.score import load_run, score_run
from .runner.validate import validate_sync
from .suites import list_suites, load_suite

app = typer.Typer(add_completion=False, help="Streaming speech-to-text benchmark.")


def _echo_meta(meta: dict) -> None:
    typer.echo(
        f"{meta.get('name')}: {meta.get('n_items')} clips, {meta.get('n_rejected')} rejected, "
        f"{meta.get('total_audio_s')}s audio, vad={meta.get('vad_method')}"
    )


def _missing_keys(names: list[str]) -> list[str]:
    missing = []
    cfgs = all_provider_cfgs()
    for name in names:
        env_name = (cfgs.get(name) or {}).get("api_key_env")
        if env_name and not settings.env(env_name):
            missing.append(f"{name} needs {env_name}")
    return missing


@app.command()
def validate(full: bool = typer.Option(False, help="Also check 100 ms and 1000 ms injected delays.")) -> None:
    """Check that the mock provider's injected delay is recovered within 5 ms."""
    delays = (0, 100, 300, 1000) if full else (0, 300)
    report = validate_sync(delays_ms=delays)
    for row in report["rows"]:
        mark = "ok" if row["pass"] else "FAIL"
        typer.echo(
            f"{mark} {row['mode']} injected={row['injected_ms']}ms "
            f"measured_p50={row['measured_p50_ms']} expected={row['expected_ms']} error={row['error_ms']}"
        )
    if not report["pass"]:
        raise typer.Exit(code=1)


@app.command("list-suites")
def list_suites_cmd() -> None:
    """Show configured evaluation suites."""
    for suite in list_suites():
        names = ", ".join(ds["name"] for ds in suite["datasets"])
        typer.echo(f"{suite['suite']} ({suite['lang']}): {names}")


@app.command("list-datasets")
def list_datasets_cmd() -> None:
    """Show datasets already ingested on this machine."""
    rows = list_datasets()
    if not rows:
        typer.echo(f"No datasets in {settings.datasets_dir()}")
        return
    for row in rows:
        typer.echo(f"{row.get('name')} lang={row.get('lang')} n={row.get('n_items')} sha={str(row.get('manifest_sha256'))[:12]}")


@app.command("list-providers")
def list_providers_cmd() -> None:
    """Show pinned providers and whether their API key is set."""
    for name, cfg in all_provider_cfgs().items():
        env_name = cfg.get("api_key_env")
        if not env_name:
            state = "no key"
        elif settings.env(env_name):
            state = f"{env_name} set"
        else:
            state = f"{env_name} missing"
        flag = "enabled" if cfg.get("enabled", True) else "disabled"
        typer.echo(f"{name}: {flag}, {state}")


@app.command()
def ingest(
    suite: str = typer.Argument(...),
    dataset: str | None = typer.Option(None, help="Ingest one dataset from the suite."),
) -> None:
    """Ingest a suite. Gated Hugging Face datasets stop until HF_TOKEN is set."""
    spec = load_suite(suite)
    chosen = [ds for ds in spec["datasets"] if dataset is None or ds["name"] == dataset]
    if not chosen:
        raise typer.BadParameter(f"{dataset} is not in suite {suite}")
    for ds in chosen:
        meta = import_from_spec(ds, progress=typer.echo)
        _echo_meta(meta)


@app.command("ingest-folder")
def ingest_folder_cmd(
    path: str = typer.Argument(..., help="Folder or zip with a manifest and audio files."),
    name: str = typer.Option(...),
    lang: str = typer.Option(...),
) -> None:
    """Ingest a local folder or zip."""
    meta = import_folder(path, name, lang, progress=typer.echo)
    _echo_meta(meta)


@app.command()
def run(
    dataset: str = typer.Option(...),
    providers: str = typer.Option("mock", help="Comma-separated provider names."),
    mode: str = typer.Option("forced"),
    condition: str = typer.Option("wideband"),
    max_items: int | None = typer.Option(None),
    concurrency: int = typer.Option(2),
) -> None:
    """Stream a dataset through providers, then score the run."""
    names = [part.strip() for part in providers.split(",") if part.strip()]
    missing = _missing_keys(names)
    if missing:
        typer.echo("Refusing to start. " + "; ".join(missing) + ". Add the key to .env.")
        raise typer.Exit(code=2)
    meta = asyncio.run(
        run_benchmark(
            dataset_name=dataset,
            providers=names,
            mode=mode,
            condition=condition,
            max_items=max_items,
            concurrency=concurrency,
            progress=typer.echo,
        )
    )
    summary = score_run(meta["run_id"])
    for row in summary["providers"]:
        wer = (row.get("wer") or {}).get("wer")
        p50 = ((row.get("latency") or {}).get("eos_to_final_ms") or {}).get("p50")
        typer.echo(f"{row['provider']}: WER {None if wer is None else round(100 * wer, 2)}%  eos_p50 {None if p50 is None else round(p50)} ms")
    typer.echo(f"report: {settings.runs_dir() / meta['run_id'] / 'report.html'}")


@app.command()
def score(run_id: str = typer.Argument(...)) -> None:
    """Recompute WER, latency, confidence intervals, and the HTML report."""
    summary = score_run(run_id)
    typer.echo(f"scored {summary['n_rows']} sessions -> {settings.runs_dir() / run_id / 'report.html'}")


@app.command("translit-mine")
def translit_mine(run_id: str = typer.Argument(...)) -> None:
    """List Latin tokens in Hindi hypotheses that are not in the lexicon yet."""
    _meta, rows = load_run(run_id)
    known = lexicon()
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("lang") not in (None, "hi"):
            continue
        for token in re.findall(r"[A-Za-z][A-Za-z']+", row.get("hypothesis") or ""):
            key = token.casefold()
            if key not in known and token not in known:
                counts[key] += 1
    if not counts:
        typer.echo("No unmapped Latin tokens.")
        return
    for token, n in counts.most_common(100):
        typer.echo(f"{n}\t{token}")


@app.command()
def serve(host: str = typer.Option("127.0.0.1"), port: int = typer.Option(8000)) -> None:
    """Open the local review UI."""
    import uvicorn

    uvicorn.run("asrbench.web.app:app", host=host, port=port)


if __name__ == "__main__":
    app()
