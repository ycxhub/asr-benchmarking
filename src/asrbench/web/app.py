from __future__ import annotations

import asyncio
import json
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .. import settings
from ..datasets.importers import import_folder, import_from_spec
from ..datasets.manifest import list_datasets
from ..providers import all_provider_cfgs
from ..runner.orchestrator import new_run_id, run_benchmark
from ..runner.score import run_dir, score_run
from ..suites import list_suites

app = FastAPI(title="asrbench")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
_jobs: dict[str, dict] = {}


def _providers() -> list[dict]:
    rows = []
    for name, cfg in all_provider_cfgs().items():
        env_name = cfg.get("api_key_env")
        rows.append({
            "name": name,
            "enabled": cfg.get("enabled", True),
            "description": cfg.get("description", ""),
            "api_key_env": env_name,
            "key_present": (not env_name) or bool(settings.env(env_name)),
        })
    return rows


def _runs() -> list[dict]:
    rows = []
    for path in sorted(settings.runs_dir().glob("*/run.json"), reverse=True):
        try:
            meta = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        run_id = meta.get("run_id") or path.parent.name
        job = _jobs.get(run_id) or {}
        if (path.parent / "scores.json").exists():
            status = "scored"
        elif job.get("status"):
            status = job["status"]
        elif meta.get("finished_at"):
            status = "finished"
        else:
            status = "incomplete"
        rows.append({**meta, "status": status, "log": job.get("log") or []})
    return rows


def _missing(names: list[str]) -> list[str]:
    cfgs = all_provider_cfgs()
    missing = []
    for name in names:
        env_name = (cfgs.get(name) or {}).get("api_key_env")
        if env_name and not settings.env(env_name):
            missing.append(f"{name} needs {env_name}")
    return missing


@app.get("/", response_class=HTMLResponse)
def index(request: Request, error: str = ""):
    return templates.TemplateResponse(request, "index.html", {
        "providers": _providers(),
        "suites": list_suites(),
        "datasets": list_datasets(),
        "runs": _runs(),
        "error": error,
    })


@app.post("/ingest")
def ingest(suite: str = Form(...), dataset: str = Form("")):
    try:
        spec = next(s for s in list_suites() if s["suite"] == suite)
        chosen = [ds for ds in spec["datasets"] if not dataset or ds["name"] == dataset]
        for ds in chosen:
            import_from_spec(ds)
    except Exception as exc:  # noqa: BLE001
        return RedirectResponse("/?error=" + quote(str(exc)), status_code=303)
    return RedirectResponse("/", status_code=303)


@app.post("/upload")
async def upload(name: str = Form(...), lang: str = Form(...), file: UploadFile = File(...)):
    dest = settings.cache_dir() / (file.filename or "upload.zip")
    dest.write_bytes(await file.read())
    try:
        import_folder(dest, name, lang)
    except Exception as exc:  # noqa: BLE001
        return RedirectResponse("/?error=" + quote(str(exc)), status_code=303)
    return RedirectResponse("/", status_code=303)


@app.post("/runs")
async def start_run(
    dataset: str = Form(...),
    mode: str = Form("forced"),
    condition: str = Form("wideband"),
    max_items: str = Form(""),
    providers: list[str] = Form(...),
):
    missing = _missing(providers)
    if missing:
        return RedirectResponse("/?error=" + quote("; ".join(missing)), status_code=303)
    run_id = new_run_id()
    _jobs[run_id] = {"status": "running", "log": ["starting"]}

    def progress(message: str) -> None:
        _jobs[run_id]["log"].append(message)

    async def job() -> None:
        try:
            await run_benchmark(
                dataset_name=dataset,
                providers=providers,
                mode=mode,
                condition=condition,
                max_items=int(max_items) if max_items.strip() else None,
                run_id=run_id,
                progress=progress,
            )
            score_run(run_id)
            _jobs[run_id]["status"] = "scored"
        except Exception as exc:  # noqa: BLE001
            _jobs[run_id]["status"] = "error"
            _jobs[run_id]["log"].append(str(exc))

    asyncio.create_task(job())
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def show_run(request: Request, run_id: str):
    path = settings.runs_dir() / run_id
    meta = {}
    if (path / "run.json").exists():
        meta = json.loads((path / "run.json").read_text())
    job = _jobs.get(run_id) or {}
    scores = None
    if (path / "scores.json").exists():
        scores = json.loads((path / "scores.json").read_text())
        status = "scored"
    else:
        status = job.get("status") or ("finished" if meta.get("finished_at") else "running")
    return templates.TemplateResponse(request, "run.html", {
        "run_id": run_id,
        "meta": meta,
        "status": status,
        "log": job.get("log") or [],
        "scores": scores,
    })


@app.get("/runs/{run_id}/progress", response_class=HTMLResponse)
def progress(request: Request, run_id: str):
    path = settings.runs_dir() / run_id
    job = _jobs.get(run_id) or {}
    if (path / "scores.json").exists():
        scores = json.loads((path / "scores.json").read_text())
        return templates.TemplateResponse(request, "_scored.html", {"run_id": run_id, "scores": scores})
    status = job.get("status") or "running"
    return templates.TemplateResponse(request, "_progress.html", {
        "run_id": run_id,
        "status": status,
        "log": job.get("log") or [],
    })


@app.get("/runs/{run_id}/clips", response_class=HTMLResponse)
def clips(request: Request, run_id: str):
    path = run_dir(run_id)
    utterances = []
    raw = path / "utterances.jsonl"
    if raw.exists():
        utterances = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    utterances.sort(key=lambda row: row.get("wer") or 0, reverse=True)
    return templates.TemplateResponse(request, "clips.html", {"run_id": run_id, "utterances": utterances[:200]})


@app.get("/runs/{run_id}/raw")
def raw_report(run_id: str):
    from fastapi.responses import FileResponse

    path = run_dir(run_id) / "report.html"
    if not path.exists():
        score_run(run_id)
    return FileResponse(path)
