from __future__ import annotations

import sqlite3

from . import settings


def _connect() -> sqlite3.Connection:
    db = settings.home() / "index.sqlite"
    con = sqlite3.connect(db)
    con.execute(
        """
        create table if not exists runs (
            run_id text primary key,
            created_at text,
            finished_at text,
            dataset text,
            lang text,
            mode text,
            condition text,
            providers text,
            n_items integer,
            path text
        )
        """
    )
    return con


def upsert_run(meta: dict, path: str) -> None:
    with _connect() as con:
        con.execute(
            """
            insert into runs (run_id, created_at, finished_at, dataset, lang, mode, condition, providers, n_items, path)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(run_id) do update set
                finished_at=excluded.finished_at,
                n_items=excluded.n_items,
                path=excluded.path
            """,
            (
                meta.get("run_id"),
                meta.get("created_at"),
                meta.get("finished_at"),
                meta.get("dataset"),
                meta.get("lang"),
                meta.get("mode"),
                meta.get("condition"),
                ",".join(meta.get("providers") or []),
                meta.get("n_items"),
                path,
            ),
        )


def list_indexed_runs() -> list[dict]:
    if not (settings.home() / "index.sqlite").exists():
        return []
    with _connect() as con:
        con.row_factory = sqlite3.Row
        return [dict(row) for row in con.execute("select * from runs order by created_at desc")]
