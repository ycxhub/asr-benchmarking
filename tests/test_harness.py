import pytest

from asrbench.providers.base import SessionContext
from asrbench.providers.mock import MockProvider
from asrbench.runner.session import SessionConfig, plan_turn_frames, run_session
from asrbench.runner.validate import _synthetic, validate


def test_turn_frames():
    recs = plan_turn_frames([[0.5, 2.5], [3.0, 3.01]], 0.02, 200)
    assert (recs[0].start_frame, recs[0].end_frame) == (25, 124)
    assert (recs[1].start_frame, recs[1].end_frame) == (150, 150)


async def test_injected_delay_recovered():
    report = await validate(delays_ms=(300,), repeats=2)
    forced = [r for r in report["rows"] if r["mode"] == "forced"][0]
    assert forced["pass"], report
    assert abs(forced["measured_p50_ms"] - 300) <= 5


async def test_final_only_and_transcript():
    turns = [[0.5, 1.5], [2.2, 3.0]]
    prov = MockProvider("m", {"params": {"finalize_delay_ms": 50, "jitter_ms": 0, "final_only": True}, "lang_map": {"en-US": {}}})
    ctx = SessionContext("en-US", 16000, "forced", oracle_text="alpha beta gamma delta", oracle_turns=turns)
    res = await run_session(prov, _synthetic(turns, 3.4), 16000, turns, ctx, SessionConfig(mode="forced"))
    assert res.status == "ok"
    assert res.transcript.split() == ["alpha", "beta", "gamma", "delta"]
    assert all(t["resolution"] == "endpoint" for t in res.turns)
    assert all(t["eos_to_first_partial_ms"] is None for t in res.turns)


async def test_connection_failure_is_error():
    prov = MockProvider("m", {"params": {}, "lang_map": {"en-US": {}}})
    prov.url = lambda ctx: "ws://127.0.0.1:9/nope"
    prov.port = 9

    async def no_server(ctx):
        from asrbench.providers.base import WebSocketProvider
        await WebSocketProvider.connect(prov, ctx)

    prov.connect = no_server
    ctx = SessionContext("en-US", 16000, "forced", oracle_text="x", oracle_turns=[[0.1, 0.5]])
    res = await run_session(prov, _synthetic([[0.1, 0.5]], 0.8), 16000, [[0.1, 0.5]], ctx, SessionConfig())
    assert res.status == "error"
    assert res.turns[0]["resolution"] == "error"
