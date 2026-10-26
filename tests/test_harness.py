import pytest

from asrbench.providers.base import SessionContext
from asrbench.providers.mock import MockProvider
from asrbench.runner.pacer import Pacer
from asrbench.runner.session import plan_turn_frames, run_item
from asrbench.runner.validate import _dataset, _synthetic, validate


def test_turn_frames():
    recs = plan_turn_frames([[0.5, 2.5], [3.0, 3.01]], 0.02, 200)
    assert (recs[0].start_frame, recs[0].end_frame) == (25, 124)
    assert (recs[1].start_frame, recs[1].end_frame) == (150, 150)


async def test_pacer_stays_close_to_deadline():
    pacer = Pacer(b"\x00" * 3200, 16000, frame_ms=20)
    for _ in range(10):
        await pacer.next_frame()
    assert pacer.p99_lateness_ms() < 15


async def test_injected_delay_recovered():
    report = await validate(delays_ms=(300,), repeats=2, concurrency=1)
    assert report["pass"], report
    forced = [r for r in report["rows"] if r["mode"] == "forced"][0]
    assert abs(forced["measured_p50_ms"] - 300) <= 5


async def test_final_only_and_transcript(tmp_path):
    turns = [[0.2, 0.5], [0.7, 1.0]]
    prov = MockProvider(
        "m",
        {"params": {"finalize_delay_ms": 40, "final_only": True}, "lang_map": {"en-US": {}}},
    )
    ds, item = _dataset(tmp_path, turns, 1.2)
    item.reference = "alpha beta gamma delta"
    ctx_lang = SessionContext("en-US", 16000, "forced", oracle_text=item.reference, oracle_turns=turns)
    assert ctx_lang.oracle_text
    res = await run_item(prov, ds, item, mode="forced", condition="wideband", timeout_s=2.0, silence_tail_s=0.25, frame_ms=20)
    assert res.ok, res.error
    assert res.hypothesis.split() == ["alpha", "beta", "gamma", "delta"]
    assert all(t.t_endpoint_ack is not None for t in res.turns)
    assert all(t.eos_to_first_partial_ms is None for t in res.turns)


async def test_connection_failure_is_error(tmp_path):
    prov = MockProvider("m", {"params": {"fail_connect": True}, "lang_map": {"en-US": {}}})
    ds, item = _dataset(tmp_path, [[0.1, 0.3]], 0.5)
    res = await run_item(prov, ds, item, mode="forced", condition="wideband", timeout_s=1.0, silence_tail_s=0.2, frame_ms=20)
    assert not res.ok
    assert res.error and res.error.startswith("connect:")


def test_synthetic_bytes_cover_turns():
    turns = [[0.1, 0.3]]
    assert len(_synthetic(turns, 0.5)) == 0.5 * 16000 * 2
