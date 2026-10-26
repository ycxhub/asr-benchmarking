import json

from asrbench.providers.elevenlabs import ElevenLabsProvider
from asrbench.providers.murf import MurfProvider


def _murf():
    return MurfProvider("murf", {"lang_map": {"en-IN": {"language": "en_IN"}}, "url": "wss://example"})


def test_murf_parses_turninfo_wrapper():
    prov = _murf()
    partials = prov.parse(
        json.dumps({
            "type": "partial",
            "transcript": "hello",
            "rawEvent": {"type": "TurnInfo", "event": "StartOfTurn", "transcript": "hello"},
        })
    )
    assert [e.kind for e in partials] == ["partial"]
    assert partials[0].text == "hello"
    finals = prov.parse(
        json.dumps({
            "type": "final",
            "transcript": "hello world",
            "rawEvent": {"type": "TurnInfo", "event": "EndOfTurn", "trigger": "manual", "transcript": "hello world"},
        })
    )
    assert [e.kind for e in finals] == ["final", "turn_end", "endpoint"]
    assert finals[0].text == "hello world"


def test_murf_still_parses_results_envelope():
    prov = _murf()
    evs = prov.parse(json.dumps({"type": "Results", "transcript": "hi", "speech_final": True}))
    assert [e.kind for e in evs] == ["final", "endpoint", "turn_end"]


def test_elevenlabs_commits_on_audio_chunk():
    prov = ElevenLabsProvider(
        "elevenlabs",
        {"lang_map": {"en-IN": {"language_code": "en"}}, "url": "wss://example", "params": {}},
    )
    msg = json.loads(prov.endpoint_message())
    assert msg["message_type"] == "input_audio_chunk"
    assert msg["commit"] is True
    assert msg["audio_base_64"]


# --- Voice Arena Hindi top-6 adapters -------------------------------------------------------
import datetime as dt

import numpy as np
import pytest

from asrbench.providers import all_provider_cfgs, load_provider
from asrbench.providers.amazon import AmazonTranscribeProvider, decode_event, encode_event, presign_url
from asrbench.providers.batch import BatchHTTPProvider, split_pcm16
from asrbench.runner.session import run_item
from asrbench.runner.validate import _dataset

# File-upload models from the Voice Arena Hindi board. Left in the tree, not registered.
NON_STREAMING = ["bodhan", "sarvam-rest", "gemini-pro", "mai-transcribe", "elevenlabs-batch"]


STREAMING = ["sarvam", "gemini-live", "amazon-transcribe", "mai-streaming", "elevenlabs"]


def test_harness_registers_streaming_only():
    cfgs = all_provider_cfgs()
    for name in STREAMING:
        assert "hi" in cfgs[name]["lang_map"], name
        load_provider(name)
    for name in NON_STREAMING:
        assert name not in cfgs, name


def test_gemini_live_setup_and_parse():
    import asyncio
    from asrbench.providers.base import SessionContext
    from asrbench.providers.gemini_live import GeminiLiveProvider

    prov = GeminiLiveProvider("gemini-live", {"lang_map": {"hi": {"language_codes": ["hi-IN"]}}, "params": {}})
    sent = []

    async def fake_send(obj):
        sent.append(obj)

    prov.send_json = fake_send
    prov.ctx = SessionContext("hi", 16000, "forced")
    asyncio.run(prov.on_open(prov.ctx))
    setup = sent[0]["setup"]
    assert setup["model"] == "models/gemini-3.5-transcribe-live"
    assert setup["inputAudioTranscription"]["languageCodes"] == ["hi-IN"]
    assert setup["realtimeInputConfig"]["automaticActivityDetection"]["disabled"] is True
    assert json.loads(prov.endpoint_message()) == {"realtimeInput": {"activityEnd": {}}}

    assert [e.kind for e in prov.parse(json.dumps({"setupComplete": {}}))] == ["ready"]
    assert [e.kind for e in prov.parse(json.dumps({"serverContent": {"interimInputTranscription": {"text": "नम"}}}))] == ["partial"]
    finals = prov.parse(json.dumps({"serverContent": {"inputTranscription": {"text": "नमस्ते"}}}))
    assert [e.kind for e in finals] == ["final", "turn_end"] and finals[0].text == "नमस्ते"


def test_mai_streaming_parse_accumulates_and_commits():
    from asrbench.providers.mai_streaming import MAIStreamingProvider

    prov = MAIStreamingProvider("mai-streaming", {"lang_map": {"hi": {"language": "hi"}}, "params": {}})
    assert prov.parse(json.dumps({"type": "session.created"})) == [] and prov._created.is_set()
    assert [e.kind for e in prov.parse(json.dumps({"type": "session.updated"}))] == ["ready"]
    tx = "conversation.item.input_audio_transcription."
    assert prov.parse(json.dumps({"type": tx + "delta", "delta": "नमस्ते"}))[0].text == "नमस्ते"
    assert prov.parse(json.dumps({"type": tx + "intermediate", "intermediate": " दुनिया"}))[0].text == "नमस्ते दुनिया"
    prov.audio_message(b"\x00\x00")
    assert prov.finish_message() is not None
    assert json.loads(prov.endpoint_message()) == {"type": "input_audio_buffer.commit"}
    assert prov.finish_message() is None
    done = prov.parse(json.dumps({"type": tx + "completed", "transcript": "नमस्ते दुनिया!"}))
    assert [e.kind for e in done] == ["final", "endpoint", "turn_end"] and done[0].text == "नमस्ते दुनिया!"
    assert prov._committed_text == ""


def test_event_stream_round_trip():
    msg = encode_event({":event-type": "AudioEvent", ":message-type": "event"}, b"\x01\x02")
    headers, payload = decode_event(msg)
    assert headers == {":event-type": "AudioEvent", ":message-type": "event"}
    assert payload == b"\x01\x02"


def test_presign_url_is_deterministic():
    now = dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc)
    kw = dict(region="ap-south-1", access_key="AKID", secret_key="secret", now=now)
    a = presign_url("h:8443", "/p", {"language-code": "hi-IN"}, **kw)
    b = presign_url("h:8443", "/p", {"language-code": "hi-IN"}, **kw)
    assert a == b and "X-Amz-Signature=" in a and "language-code=hi-IN" in a
    assert "X-Amz-Credential=AKID%2F20260102%2Fap-south-1%2Ftranscribe%2Faws4_request" in a


def test_amazon_parses_partial_and_final():
    prov = AmazonTranscribeProvider("amazon-transcribe", {"lang_map": {"hi": {}}, "params": {}})
    body = {"Transcript": {"Results": [
        {"IsPartial": True, "Alternatives": [{"Transcript": "नमस्ते"}]},
        {"IsPartial": False, "Alternatives": [{"Transcript": "नमस्ते दुनिया"}]},
    ]}}
    raw = encode_event({":event-type": "TranscriptEvent", ":message-type": "event"}, json.dumps(body).encode())
    assert [e.kind for e in prov.parse(raw)] == ["partial", "final", "turn_end"]
    exc = encode_event({":exception-type": "BadRequestException", ":message-type": "exception"}, b'{"Message":"x"}')
    assert prov.parse(exc)[0].kind == "error"


def test_split_pcm16_cuts_at_silence():
    sr = 16000
    tone = (np.sin(np.arange(sr * 40) / 5) * 8000).astype(np.int16)
    tone[sr * 27 : sr * 28] = 0
    pieces = split_pcm16(tone.tobytes(), sr, 29.5)
    assert len(pieces) == 2
    assert 27 * sr <= len(pieces[0]) // 2 <= 28 * sr
    assert sum(len(p) for p in pieces) == len(tone.tobytes())


class _FakeBatch(BatchHTTPProvider):
    sample_rates = (16000,)
    max_segment_s = None

    async def transcribe(self, wav, ctx):
        assert wav[:4] == b"RIFF"
        return "turn"


@pytest.mark.parametrize("mode", ["forced", "native"])
async def test_batch_provider_through_session(tmp_path, mode):
    prov = _FakeBatch("fake", {"lang_map": {"en-US": {}}, "params": {}})
    ds, item = _dataset(tmp_path, [[0.1, 0.4], [0.6, 0.9]], 1.1)
    res = await run_item(prov, ds, item, mode=mode, condition="wideband", timeout_s=1.0, silence_tail_s=0.2, frame_ms=20)
    assert res.ok, res.error
    assert res.hypothesis == ("turn turn" if mode == "forced" else "turn")


def test_gemini_skips_thought_parts():
    import asyncio
    import httpx
    from asrbench.providers.gemini import GeminiProvider

    def handler(req):
        body = json.loads(req.content)
        assert body["contents"][0]["parts"][1]["inlineData"]["mimeType"] == "audio/wav"
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"text": "thinking...", "thought": True}, {"text": "नमस्ते"}]}}]})

    prov = GeminiProvider("gemini-pro", {"lang_map": {"hi": {"language": "Hindi"}}, "url": "https://x", "params": {}})
    prov.api_key = lambda: "k"
    prov._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    from asrbench.providers.base import SessionContext
    assert asyncio.run(prov.transcribe(b"RIFF", SessionContext("hi", 16000, "forced"))) == "नमस्ते"


def test_mai_reads_combined_phrases(monkeypatch):
    import asyncio
    import httpx
    from asrbench.providers.base import SessionContext
    from asrbench.providers.microsoft import MAITranscribeProvider

    monkeypatch.setenv("AZURE_SPEECH_ENDPOINT", "https://res.cognitiveservices.azure.com")

    def handler(req):
        assert req.url.path == "/speechtotext/transcriptions:transcribe"
        assert b"MAI-Transcribe-2" in req.content and b"hi-IN" in req.content
        return httpx.Response(200, json={"combinedPhrases": [{"text": "नमस्ते दुनिया"}]})

    prov = MAITranscribeProvider("mai-transcribe", {"lang_map": {"hi": {"locales": ["hi-IN"]}}, "params": {}})
    prov.api_key = lambda: "k"
    prov._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert asyncio.run(prov.transcribe(b"RIFF", SessionContext("hi", 16000, "forced"))) == "नमस्ते दुनिया"
