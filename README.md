# asrbench

Streaming speech-to-text benchmark for a voice agent. It measures word error rate and end-of-speech latency for Murf against Sarvam, Cartesia, Deepgram, and ElevenLabs.

The measurement rules are in [METHODOLOGY.md](METHODOLOGY.md) (MSB v1.0). The held-out collection spec is in [docs/INDIC_AGENTTALK_SPEC.md](docs/INDIC_AGENTTALK_SPEC.md).

## Setup

```bash
uv sync --extra dev
cp .env.example .env
```

Put provider keys in `.env` when you have them. `HF_TOKEN` is required for IndicVoices, Lahaja, and Svarah after you accept each dataset's terms on Hugging Face. The smoke suite and the mock provider need neither.

Audio, models, and run output go to `ASRBENCH_HOME` (default `~/asrbench-data`), outside this iCloud folder.

## Commands

```bash
uv run asrbench validate
uv run asrbench ingest smoke
uv run asrbench run --dataset smoke-en --providers mock --max-items 2
uv run asrbench score <run_id>
uv run asrbench list-providers
uv run asrbench serve
```

`validate` checks that a 300 ms delay injected by the mock provider is measured within 5 ms. `run` writes `$ASRBENCH_HOME/runs/<run_id>/` with the config snapshot, raw provider messages, `scores.json`, and `report.html`. `serve` opens the local UI at http://127.0.0.1:8000.

`asrbench run` refuses to start a provider whose API key is missing.

## What is not done yet

- Public suites are ingested locally. The first published comparison and Artificial Analysis calibration still need a full live run, not a 1-clip smoke.
- No live provider has been called. Add the keys, then run a small `--max-items` pass before a full suite.
- Hindi script normalization uses `configs/normalization/hi_lexicon.tsv` plus a small romanization fallback. It does not use IndicXlit. `asrbench translit-mine <run_id>` lists Latin tokens that still need a lexicon row.
- The first published comparison, the Artificial Analysis calibration, and a Mumbai VM run are waiting on those keys and on the ingested audio.
