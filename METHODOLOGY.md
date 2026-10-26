# Murf Streaming STT Benchmark - Methodology

Version: **MSB v1.0** (normalizer `NORM_VERSION = en-1.0 / hi-1.0`)
Status: frozen for the first Murf vs competitor comparison. Changes follow section 12.

This benchmark answers one question for voice-AI builders in India: *for a conversational
voice agent, how accurately and how quickly does each streaming speech-to-text model
transcribe what the end user said?* It measures accuracy (word error rate) and real-time
behaviour (latency) for Murf STT, Sarvam Saaras v3, Cartesia Ink-2, Deepgram Nova-3 / Flux
and ElevenLabs Scribe v2 Realtime, on Hindi (including Hindi-English code-mixed speech),
Indian English and US English.

It borrows from three public methodologies and states where it departs from them:

- **Artificial Analysis (AA)** - streaming latency with forced endpointing driven by an
  external Silero VAD; final-transcript WER on streamed audio.
- **VoiceArena** - language-aware, symmetric, data-driven normalization ("normalize
  variation in form, preserve variation in meaning"); corpus-level WER.
- **Coval** - pinned model IDs, pinned datasets (SHA-256), versioned normalizer,
  connection setup excluded from latency.

---

## 1. Principles

1. **Measure the voice-agent path.** Every model is called through its public *streaming*
   WebSocket API with audio paced in real time, exactly as a voice agent would.
2. **Identical treatment.** Same audio bytes, same frame size, same pacing, same client,
   same time window, same turn boundaries, same normalizer for every model.
3. **Vendor-recommended configuration.** Each vendor runs with its publicly documented
   configuration for the language and use case. No per-vendor tuning on the test set,
   no keyword boosting or prompts (unless every vendor receives the same list).
4. **Nothing is dropped.** Failed, timed-out or empty responses are scored, never removed.
5. **Everything is re-scorable.** Every raw provider message is stored with its receive
   timestamp, so metrics can be recomputed without calling any API again.
6. **Uncertainty is reported.** Every headline number carries a 95% confidence interval,
   and model-vs-model differences carry a paired significance test.

---

## 2. Evaluation modes

| Mode | Who decides the user finished speaking | Purpose |
| --- | --- | --- |
| **forced** (headline) | Our Silero VAD; the harness sends the provider's *force-endpoint* message | Apples-to-apples model latency, the way production stacks with their own turn detector use STT (AA method) |
| **native** (secondary) | The provider's built-in endpointing / turn detection | Shows end-to-end responsiveness when you rely on the vendor's turn detection |

Force-endpoint messages used in `forced` mode:

| Provider (model) | Force-endpoint message | Endpoint-complete signal |
| --- | --- | --- |
| Murf STT (beta) | `{"type":"Finalize"}` | `type: final` wrapping `TurnInfo` `EndOfTurn` (`trigger: manual`) |
| Deepgram Nova-3 | `{"type":"Finalize"}` | `Results` with `from_finalize: true` |
| Deepgram Flux | `{"type":"ForceEndTurn"}` | `TurnInfo` `EndOfTurn` |
| ElevenLabs Scribe v2 Realtime | `{"message_type":"input_audio_chunk", "commit": true, ...}` | next `committed_transcript` |
| Cartesia Ink-2 (manual endpoint) | text frame `finalize` | `flush_done` |
| Sarvam Saaras v3 realtime | `{"event":"speech_end"}` (`endpointing=manual`) | next `transcript.final` |

Cartesia's `/stt/turns/websocket` (auto) endpoint has no force-endpoint message, so it is
evaluated in `native` mode only. Results from the two modes are never mixed in one table.

---

## 3. Audio and streaming protocol

- **Canonical audio.** At ingest every clip is decoded, down-mixed to mono, resampled to
  16 kHz (soxr, high quality) and stored as 16-bit PCM WAV. The SHA-256 of the stored file
  is recorded in the manifest and verified before every run. No loudness normalization is
  applied (real calls are not loudness-normalized); clipping is reported.
- **Conditions.**
  - `wideband`: the canonical 16 kHz PCM16 audio.
  - `telephony`: the canonical audio is resampled to 8 kHz and passed through a G.711
    mu-law encode/decode round trip. It is sent at 8 kHz to providers that accept 8 kHz
    input and up-sampled back to 16 kHz for providers that do not. Codec damage is
    identical for everyone.
- **Pacing.** Audio is sent in **20 ms frames** at real-time pace using absolute,
  drift-corrected deadlines on a monotonic clock (`frame i` is due at `t0 + i x 20 ms`).
  A provider whose documentation requires larger frames gets its documented minimum; the
  frame size is recorded per run. Pacing lateness (actual send time minus deadline) is
  recorded; runs where p99 lateness exceeds 10 ms are flagged.
- **Open-mic tail.** After the last audio sample, the harness keeps streaming digital
  silence in real time until every turn is resolved or the timeout passes (section 5.4),
  as a live microphone would. The session is then closed with the provider's
  close-stream message.
- **Connection setup** (TCP + TLS + WebSocket upgrade + any session handshake) happens
  before `t0`, is measured separately and is excluded from all latency metrics.
- **Warm-up.** Before measurement, each provider receives one 1-second silent session
  that is not scored, so cold-start effects are not charged to the first sample.

---

## 4. Turn segmentation (external VAD)

- Silero VAD v5 (ONNX, 512-sample windows at 16 kHz; threshold 0.5, negative threshold
  0.35, minimum speech 250 ms, minimum silence 100 ms, 30 ms padding) runs once at ingest.
  Speech segments are stored in the manifest so every run uses identical boundaries. If the
  Silero model cannot be loaded, an energy VAD is used and the dataset is flagged.
- **Turns.** A clip is one turn from first speech onset to last speech offset. Clips whose
  speech span exceeds **30 s** are split into several turns at pauses of at least 500 ms,
  so that each turn is at most ~30 s where possible (AA uses ~30 s chunks for long audio).
  All turns of a clip are streamed in one session, as consecutive turns of one call.
- `speech_start(k)` and `speech_end(k)` are the VAD onset and offset of turn `k`.

---

## 5. Latency metrics

All times are wall-clock readings of `time.perf_counter()` on the client. Network transit
is included (it is part of what a voice agent experiences). The "endpoint moment" is the
instant **after** the last audio frame containing `speech_end(k)` has been written to the
socket **and** the force-endpoint message has been written - not the nominal file time.
This removes pacing slip on either side from the measurement (AA rule).

### 5.1 Headline: end-of-speech to final (`eos_to_final_ms`)

`forced` mode, per turn `k`:

1. `T_ep(k)` = endpoint moment of turn `k`.
2. If the provider emits an explicit endpoint-complete signal (table in section 2), the
   timer stops at the first such signal received after `T_ep(k)`.
3. Otherwise it stops at the first final transcript received after `T_ep(k)`.
4. If nothing arrives within the timeout but a non-empty final for this turn arrived
   *before* `T_ep(k)`, latency is recorded as 0 ms and flagged `pre_endpoint_final`
   (AA: "if the model already shared a final, use the most recent one").
5. Otherwise the turn is a **timeout** (counted in the timeout rate, excluded from latency
   percentiles, and its missing words are counted as deletions in WER).

`native` mode, per turn `k`: the timer starts when the last frame containing
`speech_end(k)` has been sent and stops at the provider's first end-of-turn final after it.
Turn ends emitted before `speech_end(k)` are counted as **premature turn ends**.

Reported: p50, p90, p95 and mean across turns (each turn is one sample), with a bootstrap
95% CI on p50.

### 5.2 Time to first token (`ttft_ms`)

Time from sending the frame that contains `speech_start(k)` to the first **non-empty**
transcript event (partial or final) for that turn. For providers that emit only finals
(Murf STT beta), this equals the time to the first final segment. Reported p50 and p90.

### 5.3 End-of-speech to first partial (`eos_to_first_partial_ms`)

AA's second streaming metric: time from `T_ep(k)` to the first non-empty partial. It is
undefined for final-only providers and reported as "n/a", not zero.

### 5.4 Timeouts

`forced` mode: 5 s after `T_ep(k)`. `native` mode: 8 s after `speech_end(k)` (longer than
the default end-of-turn timeouts of the providers). Both are recorded in the run config.

### 5.5 Network baseline

At the start of each run the harness measures TCP-connect round-trip time to every
provider host (median of 5). Tables show raw latency (headline). An "estimated
server-side" column subtracts one median RTT; it is an estimate, labelled as such.

---

## 6. Accuracy metrics

### 6.1 Hypothesis

The hypothesis for a clip is the concatenation, in arrival order, of every **final**
transcript segment the provider sent during the session (partials are never scored).

### 6.2 Word error rate

`WER = (S + D + I) / N`, with S/D/I from a Levenshtein word alignment (jiwer) between the
normalized reference and normalized hypothesis, `N` = reference words.

- **Headline aggregation: corpus-level pooled WER** = `sum(S+D+I) / sum(N)` over all
  clips of a dataset (VoiceArena, Coval ADR-024). Every word weighs the same.
- **Also reported:** audio-duration-weighted WER (AA convention), so numbers can be
  compared with AA.
- **Failures:** a clip with an error, timeout or empty output contributes all its
  reference words as deletions.
- Clips with an empty reference are rejected at ingest.

### 6.3 Hindi and Hinglish: script-normalized WER (SN-WER)

Hindi references and model outputs mix scripts arbitrarily ("account", "अकाउंट" and
"एकाउंट" are the same spoken word). Raw WER then measures orthography, not recognition.
The **headline for `hi` is SN-WER**: after the shared Hindi normalization (section 7.2),
every Latin-script token on **both** sides is mapped into Devanagari, and Devanagari
spelling variants of loanwords are folded onto one canonical form, using:

1. the versioned lexicon `configs/normalization/hi_lexicon.tsv` (human-reviewed);
2. a deterministic rule-based transliterator for tokens not in the lexicon.

Because the mapping is applied identically to reference and hypothesis it can only remove
script-mismatch errors; a wrong word stays wrong. Lexicon growth is data-driven
(section 12). Also reported for `hi`: **raw WER** (same normalization, no
transliteration) and **space-insensitive CER** (character error rate on SN-normalized
text with spaces removed, robust to word-segmentation variants such as "के लिए" vs
"केलिए").

### 6.4 Code-mix index

For each `hi` reference, the code-mix index is the share of tokens that are English
(Latin-script tokens, or the dataset's own English-word annotation when available).
SN-WER is reported per bucket: 0, (0, 0.1], (0.1, 0.25], (0.25, 0.5], > 0.5.

---

## 7. Normalization

One deterministic, idempotent function per language, applied identically to reference and
hypothesis. Versioned as `NORM_VERSION`; every scored result stores the version.

### 7.1 English (`en-IN`, `en-US`) - `en-1.0`

Base: OpenAI Whisper `EnglishTextNormalizer` (lower-casing, contraction expansion, spoken
numbers to digits, British to American spelling, filler removal, punctuation removal).
Additions, each discovered in published methodologies or in data:

1. Unicode NFKC; curly apostrophes and quotes to ASCII.
2. Apostrophe-dropped contractions are folded before the base normalizer
   (`dont` -> `don't` -> `do not`).
3. Digit-grouping commas are removed before parsing (`2,00,000`, `1,000`), so Indian
   and Western grouping collapse.
4. **Indian numbering:** `N lakh(s)/lac(s)` -> `N x 100000`, `N crore(s)` -> `N x 10^7`
   (`two lakh` = `2 lakh` = `200000`; `1.5 crore` = `15000000`).
5. Spaced digit groups are joined (`98 76 54` = `987654`), so phone/account number
   formatting does not count as errors.
6. Letter-spaced acronyms are joined (`t v` = `tv`, `p a n` = `pan`).
7. Stretched interjections and laughter are collapsed or removed (`sooo` -> `so`, `haha`).
8. Informal variants fold onto one form (`okay` = `ok`, `all right` = `alright`,
   `yep`/`yup` = `yeah`), plus a small compound-spacing table (`e mail` = `email`).

### 7.2 Hindi / Hinglish (`hi`) - `hi-1.0`

Whisper's `BasicTextNormalizer` is **not** used: it strips Devanagari vowel signs (matras).

1. Unicode NFC; Latin lower-cased; bracketed annotations (`[noise]`, `<laugh>`) removed.
2. IndicNLP Devanagari normalization; nukta removal (`ज़` = `ज`, `फ़` = `फ`); chandrabindu
   -> anusvara (`पाँच` = `पांच`); half-nasal consonant before a consonant -> anusvara
   (`हिन्दी` = `हिंदी`); candra vowels folded (`डॉक्टर` = `डोक्टर`); ZWJ/ZWNJ removed.
3. Danda and all punctuation removed; Devanagari digits -> ASCII digits.
4. Latin-script runs get English number normalization (`five thousand` = `5000`).
5. Hindi number words -> digits with the Indian system: 0-99 words, `सौ`, `हजार`,
   `लाख`, `करोड`, `अरब`, and fractional prefixes `डेढ़` (1.5), `ढाई` (2.5), `साढ़े` (+0.5),
   `सवा` (+0.25), `पौने` (-0.25): `डेढ़ लाख` = `150000`, `साढ़े तीन सौ` = `350`.
   Digit-by-digit speech (`नौ आठ सात`) becomes `987` via spaced-digit joining.
6. `₹N`, `Rs N`, `N रु` -> `N रुपये`; `%` -> `प्रतिशत`.
7. Pure non-lexical fillers removed (`हम्म`, `उम्म`, `hmm`, `um`, `uh`). Words with meaning
   (e.g. `हम` "we", `हाँ` "yes") are never removed.
8. SN-WER only: script normalization (section 6.3).

---

## 8. Statistics

- **Confidence intervals:** 1,000 bootstrap resamples (seed 1234) of clips - or of
  speakers when the dataset provides speaker IDs (cluster bootstrap) - recomputing pooled
  WER each time; the 2.5th and 97.5th percentiles form the 95% CI. Latency p50 CIs
  resample turns.
- **Paired comparison vs Murf:** on clips scored for both models, a paired bootstrap of
  `WER_model - WER_murf` gives a 95% CI and a two-sided p-value. A difference is called
  significant only if p < 0.05 **and** the CI excludes 0. The same is done for median
  latency.
- **Sample size:** at least 1,000 clips or 10,000 reference words per language is the
  target for the published headline (95% CI roughly +/-1 WER point). Smaller runs are
  marked "indicative".
- **Repeat runs:** latency is measured in at least two time slots on different days.
  The published latency is the median of the per-run medians; per-run values are kept.

---

## 9. Datasets and suites

Only datasets that are free to download and carry permissive licences are in the default
suites. Results are always shown per dataset; a per-language composite uses the declared
weights (renormalized over the datasets actually run).

| Suite | Dataset | Licence | Access | What it represents | Weight |
| --- | --- | --- | --- | --- | --- |
| `hi` | `ai4bharat/IndicVoices` (Hindi, valid split) | CC BY 4.0 | free HF login, auto-approved | spontaneous extempore + conversational Hindi, 400+ districts | 0.4 |
| `hi` | `dianavdavidson/MUCS-Hinglish` (MUCS 2021, test) | CC BY 4.0 | open | Hindi-English code-mixed speech (technical tutorials) | 0.3 |
| `hi` | `ai4bharat/Lahaja` (test) | MIT | free HF login, auto-approved | Hindi across 132 speakers / 83 districts, accent diversity | 0.3 |
| `en-IN` | `ai4bharat/Svarah` (test) | CC BY 4.0 | free HF login, auto-approved | Indian-accented English incl. payments, groceries, government services | 1.0 |
| `en-US` | `pipecat-ai/stt-benchmark-data` | CC BY 4.0 | open | short voice-agent turns | 0.5 |
| `en-US` | `ArtificialAnalysis/VoxPopuli-Cleaned-AA` | Apache-2.0 | open | read/prepared speech, AA-cleaned references | 0.25 |
| `en-US` | `ArtificialAnalysis/Earnings22-Cleaned-AA` | Apache-2.0 | open | long-form, accented, entity-dense speech | 0.25 |

Excluded from defaults: `soketlabs/CoSHE-Eval` (CC BY-NC, non-commercial), the Gnani
Hinglish telephony set (no longer public), FLEURS / Kathbath (read speech, saturated).

**Selection rule.** Items are drawn with a seeded (seed 1234) random sample from the
listed split, after filtering clips to 1-60 s with non-empty references. The sample size
per dataset is declared in the suite file; the manifest (ids + SHA-256) is the frozen
record of what was evaluated.

**Contamination.** All public sets may appear in any vendor's training data, including
Murf's. Murf's ML team must confirm which of these sets were *not* used for training; any
set used in training is reported but excluded from Murf's headline. The long-term headline
is the held-out IndicAgentTalk set (see `docs/INDIC_AGENTTALK_SPEC.md`).

**Calibration.** The AA VoxPopuli and Earnings22 sets let us compare our WER for Deepgram
and ElevenLabs with AA's published values. Agreement within ~1 WER point indicates that
our streaming harness and normalizer behave as expected; larger gaps are investigated
before results are published.

---

## 10. Provider configurations (pinned in `configs/providers.yaml`)

Exact model IDs and parameters are stored with every run. Language mapping:

| Provider | `hi` | `en-IN` | `en-US` | Notes |
| --- | --- | --- | --- | --- |
| Murf STT beta | `language=hi` | `language=en_IN` | `language=en_IN` | India region only; partials and finals wrapping TurnInfo |
| Sarvam `saaras:v3-realtime` | `hi-IN`, `mode=codemix` | `en-IN` | `en-IN` | `endpointing=manual` in forced mode |
| Cartesia `ink-2` | auto-detect | auto-detect | auto-detect | manual endpoint (forced) or turns endpoint (native) |
| Deepgram `nova-3` | `language=multi`, `endpointing=100` | `en-IN` | `en-US` | multi recommended for code-switching |
| Deepgram `flux-general-multi` / `flux-general-en` | multi, `language_hint=hi,en` | `flux-general-en` | `flux-general-en` | |
| ElevenLabs `scribe_v2_realtime` | `language_code=hi`, `secondary_languages=en` | `en` | `en` | `commit_strategy=manual` (forced), `vad` (native) |

US English is not a supported language for Murf or Sarvam; their English (India) model is
used and this is stated next to their US English results.

---

## 11. Client location and network

The target market is India, so measurements are taken **from India**. Two setups are
supported and the setup is recorded in every run:

- **Laptop in India (default, free).** Use a wired or stable connection, keep the machine
  awake (`caffeinate`), close bandwidth-heavy apps, and run all providers interleaved in
  the same session. Network RTT to each provider is recorded (section 5.5).
- **Cloud VM in Mumbai/Hyderabad (optional).** Closer to where production voice agents run;
  lower jitter. Same commands.

Results from different setups are not mixed in one table.

---

## 12. Fairness rules and change control

- Provider order is randomized per clip (seeded), and all providers run interleaved in
  the same time window, so time-of-day load affects everyone equally.
- Concurrency per provider is fixed (default 2) and recorded.
- The normalizer and lexicon are **frozen before** a Murf-vs-competitor comparison is
  scored. Changes follow the VoiceArena process: align hypotheses with references,
  extract error spans (`asrbench translit mine` lists script-mismatch candidates), have a
  human approve only pairs that are the *same word in a different form*, apply the fold
  symmetrically to every model, bump `NORM_VERSION`, and re-score all models.
- Scores produced under different `NORM_VERSION`s are never compared.

---

## 13. Reproducibility

Each run folder (`$ASRBENCH_HOME/runs/<run_id>/`) contains the full config snapshot,
provider configs, dataset manifest hashes, package version, normalizer version, lockfile
hash, git commit (if any), network baseline, and every raw provider message with receive
timestamps (`results.jsonl`). `asrbench score <run_id>` recomputes all metrics offline.

---

## 14. Known limitations

- Public datasets are not voice-agent conversations in the eight target verticals; the
  held-out IndicAgentTalk set addresses this.
- Some public references are imperfect (e.g. MUCS spells some English words in
  Devanagari, others in Latin). SN-WER reduces but does not remove this noise; the spot
  audit (50 highest-error clips per language) is part of every published run.
- Latency includes the public internet path from the client to each vendor's servers;
  vendors without servers in India are slower for Indian customers, which is a real
  effect but not a property of the model alone.
- Final-only providers have no partial-transcript metrics.

---

## 15. Version history

- **MSB v1.0** (October 2026) - initial release: forced and native modes, SN-WER for
  Hindi, Indian-numbering-aware English normalizer, paired bootstrap tests.
