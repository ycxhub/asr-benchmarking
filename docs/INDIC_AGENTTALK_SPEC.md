# IndicAgentTalk — held-out dataset spec

Status: spec only. Collection is a separate project. This set becomes the headline once it exists. Public suites in `configs/suites/` stay available for calibration and for comparisons before this set is ready.

MSB v1.0 scores it with the same harness, the same normalizer, and the same forced-endpoint latency metrics as every public suite. It is never mixed into a table with a run that used a different `NORM_VERSION`.

## Purpose

Public Hindi and Indian-English sets are not conversations in the verticals Murf's voice agents actually handle, and any of them may appear in a vendor's training data. IndicAgentTalk is the held-out set those comparisons move to.

## Size

3 to 5 hours of speech per language:

- Hindi, including Hindi-English code-mixed turns
- Indian English

US English is out of scope for this set.

Clips are one caller turn, or a short run of turns from the same call, targeting 2 to 30 seconds of speech. A session may contain several turns. The ingest VAD stores the boundaries; collectors do not have to pre-cut on silence.

## Verticals

Cover all of these. Do not let one vertical exceed about twice the audio of the smallest one.

- Banking
- Insurance
- Lending
- Real estate
- D2C / commerce
- Personal assistants
- Customer support
- Outbound sales

## What the speech must contain

Entity-dense, spoken the way a caller says it, not read from a prompt card:

- Amounts in lakh and crore, including fractions such as ढाई लाख / 2.5 lakh
- Account numbers, phone numbers, OTPs spoken as digits
- PAN, IFSC, UPI IDs
- Dates and times
- Person and place names

Include both careful speech and ordinary disfluency. Do not script the references; transcribe what was said.

## Audio conditions

Mix, and tag each clip:

- 16 kHz wideband and 8 kHz telephony
- Quiet room and ordinary background noise (street, office, fan)
- Mobile handset and headset

The harness applies its own G.711 mu-law condition at run time. Collectors still record native 8 kHz calls when the source is a phone call, and store that as the telephony original rather than a downsampled wideband file.

## Transcription

- Verbatim.
- Hindi words in Devanagari.
- English words in Latin script, including English words inside a Hindi sentence.
- Digits as spoken words or digit sequences, consistently within a clip, and never normalized to a canonical amount in the reference. The scorer's normalizer does that.
- No punctuation that changes the word sequence. Fillers that were spoken stay in the reference.
- One reference per clip. Do not keep alternate transcripts in the manifest; adjudication happens before ingest.

## Three-pass verification

1. **Pass A.** A transcriber who speaks the clip's language writes the reference without seeing another transcript.
2. **Pass B.** A second transcriber writes an independent reference. Disagreements are marked at word level, not resolved by the second person editing the first draft in place.
3. **Pass C.** A reviewer who did not transcribe the clip resolves every disagreement by listening again. Clips the reviewer cannot resolve are dropped, not averaged.

Keep the three drafts until the clip is accepted. The manifest stores only the accepted reference.

## What never happens to this audio

- It is not used to train, fine-tune, or prompt any model that will be scored on it.
- It is not uploaded to a public Hugging Face dataset.
- It is not sent to a provider except through a benchmark run.

## Manifest

Ingest with `asrbench ingest-folder`. The folder contains `manifest.csv` (or jsonl) and the audio files. Required columns: `audio`, `text`. Also provide `id`, `speaker`, `lang` (`hi` or `en-IN`), and tags: `vertical`, `sample_rate_hz`, `channel` (`wideband` or `telephony`), `noise` (`quiet` or `noisy`), `device`.

## Open decisions

These are not decided in this spec:

- Who records the calls, and whether any audio is from production traffic or all of it is acted.
- The consent and retention review for production audio.
- The hour split inside the 3 to 5 hour range, per vertical.
- Where the master files live, and who on the ML team signs the contamination statement that this set was not in training.
