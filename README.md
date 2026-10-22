# ClassLens AI

ClassLens AI is a local Streamlit application for evidence-based review of recorded classes. It accepts a new recording, inspects media locally, and—only after explicit permission—can use Deepgram and OpenAI to create a cited, structured assessment.

## Privacy first

`ALLOW_EXTERNAL_PROCESSING=false` is the default. In this mode the app never sends recordings, audio, transcripts, or frames to Deepgram or OpenAI. It can still validate uploads, inspect streams, extract audio locally, calculate signal measurements, parse an optional VTT upload, and sample frames locally. It intentionally does **not** present this local inspection as a teaching-quality rating.

Only set `ALLOW_EXTERNAL_PROCESSING=true` after the organiser has approved the data processing. Keep `.env` local; it is ignored by Git. The app sends only extracted WAV audio to Deepgram when a paired transcript is unavailable, and sends a bounded transcript excerpt, local audio measurements, and modest sampled frames to OpenAI. It does not send the dataset in bulk. Temporary files live inside a per-run temporary directory and are deleted after processing.

## Setup

Prerequisites: Python 3.10+, FFmpeg (including `ffprobe`), and optionally an OpenAI/Deepgram account after permission is confirmed.

```bash
cd /Users/vanshikaarora/Desktop/HumanitiAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

For the approved external workflow, add the API keys to `.env` and explicitly set:

```dotenv
ALLOW_EXTERNAL_PROCESSING=true
```

Use `OPENAI_MODEL` to choose a deployed vision-capable model that supports Responses structured outputs. Do not enable the flag merely because keys exist.

## Architecture

| Module | Responsibility |
| --- | --- |
| `app.py` | Streamlit workflow, progress, errors, and presentation |
| `media_processing.py` | File validation, `ffprobe`, safe local audio extraction |
| `transcription.py` | Timestamped VTT parser and permission-gated Deepgram client |
| `audio_analysis.py` | Local speech/silence and clipping review measurements |
| `video_analysis.py` | Stratified, resized frame sampling with timestamps |
| `llm_assessment.py` | Permission-gated, structured OpenAI evidence assessment |
| `schemas.py` | Transcript/evidence schema and citation validation |
| `evaluator.py` | Deterministic weighted score and four-class thresholds |
| `dataset.py` | Reference-recording discovery only; never prediction features |

## Assessment method

The initial 100-point rubric is configurable in `config.py`:

- Lesson structure and clarity: 25
- Teaching explanations and examples: 25
- Checks for understanding and feedback: 25
- Inclusive, purposeful interaction: 15
- Technical delivery and usable visual/audio support: 10

Initial score bands are Poor `<50`, Average `50–69`, Good `70–84`, and Excellent `85–100`. These are hypotheses to calibrate, not validated claims. The model proposes per-dimension scores; Python computes the final weighted score and label separately. Technical flags remain distinct from the ordinal judgement.

The supplied `weak` reference folder is surfaced as a proposed `Poor` mapping but is explicitly marked uncertain; `quality issue` and `demo` are non-ordinal reference cohorts. Names, batch IDs, filenames, dates, and reference labels are not input features for a new-recording prediction.

## Evidence safeguards

Each observation must cite a VTT segment or a sampled frame. The app rejects model output with a timestamp outside the recording, transcript quotations that do not occur in the transcript, invalid frame references, invalid score ranges, or missing rubric dimensions. Audio signal data is presented as a review candidate only. It does not infer emotions, engagement, attention, learning outcomes, or teaching quality from appearance, silence, accent, loudness, camera presence, slide count, or raw student-utterance counts.

## Tests

Run the standard-library test suite without external credentials or recordings:

```bash
python3 -m unittest discover -s tests -v
```

Run on 2026-10-22: `python -m unittest discover -s tests -v` completed with **13 passing tests**. Coverage includes VTT parsing, label-mapping uncertainty, scoring thresholds, invalid uploads, missing streams, malformed/ungrounded/contradictory model output, the external-processing gate, mocked API timeout handling, unavailable OpenCV, and temporary-workspace cleanup.

A local verification processed one supplied 37.9-minute reference MP4 without external calls: FFmpeg found 1920×1080 video and AAC audio, 329 VTT cues were reused, extracted-audio duration aligned with the VTT/media duration, and four sampled-frame timestamps fell within the recording. This verifies the local components only; it is not an instructional-quality evaluation and does not validate cloud assessment.

API timeout/failure handling is implemented with actionable errors but requires approved credentials and a permitted test recording to verify end-to-end.

## Limitations and next steps

- FFmpeg is required for complete media analysis; the current workspace does not include it.
- No end-to-end cloud request has been made, by design, because permission is not confirmed.
- The source directory has not yet been initialized as a Git repository; create a private repository and verify ignored files before committing.
- The tiny reference set is not suitable for training or claiming generalisation.
- Conduct a permission-approved, held-out evaluation before adopting the initial thresholds.
