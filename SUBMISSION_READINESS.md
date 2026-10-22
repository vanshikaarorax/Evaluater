# Submission materials

## Project explanation (242 words)

ClassLens AI is a privacy-first Streamlit application for assessing a new recorded class using timestamped evidence rather than opaque labels. It inspects media locally with FFmpeg, reuses an optional VTT transcript when supplied, measures audio signal conditions, and samples a small set of timestamped frames. These local stages are deliberately descriptive: silence, volume, accent, camera use, slide count, and student-utterance count are never converted directly into a teaching-quality judgement.

When the organiser explicitly authorises external processing, the application can transcribe missing audio through Deepgram and send a compact evidence bundle to an OpenAI vision-capable model. The model must return structured observations, counterevidence, technical flags, limitations, and citations. Python then validates transcript excerpts, timestamps, frame references, score ranges, dimension explanations, and consistency between the model’s overall and dimension scores. A configurable deterministic rubric calculates the final Poor/Average/Good/Excellent label separately. Technical issues remain separate from the ordinal label.

What worked: the local pipeline processed a supplied 37.9-minute recording without external calls, found matched AAC audio and 1080p video metadata, reused all 329 VTT cues, extracted an audio waveform, and sampled valid frames. Twelve automated tests pass.

What remains unverified: no external request was made because permission for children’s recordings has not been confirmed. The reference set is only eight ordinal examples; it is insufficient for model training or validating thresholds. “Weak” to “Poor” is a configurable proposed mapping, not an established ground truth. The app should therefore be presented as a transparent decision-support prototype pending organiser-approved evaluation on held-out recordings.

## Submission checklist

- [ ] Confirm organiser approval and a retention policy before any external processing.
- [ ] Confirm whether `Weak -> Poor` is the intended mapping.
- [ ] Run one approved end-to-end, held-out recording and document only aggregate results.
- [ ] Create a private repository; verify ignored media/credentials are absent before the first commit.
- [ ] Do not host recordings, VTTs, frames, audio, reports, or credentials publicly.
- [ ] If hosting is requested, use a private authenticated environment with server-side secrets, upload limits, encrypted storage, and automatic deletion.
- [ ] Link the actual repository and any approved private deployment in the Google Form.
- [ ] Include the genuine AI interaction history only if the challenge explicitly requires it; do not fabricate logs.
