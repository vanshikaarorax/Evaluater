
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from config import SETTINGS
from media_processing import (
    extract_mono_wav,
    inspect_media,
    validate_upload,
)
from transcription import parse_vtt, transcribe_deepgram
from audio_analysis import analyse_wav
from video_analysis import sample_frames
from evaluator import deterministic_assessment, local_inspection_summary
from llm_assessment import assess_with_openai


st.set_page_config(
    page_title="ClassLens AI",
    page_icon="🔎",
    layout="wide",
)

st.title("ClassLens AI")
st.caption(
    "Evidence-based classroom assessment: structured observations, "
    "not assumptions about students or teachers."
)

if SETTINGS.allow_external_processing:
    st.warning(
        "External processing is enabled. Classroom audio, transcripts, "
        "and sampled video frames may be sent to external AI services "
        "during assessment."
    )
else:
    st.info(
        "Privacy: external processing is disabled. No recording, transcript, "
        "audio, or frame is sent to Deepgram or OpenAI in local-inspection mode."
    )

with st.expander("Method and limits"):
    st.write(
        "The app inspects media, uses an explicitly uploaded timestamped VTT "
        "transcript when provided, and otherwise transcribes the recording's "
        "audio with Deepgram when external processing is enabled. It applies "
        "an initial 100-point rubric. Automated observations are estimates "
        "that require evidence-based review."
    )
    st.json(SETTINGS.rubric_weights)

upload = st.file_uploader(
    "Upload a class recording",
    type=["mp4", "mov", "mkv", "avi", "webm"],
)

transcript_upload = st.file_uploader(
    "Optional timestamped VTT transcript",
    type=["vtt"],
    help=(
        "If provided, this exact transcript will be used. If omitted, "
        "Deepgram will transcribe the recording's audio when external "
        "processing is enabled."
    ),
)

if upload and st.button("Assess recording", type="primary"):
    with tempfile.TemporaryDirectory(prefix="classlens_") as tmp:
        root = Path(tmp)
        source = root / Path(upload.name).name
        source.write_bytes(upload.getbuffer())

        try:
            validate_upload(source, SETTINGS.max_upload_mb)

            progress = st.progress(0, text="Inspecting media…")
            limitations = []

            try:
                metadata = inspect_media(source)
            except Exception as exc:
                metadata = {}
                limitations.append(
                    f"Media inspection unavailable: {exc}"
                )

            # Use only the transcript explicitly uploaded by the user.
            # Never search the recording's folder for a guessed transcript.
            progress.progress(20, text="Loading transcript…")

            if transcript_upload is not None:
                transcript_text = transcript_upload.getvalue().decode(
                    "utf-8-sig"
                )
                transcript = parse_vtt(transcript_text)
            else:
                transcript = []

            # Extract and analyse audio when the recording has an audio stream.
            audio = {}
            wav = None

            if metadata.get("has_audio"):
                try:
                    wav = extract_mono_wav(
                        source,
                        root / "audio.wav",
                    )
                    audio = analyse_wav(wav)
                except Exception as exc:
                    limitations.append(
                        f"Audio analysis unavailable: {exc}"
                    )
            else:
                limitations.append(
                    "No audio stream was verified."
                )

            # Sample video frames independently of the transcript.
            progress.progress(45, text="Sampling video frames…")

            frames = []

            if metadata.get("has_video"):
                try:
                    frames = sample_frames(
                        source,
                        root / "frames",
                        SETTINGS.frame_count,
                    )
                except Exception as exc:
                    limitations.append(
                        f"Frame sampling unavailable: {exc}"
                    )
            else:
                limitations.append(
                    "No video stream was verified."
                )

            progress.progress(65, text="Preparing assessment…")

            if SETTINGS.allow_external_processing:
                # No uploaded transcript means Deepgram must transcribe
                # the audio. Do not guess or load another VTT file.
                if not transcript:
                    if wav is None:
                        raise ValueError(
                            "No VTT transcript was uploaded and no audio "
                            "could be extracted for Deepgram transcription."
                        )

                    progress.progress(
                        70,
                        text="Transcribing audio with Deepgram…",
                    )

                    transcript = transcribe_deepgram(wav)

                    if not transcript:
                        raise ValueError(
                            "Deepgram returned no transcript. Please check "
                            "the recording's audio or upload a valid "
                            "timestamped VTT transcript."
                        )

                progress.progress(
                    80,
                    text="Generating assessment…",
                )

                model = assess_with_openai(
                    transcript,
                    audio,
                    frames,
                    metadata.get("duration"),
                )

                dimension_scores = {
                    name: item["score"]
                    for name, item in model["dimension_scores"].items()
                }

                final = deterministic_assessment(
                    dimension_scores,
                    limitations + model.get("limitations", []),
                )

                st.success(
                    f"{final['class_label']} — "
                    f"{final['final_score']}/100"
                )

                st.caption(
                    f"AI-proposed result: {model['class_label']} "
                    f"({model['score']}/100). The displayed final score "
                    "uses the configured deterministic rubric."
                )

                st.subheader("Dimension scores")

                for name, item in model["dimension_scores"].items():
                    score = item.get("score", 0)
                    explanation = item.get("explanation", "")

                    st.markdown(
                        f"**{name.replace('_', ' ').title()} "
                        f"— {score}/100**"
                    )
                    st.progress(
                        max(0, min(100, int(score)))
                    )
                    st.write(explanation)

                st.subheader("Strengths")

                strengths = model.get("strengths", [])

                if strengths:
                    for strength in strengths:
                        st.markdown(f"- {strength}")
                else:
                    st.write("No strengths were returned.")

                st.subheader("Actionable improvements")

                improvements = model.get("improvements", [])

                if improvements:
                    for improvement in improvements:
                        st.markdown(f"- {improvement}")
                else:
                    st.write("No improvements were returned.")

                st.subheader("Evidence")

                evidence = model.get("evidence", [])

                if isinstance(evidence, list):
                    for index, item in enumerate(evidence, start=1):
                        with st.container(border=True):
                            st.markdown(f"**Evidence {index}**")

                            if isinstance(item, dict):
                                for key, value in item.items():
                                    st.markdown(
                                        f"**{key.replace('_', ' ').title()}:** "
                                        f"{value}"
                                    )
                            else:
                                st.write(item)

                elif isinstance(evidence, dict):
                    for key, value in evidence.items():
                        st.markdown(
                            f"**{key.replace('_', ' ').title()}**"
                        )
                        st.write(value)
                else:
                    st.write(
                        evidence or "No evidence was returned."
                    )

                st.subheader("Technical flags")

                technical_flags = model.get("technical_flags", [])

                if technical_flags:
                    for flag in technical_flags:
                        st.markdown(f"- {flag}")
                else:
                    st.write("No technical flags were returned.")

                st.subheader("Confidence and limitations")

                st.write(
                    f"**Model confidence:** "
                    f"{model.get('confidence', 'Not reported')}"
                )

                all_limitations = (
                    final.get("limitations", [])
                    + model.get("limitations", [])
                )
                all_limitations = list(dict.fromkeys(all_limitations))

                if all_limitations:
                    for limitation in all_limitations:
                        st.markdown(f"- {limitation}")
                else:
                    st.write(
                        "No additional limitations were reported."
                    )

                counterevidence = model.get("counterevidence", [])

                st.markdown("**Counterevidence and uncertainty**")

                if counterevidence:
                    for item in counterevidence:
                        st.markdown(f"- {item}")
                else:
                    st.write("No counterevidence was returned.")

            else:
                # Privacy-preserving mode: no Deepgram or OpenAI calls.
                report = local_inspection_summary(
                    transcript,
                    audio,
                    metadata,
                )
                report["limitations"].extend(limitations)

                st.warning(
                    "Local-inspection mode: no four-class teaching rating "
                    "has been generated. Enable external processing to "
                    "use Deepgram and OpenAI."
                )
                st.json(report)

            if frames:
                st.subheader("Sampled video frames")

                st.image(
                    [frame["path"] for frame in frames],
                    caption=[
                        f"{frame['reference']} "
                        f"({frame['timestamp']}s)"
                        for frame in frames
                    ],
                    width=220,
                )

            progress.progress(
                100,
                text="Assessment finished.",
            )

        except Exception as exc:
            st.error(
                f"Could not process this recording: {exc}. "
                "Check the file format, FFmpeg installation, "
                "API settings, and transcript format."
            )
