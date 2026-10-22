
"""Data models and validation for ClassLens AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re
import unicodedata

from config import SETTINGS


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    speaker: str | None
    text: str


@dataclass(frozen=True)
class Evidence:
    observation: str
    modality: str
    timestamp: float | None = None
    excerpt: str | None = None
    frame_reference: str | None = None


@dataclass
class LocalInspection:
    metadata: dict[str, Any] = field(default_factory=dict)
    transcript: list[TranscriptSegment] = field(default_factory=list)
    audio: dict[str, Any] = field(default_factory=dict)
    frames: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


REQUIRED_DIMENSIONS = {
    "structure_clarity",
    "explanations_examples",
    "checks_feedback",
    "inclusive_interaction",
    "technical_delivery",
}


def normalize_evidence_text(value: str) -> str:
    """Normalize Unicode, case, punctuation, and whitespace."""
    value = unicodedata.normalize("NFKC", value)
    value = value.casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _transcript_contains_excerpt(
    excerpt: str,
    transcript_text: str,
    min_words: int = 3,
) -> bool:
    """Check normalized evidence text against the supplied transcript."""
    normalized_excerpt = normalize_evidence_text(excerpt)
    normalized_transcript = normalize_evidence_text(transcript_text)

    if len(normalized_excerpt.split()) < min_words:
        return False

    if not normalized_transcript:
        return False

    # Exact normalized match is preferred.
    if normalized_excerpt in normalized_transcript:
        return True

    # Allow minor transcription/formatting differences, but require a
    # substantial matching sequence. Short quotes must match exactly.
    excerpt_words = normalized_excerpt.split()
    transcript_words = normalized_transcript.split()

    if len(excerpt_words) < 6:
        return False

    # Find the longest contiguous word sequence shared by the quote and
    # transcript. This tolerates a small number of differing words.
    required_match = max(6, int(len(excerpt_words) * 0.8))
    required_match = min(required_match, len(excerpt_words))

    for start in range(len(excerpt_words) - required_match + 1):
        sequence = excerpt_words[start : start + required_match]
        sequence_len = len(sequence)

        for transcript_start in range(
            max(0, len(transcript_words) - sequence_len + 1)
        ):
            if transcript_words[
                transcript_start : transcript_start + sequence_len
            ] == sequence:
                return True

    return False


def validate_llm_assessment(
    payload: Any,
    duration: float | None,
    transcript: list[TranscriptSegment],
    frame_refs: set[str],
) -> dict[str, Any]:
    """Validate assessment structure and ensure evidence is grounded."""

    if not isinstance(payload, dict):
        raise ValueError("Assessment must be a JSON object.")

    if payload.get("class_label") not in {
        "Poor",
        "Average",
        "Good",
        "Excellent",
    }:
        raise ValueError("Invalid class_label.")

    score = payload.get("score")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not 0 <= score <= 100
    ):
        raise ValueError("score must be 0-100.")

    dims = payload.get("dimension_scores")
    if not isinstance(dims, dict) or set(dims) != REQUIRED_DIMENSIONS:
        raise ValueError("Missing or unknown rubric dimensions.")

    for name, item in dims.items():
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("score"), (int, float))
            or isinstance(item.get("score"), bool)
            or not 0 <= item["score"] <= 100
            or not isinstance(item.get("explanation"), str)
            or not item["explanation"].strip()
        ):
            raise ValueError(f"Invalid score for {name}.")

    weighted_score = sum(
        dims[name]["score"] * SETTINGS.rubric_weights[name] / 100
        for name in REQUIRED_DIMENSIONS
    )

    if abs(weighted_score - score) > 5:
        raise ValueError(
            "Model overall score conflicts with its dimension scores."
        )

    evidence_items = payload.get("evidence")
    if not isinstance(evidence_items, list) or not evidence_items:
        raise ValueError("At least one grounded evidence item is required.")

    transcript_text = " ".join(
        segment.text
        for segment in transcript
        if isinstance(segment.text, str)
    )

    valid_evidence = []

    for item in evidence_items:
        if (
            not isinstance(item, dict)
            or item.get("modality") not in {"transcript", "audio", "video"}
            or not isinstance(item.get("observation"), str)
            or not item["observation"].strip()
        ):
            continue

        ts = item.get("timestamp")
        if ts is not None and (
            not isinstance(ts, (int, float))
            or isinstance(ts, bool)
            or ts < 0
            or (duration is not None and ts > duration + 1)
        ):
            continue

        if item["modality"] == "transcript":
            excerpt = item.get("excerpt")

            if (
                not isinstance(excerpt, str)
                or not _transcript_contains_excerpt(
                    excerpt,
                    transcript_text,
                )
            ):
                # Drop this unsupported evidence item instead of failing
                # the entire assessment.
                continue

        if (
            item["modality"] == "video"
            and item.get("frame_reference") not in frame_refs
        ):
            continue

        valid_evidence.append(item)

    if not valid_evidence:
        raise ValueError(
            "Transcript evidence is not present in the supplied transcript, "
        "or no other valid evidence remained after validation. "
        "Please retry or provide a matching VTT transcript."
        )

    # Return only evidence that passed validation.
    payload["evidence"] = valid_evidence

    return payload
