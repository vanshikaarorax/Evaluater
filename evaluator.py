
"""Deterministic scoring and local-inspection helpers for ClassLens AI."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from config import SETTINGS


def label_for_score(score: float) -> str:
    """Return the configured label for a score between 0 and 100."""
    numeric_score = float(score)

    if not math.isfinite(numeric_score):
        raise ValueError("Score must be a finite number.")

    if not 0 <= numeric_score <= 100:
        raise ValueError("Score must be between 0 and 100.")

    for minimum, label in SETTINGS.thresholds:
        if numeric_score >= minimum:
            return label

    # Configuration validation guarantees that Poor starts at zero.
    raise RuntimeError("No grade label matched the score.")


def deterministic_assessment(
    dimension_scores: Mapping[str, float],
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    """Calculate a weighted overall score without altering dimension scores.

    The result is a weighted mean of the five rubric dimensions.
    Recording limitations are reported separately and do not automatically
    increase or decrease the score.
    """
    if not isinstance(dimension_scores, Mapping):
        raise TypeError("Dimension scores must be provided as a mapping.")

    weights = SETTINGS.rubric_weights
    required = set(weights)
    provided = set(dimension_scores)

    missing = required - provided
    unexpected = provided - required

    if missing:
        raise ValueError(
            f"Missing required dimension scores: {sorted(missing)}"
        )

    if unexpected:
        raise ValueError(
            f"Unexpected dimension scores: {sorted(unexpected)}"
        )

    validated_scores: dict[str, float] = {}

    for name in weights:
        try:
            value = float(dimension_scores[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Score for '{name}' must be numeric."
            ) from exc

        if not math.isfinite(value):
            raise ValueError(
                f"Score for '{name}' must be a finite number."
            )

        if not 0 <= value <= 100:
            raise ValueError(
                f"Score for '{name}' must be between 0 and 100."
            )

        validated_scores[name] = value

    # The weights total 100, so dividing each weight by 100
    # produces the correct weighted mean.
    weighted_score = sum(
        validated_scores[name] * weights[name] / 100.0
        for name in weights
    )

    final_score = round(weighted_score, 1)

    clean_limitations = list(limitations or [])

    return {
        "final_score": final_score,
        "class_label": label_for_score(final_score),
        "dimension_scores": validated_scores,
        "limitations": clean_limitations,
        "method": (
            "Deterministic weighted rubric using configured dimension "
            "weights and grade thresholds. Thresholds are provisional "
            "and require calibration against human-reviewed recordings."
        ),
    }


def local_inspection_summary(
    transcript,
    audio,
    metadata,
) -> dict[str, Any]:
    """Report available local measurements without grading teaching quality."""
    segments = list(transcript or [])

    duration = max(
        (float(segment.end) for segment in segments),
        default=0.0,
    )

    return {
        "transcript_segments": len(segments),
        "transcript_duration_seconds": round(duration, 1),
        "audio_measurements": audio,
        "metadata": metadata,
        "limitations": [
            (
                "Local-inspection mode does not produce an "
                "instructional-quality rating."
            ),
            (
                "No external transcription or model assessment "
                "was performed."
            ),
        ],
    }
