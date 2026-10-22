
"""Central configuration for ClassLens AI.

Keep rubric weights and grade thresholds in one place so the UI,
LLM prompt, and deterministic evaluator use consistent settings.
Secrets must remain in environment variables.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _truthy(value: str | None) -> bool:
    """Interpret a boolean environment variable safely."""
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Application settings and the initial assessment rubric."""

    allow_external_processing: bool = _truthy(
        os.getenv("ALLOW_EXTERNAL_PROCESSING")
    )
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "2048"))
    frame_count: int = int(os.getenv("FRAME_COUNT", "12"))
    weak_label_mapping: str = os.getenv("WEAK_LABEL_MAPPING", "Poor")

    # These weights must total 100. They determine the overall score,
    # not the scores assigned by the AI to individual dimensions.
    rubric_weights: dict[str, int] = field(
        default_factory=lambda: {
            "structure_clarity": 25,
            "explanations_examples": 25,
            "checks_feedback": 25,
            "inclusive_interaction": 15,
            "technical_delivery": 10,
        }
    )

    # Minimum overall score required for each label.
    # These are provisional thresholds and should be calibrated
    # against human-reviewed classroom recordings.
    thresholds: tuple[tuple[int, str], ...] = (
        (85, "Excellent"),
        (70, "Good"),
        (60, "Average"),
        (0, "Poor"),
    )


SETTINGS = Settings()


def validate_settings(settings: Settings = SETTINGS) -> None:
    """Fail early if the rubric configuration is inconsistent."""
    weights = settings.rubric_weights

    required_dimensions = {
        "structure_clarity",
        "explanations_examples",
        "checks_feedback",
        "inclusive_interaction",
        "technical_delivery",
    }

    if set(weights) != required_dimensions:
        raise ValueError(
            "Rubric weights must define exactly these dimensions: "
            f"{sorted(required_dimensions)}"
        )

    if any(
        not isinstance(weight, (int, float))
        or isinstance(weight, bool)
        or not math.isfinite(weight)
        or weight <= 0
        for weight in weights.values()
    ):
        raise ValueError("Every rubric weight must be a positive number.")

    if not math.isclose(sum(weights.values()), 100.0):
        raise ValueError("Rubric weights must total exactly 100.")

    thresholds = settings.thresholds

    if not thresholds or thresholds[-1] != (0, "Poor"):
        raise ValueError(
            "Thresholds must end with (0, 'Poor') so every score has a label."
        )

    minimums = [minimum for minimum, _ in thresholds]

    if any(
        not isinstance(minimum, int) or isinstance(minimum, bool)
        for minimum in minimums
    ):
        raise ValueError("Grade thresholds must be integers.")

    if any(
        not 0 <= minimum <= 100
        for minimum in minimums
    ):
        raise ValueError("Grade thresholds must be between 0 and 100.")

    if any(
        minimums[index] <= minimums[index + 1]
        for index in range(len(minimums) - 1)
    ):
        raise ValueError(
            "Grade thresholds must be ordered from highest to lowest."
        )

    if len({label for _, label in thresholds}) != len(thresholds):
        raise ValueError("Every grade threshold must have a unique label.")

    if settings.max_upload_mb <= 0:
        raise ValueError("MAX_UPLOAD_MB must be positive.")

    if settings.frame_count <= 0:
        raise ValueError("FRAME_COUNT must be positive.")

    if not settings.openai_model.strip():
        raise ValueError("OPENAI_MODEL cannot be empty.")


validate_settings()
