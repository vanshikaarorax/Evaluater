
"""Central configuration for ClassLens AI.

Supports local environment variables, .env files, and Streamlit Cloud Secrets.
Keeps rubric weights and grade thresholds consistent across the application.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _read_setting(name: str, default: Any = None) -> Any:
    """Read a setting from environment variables or Streamlit Secrets.

    Environment variables take precedence, allowing local overrides.
    """
    value = os.getenv(name)

    if value is not None:
        return value

    try:
        import streamlit as st

        value = st.secrets.get(name, default)
        return value
    except Exception:
        # Streamlit Secrets may be unavailable during local tests.
        return default


def _truthy(value: Any) -> bool:
    """Interpret a boolean setting safely."""
    if isinstance(value, bool):
        return value

    return str(value or "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _positive_int(name: str, default: int) -> int:
    """Read and validate a positive integer setting."""
    value = _read_setting(name, default)

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc

    if parsed <= 0:
        raise ValueError(f"{name} must be positive.")

    return parsed


@dataclass(frozen=True)
class Settings:
    """Application settings and the initial assessment rubric."""

    allow_external_processing: bool = _truthy(
        _read_setting("ALLOW_EXTERNAL_PROCESSING", False)
    )

    openai_model: str = str(
        _read_setting("OPENAI_MODEL", "gpt-4o-mini")
    ).strip()

    max_upload_mb: int = _positive_int("MAX_UPLOAD_MB", 2048)

    frame_count: int = _positive_int("FRAME_COUNT", 12)

    weak_label_mapping: str = str(
        _read_setting("WEAK_LABEL_MAPPING", "Poor")
    ).strip()

    # These weights determine the overall score, not individual AI scores.
    rubric_weights: dict[str, int] = field(
        default_factory=lambda: {
            "structure_clarity": 25,
            "explanations_examples": 25,
            "checks_feedback": 25,
            "inclusive_interaction": 15,
            "technical_delivery": 10,
        }
    )

    # Provisional thresholds; calibrate against human-reviewed recordings.
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
            "Thresholds must end with (0, 'Poor') "
            "so every score has a label."
        )

    minimums = [minimum for minimum, _ in thresholds]

    if any(
        not isinstance(minimum, int) or isinstance(minimum, bool)
        for minimum in minimums
    ):
        raise ValueError("Grade thresholds must be integers.")

    if any(not 0 <= minimum <= 100 for minimum in minimums):
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

    if not settings.openai_model:
        raise ValueError("OPENAI_MODEL cannot be empty.")

    if not settings.weak_label_mapping:
        raise ValueError("WEAK_LABEL_MAPPING cannot be empty.")


validate_settings()
