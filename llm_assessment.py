
"""OpenAI-based classroom assessment with grounded evidence validation."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from config import SETTINGS
from prompts import SYSTEM_PROMPT, build_prompt
from schemas import validate_llm_assessment


MAX_TRANSCRIPT_SEGMENTS = 220
MAX_TRANSCRIPT_CHARS = 18000


def _schema() -> dict:
    dimension = {
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
            },
            "explanation": {"type": "string"},
        },
        "required": ["score", "explanation"],
        "additionalProperties": False,
    }

    evidence = {
        "type": "object",
        "properties": {
            "observation": {"type": "string"},
            "modality": {
                "type": "string",
                "enum": ["transcript", "audio", "video"],
            },
            "timestamp": {
                "type": ["number", "null"],
            },
            "excerpt": {
                "type": ["string", "null"],
            },
            "frame_reference": {
                "type": ["string", "null"],
            },
        },
        "required": [
            "observation",
            "modality",
            "timestamp",
            "excerpt",
            "frame_reference",
        ],
        "additionalProperties": False,
    }

    return {
        "type": "object",
        "properties": {
            "class_label": {
                "type": "string",
                "enum": ["Poor", "Average", "Good", "Excellent"],
            },
            "score": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
            },
            "dimension_scores": {
                "type": "object",
                "properties": {
                    name: dimension
                    for name in SETTINGS.rubric_weights
                },
                "required": list(SETTINGS.rubric_weights),
                "additionalProperties": False,
            },
            "strengths": {
                "type": "array",
                "items": {"type": "string"},
            },
            "improvements": {
                "type": "array",
                "items": {"type": "string"},
            },
            "evidence": {
                "type": "array",
                "items": evidence,
            },
            "counterevidence": {
                "type": "array",
                "items": {"type": "string"},
            },
            "technical_flags": {
                "type": "array",
                "items": {"type": "string"},
            },
            "limitations": {
                "type": "array",
                "items": {"type": "string"},
            },
            "confidence": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["level", "rationale"],
                "additionalProperties": False,
            },
        },
        "required": [
            "class_label",
            "score",
            "dimension_scores",
            "strengths",
            "improvements",
            "evidence",
            "counterevidence",
            "technical_flags",
            "limitations",
            "confidence",
        ],
        "additionalProperties": False,
    }


def _select_transcript_segments(transcript) -> list:
    """Select segments across the recording rather than only its beginning."""
    total = len(transcript)

    if total <= MAX_TRANSCRIPT_SEGMENTS:
        return list(transcript)

    # Evenly spaced indices give later sections a chance to be represented.
    indices = [
        round(
            i * (total - 1) / (MAX_TRANSCRIPT_SEGMENTS - 1)
        )
        for i in range(MAX_TRANSCRIPT_SEGMENTS)
    ]

    return [transcript[index] for index in indices]


def _build_transcript_excerpt(transcript) -> str:
    """Format selected timestamped segments within the character budget."""
    selected = _select_transcript_segments(transcript)
    lines = []
    chars_used = 0

    for segment in selected:
        speaker_prefix = (
            f"{segment.speaker}: " if segment.speaker else ""
        )
        line = (
            f"[{segment.start:.1f}-{segment.end:.1f}] "
            f"{speaker_prefix}{segment.text}"
        )

        remaining = MAX_TRANSCRIPT_CHARS - chars_used

        if remaining <= 0:
            break

        if len(line) > remaining:
            # Preserve a bounded excerpt; never exceed the configured limit.
            line = line[:remaining]
            lines.append(line)
            break

        lines.append(line)
        chars_used += len(line) + 1

    return "\n".join(lines)


def assess_with_openai(
    transcript,
    audio: dict,
    frames: list[dict],
    duration: float | None,
) -> dict:
    """Generate and validate an assessment using approved external processing."""
    if not SETTINGS.allow_external_processing:
        raise PermissionError("External processing is disabled.")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    if not transcript:
        raise ValueError(
            "No transcript is available for assessment."
        )

    from openai import OpenAI

    excerpt = _build_transcript_excerpt(transcript)

    frame_refs = [
        {
            "reference": frame["reference"],
            "timestamp": frame["timestamp"],
        }
        for frame in frames
    ]

    prompt = build_prompt(
        SETTINGS.rubric_weights,
        excerpt,
        audio,
        frame_refs,
        SETTINGS.thresholds,
    )

    content = [
        {
            "type": "input_text",
            "text": prompt,
        }
    ]

    for frame in frames:
        encoded = base64.b64encode(
            Path(frame["path"]).read_bytes()
        ).decode("ascii")

        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{encoded}",
                "detail": "low",
            }
        )

    client = OpenAI()

    response = client.responses.create(
        model=SETTINGS.openai_model,
        store=False,
        instructions=SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "classlens_assessment",
                "strict": True,
                "schema": _schema(),
            }
        },
    )

    try:
        payload = json.loads(response.output_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Model response was not valid assessment JSON."
        ) from exc

    return validate_llm_assessment(
        payload,
        duration,
        transcript,
        {frame["reference"] for frame in frames},
    )
