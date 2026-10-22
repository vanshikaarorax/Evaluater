"""Dataset discovery only; labels are reference metadata, never model features."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from media_processing import SUPPORTED_SUFFIXES
from transcription import paired_vtt


@dataclass(frozen=True)
class RecordingReference:
    recording: Path
    transcript: Path | None
    folder_label: str | None
    ordinal_label: str | None
    mapping_certain: bool


def discover_recordings(root: Path) -> list[RecordingReference]:
    items = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES: continue
        folder = path.parent.name.strip().lower()
        labels = {"excellent": "Excellent", "good": "Good", "average": "Average", "weak": "Poor"}
        # Weak -> Poor is deliberately marked proposed rather than certain.
        ordinal = labels.get(folder)
        items.append(RecordingReference(path, paired_vtt(path), folder or None, ordinal, folder in {"excellent", "good", "average"}))
    return items


def uncertain_mapping(reference: RecordingReference) -> bool:
    return reference.folder_label in {"weak", "quality issue", "demo"} or reference.transcript is None
