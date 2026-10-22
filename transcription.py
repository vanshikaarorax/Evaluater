from __future__ import annotations
import json, re, urllib.request
from pathlib import Path
from config import SETTINGS
from schemas import TranscriptSegment

_TIMING = re.compile(r"(?P<start>\d{2}:\d{2}:\d{2}[.,]\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2}[.,]\d{3})")


def timestamp_to_seconds(value: str) -> float:
    hours, minutes, seconds = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_vtt(text: str) -> list[TranscriptSegment]:
    lines = text.replace("\ufeff", "").splitlines(); output = []; index = 0
    while index < len(lines):
        match = _TIMING.search(lines[index])
        if not match:
            index += 1; continue
        index += 1; content = []
        while index < len(lines) and lines[index].strip():
            content.append(lines[index].strip()); index += 1
        raw = " ".join(content).strip()
        speaker, sep, words = raw.partition(":")
        output.append(TranscriptSegment(timestamp_to_seconds(match["start"]), timestamp_to_seconds(match["end"]), speaker.strip() if sep else None, words.strip() if sep else raw))
    if not output:
        raise ValueError("No timestamped VTT cues found.")
    return output


def paired_vtt(recording: Path) -> Path | None:
    candidates = list(recording.parent.glob("*.vtt"))
    stem_tokens = set(re.findall(r"[a-z0-9]+", recording.stem.lower()))
    return max(candidates, key=lambda p: len(stem_tokens & set(re.findall(r"[a-z0-9]+", p.stem.lower()))), default=None)


def load_paired_transcript(recording: Path) -> list[TranscriptSegment]:
    candidate = paired_vtt(recording)
    if not candidate: return []
    try: return parse_vtt(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError): return []


def transcribe_deepgram(audio_path: Path) -> list[TranscriptSegment]:
    """Approved-only upload. Caller must pass extracted audio, never the original video."""
    if not SETTINGS.allow_external_processing:
        raise PermissionError("External processing is disabled (ALLOW_EXTERNAL_PROCESSING=false).")
    import os
    key = os.getenv("DEEPGRAM_API_KEY")
    if not key: raise RuntimeError("DEEPGRAM_API_KEY is not configured.")
    query = "model=nova-3&smart_format=true&punctuate=true&utterances=true&diarize_model=latest"
    request = urllib.request.Request(f"https://api.deepgram.com/v1/listen?{query}", data=audio_path.read_bytes(), headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=600) as response: data = json.loads(response.read())
    except Exception as exc: raise RuntimeError("Deepgram transcription failed. Check permission, credits, file length, and network.") from exc
    utterances = data.get("results", {}).get("utterances", [])
    return [TranscriptSegment(float(u["start"]), float(u["end"]), str(u.get("speaker")) if "speaker" in u else None, u.get("transcript", "")) for u in utterances if u.get("transcript")]
