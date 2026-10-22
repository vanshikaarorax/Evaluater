from __future__ import annotations
import json, shutil, subprocess
from pathlib import Path

SUPPORTED_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} is required but was not found. Install FFmpeg and retry.")
    return path


def inspect_media(path: Path) -> dict:
    ffprobe = require_binary("ffprobe")
    result = subprocess.run([ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)], capture_output=True, text=True, timeout=30, check=True)
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return {"duration": float(data.get("format", {}).get("duration", 0) or 0), "size_bytes": path.stat().st_size,
            "has_video": video is not None, "has_audio": audio is not None,
            "width": video.get("width") if video else None, "height": video.get("height") if video else None,
            "fps": video.get("r_frame_rate") if video else None, "audio_codec": audio.get("codec_name") if audio else None}


def extract_mono_wav(source: Path, destination: Path) -> Path:
    ffmpeg = require_binary("ffmpeg")
    subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination)], timeout=180, check=True)
    return destination


def validate_upload(path: Path, max_upload_mb: int) -> None:
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("Unsupported format. Upload MP4, MOV, MKV, AVI, or WEBM.")
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError("The uploaded file is empty.")
    if path.stat().st_size > max_upload_mb * 1024 * 1024:
        raise ValueError(f"File exceeds the {max_upload_mb} MB limit.")
