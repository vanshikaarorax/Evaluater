from __future__ import annotations
import array, math, wave
from pathlib import Path


def analyse_wav(path: Path, silence_db: float = -45.0) -> dict:
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2: raise ValueError("Expected mono 16-bit WAV.")
        rate, frames = audio.getframerate(), audio.getnframes(); samples = array.array("h", audio.readframes(frames))
    if not samples: return {"duration_seconds": 0, "limitations": ["No audio samples."]}
    chunk = max(rate, 1); levels = []
    for start in range(0, len(samples), chunk):
        block = samples[start:start + chunk]
        rms = math.sqrt(sum(s * s for s in block) / len(block)) if block else 0
        levels.append(20 * math.log10(max(rms / 32768, 1e-8)))
    silent = [i for i, db in enumerate(levels) if db < silence_db]
    clips = sum(1 for s in samples if abs(s) >= 32700) / len(samples)
    return {"duration_seconds": round(len(samples) / rate, 2), "silent_seconds_estimate": len(silent), "possible_long_silence_windows": [i for i in silent if i and i - 1 in silent], "clipping_sample_ratio": round(clips, 6), "mean_rms_dbfs": round(sum(levels) / len(levels), 2), "note": "Signal measurements are review candidates, not a teaching-quality score."}
