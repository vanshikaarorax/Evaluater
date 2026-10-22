from __future__ import annotations
from pathlib import Path


def sample_frames(source: Path, output_dir: Path, count: int = 12) -> list[dict]:
    try: import cv2
    except ImportError as exc: raise RuntimeError("OpenCV is required for frame sampling.") from exc
    capture = cv2.VideoCapture(str(source)); fps = capture.get(cv2.CAP_PROP_FPS) or 0; total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if not fps or not total: capture.release(); raise ValueError("Video stream unavailable for sampling.")
    output_dir.mkdir(parents=True, exist_ok=True); results = []
    for index, frame_no in enumerate(sorted({int(total * i / (count + 1)) for i in range(1, count + 1)})):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no); ok, frame = capture.read()
        if not ok: continue
        h, w = frame.shape[:2]; scale = min(1.0, 960 / max(w, h)); frame = cv2.resize(frame, (round(w * scale), round(h * scale)))
        filename = f"frame_{index:02d}_{frame_no / fps:.1f}s.jpg"; target = output_dir / filename
        cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        results.append({"reference": filename, "timestamp": round(frame_no / fps, 2), "path": str(target)})
    capture.release(); return results
