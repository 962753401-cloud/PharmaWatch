"""FFmpeg-based video clip slicer with boundary clamping."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from config import SETTINGS

logger = logging.getLogger("v3_slicer")


def get_video_duration(video_path):
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def slice_video_clip(
    video_path,
    abs_ts,
    task_id,
    duration=None,
    keyword="",
    violation_type="",
    transcript="",
    verdict="",
    reason="",
):
    """Cut a ~2-minute video clip centered on abs_ts with boundary clamping.

    Start = max(0, abs_ts - SLICE_HALF)
    End   = min(duration, abs_ts + SLICE_HALF)
    """
    if duration is None or duration <= 0:
        duration = get_video_duration(video_path)
    if duration <= 0:
        return None

    cfg = SETTINGS
    start = max(0.0, abs_ts - cfg.slice_half)
    end = min(duration, abs_ts + cfg.slice_half)
    if end - start > cfg.slice_max:
        end = start + cfg.slice_max
    if end - start < 2.0:
        return None

    filename = f"clip_{task_id}_{abs_ts:.1f}.mp4"
    cfg.clips_dir.mkdir(parents=True, exist_ok=True)
    output = cfg.clips_dir / filename

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{end - start:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-avoid_negative_ts", "make_zero",
        str(output),
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=180)
    except Exception as e:
        logger.warning("slice failed: %s", e)

    if not output.exists() or output.stat().st_size < 1000:
        return None

    return {
        "filename": filename,
        "timestamp": round(abs_ts, 1),
        "start": round(start, 1),
        "end": round(end, 1),
        "duration": round(end - start, 1),
        "download_url": f"/api/clips/{filename}",
        "keyword": keyword,
        "violation_type": violation_type,
        "transcript": transcript,
        "verdict": verdict,
        "reason": reason,
    }
