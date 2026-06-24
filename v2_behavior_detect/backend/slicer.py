"""FFmpeg-based video clip slicer."""
from __future__ import annotations

import subprocess
from pathlib import Path

from config import SETTINGS


def get_video_duration(video_path: str | Path) -> float:
    """Get video duration in seconds using ffprobe."""
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


def slice_clip(
    video_path: str | Path,
    trigger_timestamp: float,
    trigger_type: str,
    duration: float | None = None,
) -> dict | None:
    """Slice a clip centered on trigger_timestamp.

    - ±clip_half seconds around the trigger, clamped to [0, duration]
    - total length capped at clip_max
    - saved to clips/ as MP4 via ffmpeg stream copy
    """
    cfg = SETTINGS

    if duration is None:
        duration = get_video_duration(video_path)

    start = max(0.0, trigger_timestamp - cfg.clip_half)
    end = min(duration, trigger_timestamp + cfg.clip_half)

    if end - start > cfg.clip_max:
        end = start + cfg.clip_max

    if end - start < 2.0:
        return None

    hh = int(trigger_timestamp // 3600)
    mm = int((trigger_timestamp % 3600) // 60)
    ss = int(trigger_timestamp % 60)
    filename = f"clip_{hh:02d}{mm:02d}{ss:02d}_{trigger_type}.mp4"

    cfg.clips_dir.mkdir(parents=True, exist_ok=True)
    output_path = cfg.clips_dir / filename

    # Re-encode for frame-accurate cuts and strict duration capping (<=clip_max).
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{end - start:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, timeout=180)
    except Exception:
        pass

    if not output_path.exists() or output_path.stat().st_size < 1000:
        return None

    return {
        "filename": filename,
        "trigger_type": trigger_type,
        "timestamp": round(trigger_timestamp, 1),
        "start": round(start, 1),
        "end": round(end, 1),
        "duration": round(end - start, 1),
        "download_url": f"/api/clips/{filename}",
    }
