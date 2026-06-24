"""Runtime configuration for v2 behavior detection system."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _BACKEND_DIR.parent
_REPO_DIR = _PROJECT_DIR.parent  # v2 weights live at project root /weights

def _find_weights():
    env = os.getenv('WEIGHTS_PATH')
    if env:
        return Path(env)
    for cand in [_REPO_DIR / 'weights' / 'yolov8s-world.pt', _PROJECT_DIR / 'weights' / 'yolov8s-world.pt', _BACKEND_DIR / 'weights' / 'yolov8s-world.pt']:
        if cand.exists():
            return cand
    return _PROJECT_DIR / 'weights' / 'yolov8s-world.pt'


@dataclass(frozen=True)
class Settings:
    weights_path: Path
    clips_dir: Path
    uploads_dir: Path
    frontend_dir: Path

    # Detection classes (YOLO-World open vocabulary)
    classes: tuple[str, ...]
    object_classes: tuple[str, ...]  # classes for trigger 2 & 3
    chair_classes: tuple[str, ...]   # classes for trigger 3 (seating)
    conf_person: float
    conf_object: float
    sample_interval: int

    # Trigger 1: Crowd
    crowd_threshold: int
    crowd_consecutive: int

    # Trigger 2: Key items
    carry_target_classes: tuple[str, ...]
    carry_sustain_frames: int
    carry_expand: int

    # Trigger 3: Sit + stay
    loiter_seconds: float
    loiter_radius: float
    lost_track_buffer: int
    frame_rate: float

    # Slicing
    clip_half: float
    clip_max: float
    cooldown: float

    # Annotated frames (live preview)
    frames_dir: Path
    annotated_jpeg_quality: int
    annotated_max_width: int

    # Server
    max_upload_mb: int


def _env_float(name, default):
    raw = os.getenv(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_int(name, default):
    raw = os.getenv(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def load_settings():
    return Settings(
        weights_path=Path(
            os.getenv("WEIGHTS_PATH", str(_PROJECT_DIR / "weights" / "yolov8s-world.pt"))
        ),
        clips_dir=Path(os.getenv("CLIPS_DIR", str(_BACKEND_DIR / "clips"))),
        uploads_dir=Path(os.getenv("UPLOADS_DIR", str(_BACKEND_DIR / "uploads"))),
        frontend_dir=Path(os.getenv("FRONTEND_DIR", str(_PROJECT_DIR / "frontend"))),
        frames_dir=Path(os.getenv("FRAMES_DIR", str(_BACKEND_DIR / "frames"))),
        classes=(
            "person", "hand", "card", "box", "bottle",
            "bag", "backpack", "handbag", "phone",
           "chair", "bench", "seat", "hat",
        ),
        object_classes=(
            "box", "bottle", "card", "bag", "backpack", "handbag", "phone", "hand",
        ),
        chair_classes=("chair", "bench", "seat"),
        conf_person=_env_float("CONF_PERSON", 0.25),
        conf_object=_env_float("CONF_OBJECT", 0.15),
        sample_interval=_env_int("SAMPLE_INTERVAL", 15),
        crowd_threshold=_env_int("CROWD_THRESHOLD", 12),
        crowd_consecutive=_env_int("CROWD_CONSECUTIVE", 3),
        carry_target_classes=("phone", "backpack", "hat"),
        carry_sustain_frames=_env_int("CARRY_SUSTAIN_FRAMES", 2),
        carry_expand=_env_int("CARRY_EXPAND", 50),
        loiter_seconds=_env_float("LOITER_SECONDS", 2.0),
        loiter_radius=_env_float("LOITER_RADIUS", 80),
        lost_track_buffer=_env_int("LOST_TRACK_BUFFER", 60),
        frame_rate=_env_float("FRAME_RATE", 2.0),
        clip_half=_env_float("CLIP_HALF", 5),
        clip_max=_env_float("CLIP_MAX", 10),
        cooldown=_env_float("COOLDOWN", 15),
        annotated_jpeg_quality=_env_int("ANNOTATED_JPEG_QUALITY", 70),
        annotated_max_width=_env_int("ANNOTATED_MAX_WIDTH", 960),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 800),
    )


SETTINGS = load_settings()
