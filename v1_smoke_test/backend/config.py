"""Runtime configuration loaded from environment / .env for the v1 smoke test."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root and from backend/ if present.
_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _BACKEND_DIR.parent
for _candidate in (_PROJECT_DIR / ".env", _BACKEND_DIR / ".env"):
    if _candidate.exists():
        load_dotenv(_candidate, override=False)


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    fps: float
    max_frames: int
    timeout_seconds: int
    max_video_mb: int
    allowed_mime: tuple[str, ...]

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key.strip())


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_settings() -> Settings:
    return Settings(
        api_key=os.getenv("DASHSCOPE_API_KEY", "").strip(),
        base_url=os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).strip(),
        model=os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus").strip(),
        fps=_get_float("QWEN_VL_FPS", 2.0),
        max_frames=_get_int("QWEN_VL_MAX_FRAMES", 80),
        timeout_seconds=_get_int("QWEN_VL_TIMEOUT", 120),
        max_video_mb=_get_int("MAX_VIDEO_MB", 50),
        allowed_mime=(
            "video/mp4",
            "video/quicktime",
            "video/x-msvideo",
        ),
    )


SETTINGS = load_settings()
