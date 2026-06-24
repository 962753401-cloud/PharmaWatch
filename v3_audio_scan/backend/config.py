"""Runtime configuration for v3 audio-first scan system."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _BACKEND_DIR.parent

# Load .env from project root then backend (project root wins override=False).
for _candidate in (_PROJECT_DIR / ".env", _BACKEND_DIR / ".env"):
    if _candidate.exists():
        load_dotenv(_candidate, override=False)


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


def _env_bool(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    asr_model: str
    asr_ws_url: str
    llm_model: str
    llm_fallback: str

    chunk_seconds: float
    vad_min_speech: float
    vad_threshold: float
    slice_half: float
    slice_max: float

    clips_dir: Path
    uploads_dir: Path
    work_dir: Path
    frontend_dir: Path

    demo_mode: bool
    mock_asr: bool
    max_upload_mb: int
    asr_send_interval: float
    llm_timeout: int

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key.strip())


def load_settings() -> Settings:
    return Settings(
        api_key=os.getenv("DASHSCOPE_API_KEY", "").strip(),
        base_url=os.getenv(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).strip(),
        asr_model=os.getenv("ASR_MODEL", "fun-asr-realtime-2026-02-28").strip(),
        asr_ws_url=os.getenv(
            "ASR_WS_URL", "wss://dashscope.aliyuncs.com/api-ws/v1/inference/"
        ).strip(),
        llm_model=os.getenv("LLM_MODEL", "qwen3.5-flash").strip(),
        llm_fallback=os.getenv("LLM_FALLBACK", "qwen-plus").strip(),
        chunk_seconds=_env_float("CHUNK_SECONDS", 300.0),
        vad_min_speech=_env_float("VAD_MIN_SPEECH", 5.0),
        vad_threshold=_env_float("VAD_THRESHOLD", 0.5),
        slice_half=_env_float("SLICE_HALF", 10.0),
        slice_max=_env_float("SLICE_MAX", 20.0),
        clips_dir=Path(os.getenv("CLIPS_DIR", str(_BACKEND_DIR / "clips"))),
        uploads_dir=Path(os.getenv("UPLOADS_DIR", str(_BACKEND_DIR / "uploads"))),
        work_dir=Path(os.getenv("WORK_DIR", str(_BACKEND_DIR / "work"))),
        frontend_dir=Path(os.getenv("FRONTEND_DIR", str(_PROJECT_DIR / "frontend"))),
        demo_mode=_env_bool("DEMO_MODE", True),
        mock_asr=_env_bool("MOCK_ASR", False),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 800),
        asr_send_interval=_env_float("ASR_SEND_INTERVAL", 0.0),
        llm_timeout=_env_int("LLM_TIMEOUT", 60),
    )


SETTINGS = load_settings()
SETTINGS.clips_dir.mkdir(parents=True, exist_ok=True)
SETTINGS.uploads_dir.mkdir(parents=True, exist_ok=True)
SETTINGS.work_dir.mkdir(parents=True, exist_ok=True)
