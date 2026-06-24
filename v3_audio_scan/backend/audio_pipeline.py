"""Audio extraction + Silero VAD chunking for the v3 audio-first pipeline.

The venv lives under a path containing non-ASCII characters, which breaks
``torch.jit.load`` (its C-level fopen cannot open the bundled model). We work
around this by copying the bundled ``silero_vad.jit`` to an ASCII temp path and
loading it from there. No torchaudio dependency: WAV reading uses the stdlib
``wave`` module plus numpy.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from config import SETTINGS

logger = logging.getLogger("v3_audio_pipeline")

_VAD_MODEL = None
_VAD_LOAD_LOCK = __import__("threading").Lock()
_ASCII_MODEL_PATH: Path | None = None


@dataclass
class VadSegment:
    """A speech segment inside a chunk, in chunk-relative seconds."""
    start: float
    end: float


def get_video_duration(video_path: str | Path) -> float:
    """Get media duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=60,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def extract_audio(video_path: str | Path, out_wav: str | Path) -> bool:
    """Extract the whole video soundtrack to a 16kHz mono 16-bit PCM WAV.

    One FFmpeg command. Returns True on success.
    """
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                  # drop video
        "-ac", "1",             # mono
        "-ar", "16000",         # 16 kHz
        "-c:a", "pcm_s16le",    # 16-bit PCM
        str(out_wav),
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=1800)
    except Exception as exc:
        logger.exception("ffmpeg audio extraction failed: %s", exc)
        return False
    ok = out_wav.exists() and out_wav.stat().st_size > 44
    if ok:
        logger.info("extracted audio -> %s (%.1f MB)", out_wav, out_wav.stat().st_size / 1e6)
    return ok


def _load_vad():
    """Load Silero VAD model, working around the non-ASCII venv path."""
    global _VAD_MODEL, _ASCII_MODEL_PATH
    if _VAD_MODEL is not None:
        return _VAD_MODEL
    with _VAD_LOAD_LOCK:
        if _VAD_MODEL is not None:
            return _VAD_MODEL
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps  # noqa: F401
        except Exception as exc:
            logger.exception("silero_vad import failed: %s", exc)
            raise

        # Locate the bundled jit model file.
        try:
            from importlib import resources as impresources
            pkg = "silero_vad.data"
            try:
                src = Path(str(impresources.files(pkg).joinpath("silero_vad.jit")))
            except Exception:
                with impresources.path(pkg, "silero_vad.jit") as f:
                    src = Path(f)
        except Exception as exc:
            logger.exception("cannot locate silero_vad.jit: %s", exc)
            raise

        # Copy to an ASCII-only temp path so torch.jit.load's fopen works.
        tmp_dir = Path(tempfile.gettempdir())
        dst = tmp_dir / "codex_silero_vad.jit"
        try:
            shutil.copy2(src, dst)
        except Exception as exc:
            logger.exception("cannot copy silero model to temp: %s", exc)
            raise
        _ASCII_MODEL_PATH = dst
        logger.info("loading silero VAD from %s", dst)
        _VAD_MODEL = torch.jit.load(str(dst), map_location="cpu")
        _VAD_MODEL.eval()
        return _VAD_MODEL


def _read_wav_window(wav_path: str | Path, start_sec: float, end_sec: float):
    """Read [start_sec, end_sec) of a 16kHz mono 16-bit WAV as a float32 tensor."""
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_total = wf.getnframes()
        start_frame = max(0, int(start_sec * sr))
        end_frame = min(n_total, int(end_sec * sr))
        wf.setpos(start_frame)
        raw = wf.readframes(end_frame - start_frame)
    if not raw:
        return torch.zeros(1, dtype=torch.float32), sr
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(arr), sr


def vad_speech_segments(wav_path: str | Path, start_sec: float, end_sec: float):
    """Return speech segments (chunk-relative seconds) and total speech seconds.

    Runs Silero VAD over the [start_sec, end_sec] window of the big WAV.
    """
    model = _load_vad()
    audio, sr = _read_wav_window(wav_path, start_sec, end_sec)
    if audio.numel() < 512:
        return [], 0.0
    from silero_vad import get_speech_timestamps
    ts = get_speech_timestamps(
        audio, model,
        return_seconds=True,
        sampling_rate=sr,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300,
        speech_pad_ms=200,
    )
    segs = [VadSegment(start=float(s["start"]), end=float(s["end"])) for s in ts]
    total = sum(s.end - s.start for s in segs)
    return segs, total


def iter_chunks(total_duration: float, chunk_seconds: float):
    """Yield (chunk_index, start_sec, end_sec) for fixed-size chunks."""
    if total_duration <= 0:
        return
    idx = 0
    start = 0.0
    while start < total_duration:
        end = min(start + chunk_seconds, total_duration)
        yield idx, start, end
        start = end
        idx += 1


def slice_audio_chunk(big_wav: str | Path, start_sec: float, end_sec: float,
                      out_path: str | Path) -> bool:
    """Cut [start_sec, end_sec] out of the big WAV into a 16kHz mono PCM WAV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{end_sec - start_sec:.3f}",
        "-i", str(big_wav),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=300)
    except Exception as exc:
        logger.exception("audio chunk slice failed: %s", exc)
        return False
    return out_path.exists() and out_path.stat().st_size > 44
