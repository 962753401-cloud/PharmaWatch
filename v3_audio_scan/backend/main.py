"""FastAPI entrypoint for v3 audio-first scan system."""
from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from config import SETTINGS
from scanner import AudioScanTask

_CHUNK = 4 * 1024 * 1024  # 4MB streaming chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("v3_audio_scan")

app = FastAPI(title="v3 Audio Scan", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_tasks: dict[str, AudioScanTask] = {}
_tasks_lock = threading.Lock()

SETTINGS.clips_dir.mkdir(parents=True, exist_ok=True)
SETTINGS.uploads_dir.mkdir(parents=True, exist_ok=True)
SETTINGS.work_dir.mkdir(parents=True, exist_ok=True)


class AnalyzePathRequest(BaseModel):
    video_path: str


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "has_api_key": SETTINGS.has_api_key,
       "asr_model": SETTINGS.asr_model,
       "llm_model": SETTINGS.llm_model,
       "demo_mode": SETTINGS.demo_mode,
       "chunk_seconds": SETTINGS.chunk_seconds,
       "slice_half": SETTINGS.slice_half,
       "config": {
            "chunk_seconds": SETTINGS.chunk_seconds,
            "vad_min_speech": SETTINGS.vad_min_speech,
            "slice_half": SETTINGS.slice_half,
            "slice_max": SETTINGS.slice_max,
            "max_upload_mb": SETTINGS.max_upload_mb,
        },
    }


@app.post("/api/analyze")
async def analyze_upload(
    video: UploadFile = File(..., description="video file (mp4/mov/avi)"),
):
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    task_id = uuid.uuid4().hex[:12]
    save_path = SETTINGS.uploads_dir / f"{task_id}{suffix}"
    max_bytes = SETTINGS.max_upload_mb * 1024 * 1024

    try:
        written = 0
        with open(save_path, "wb") as out:
            while True:
                chunk = await video.read(_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    out.close()
                    save_path.unlink(missing_ok=True)
                    return JSONResponse(
                        status_code=413,
                        content={
                            "ok": False,
                            "error": "file_too_large",
                            "message": f"video exceeds limit {SETTINGS.max_upload_mb}MB",
                        },
                    )
                out.write(chunk)

        if written == 0:
            save_path.unlink(missing_ok=True)
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "empty_file", "message": "uploaded video is empty"},
            )

        size_mb = written / (1024 * 1024)
        logger.info("Task %s: uploaded %s (%.1fMB)", task_id, save_path.name, size_mb)

        _start_task(task_id, str(save_path))
        return {"task_id": task_id, "ok": True}
    except Exception as e:
        save_path.unlink(missing_ok=True)
        logger.exception("Upload failed")
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "upload_failed", "message": str(e)},
        )


@app.post("/api/analyze/path")
async def analyze_path(req: AnalyzePathRequest):
    video_path = Path(req.video_path)
    if not video_path.exists():
        return JSONResponse(
            status_code=404,
            content={
                "ok": False,
                "error": "file_not_found",
                "message": f"video not found: {req.video_path}",
            },
        )

    task_id = uuid.uuid4().hex[:12]
    _start_task(task_id, str(video_path))
    return {"task_id": task_id, "ok": True}


def _start_task(task_id: str, video_path: str):
    task = AudioScanTask(task_id, video_path)
    with _tasks_lock:
        _tasks[task_id] = task
    task.start()


@app.get("/api/task/{task_id}")
async def get_task(task_id: str):
    with _tasks_lock:
        task = _tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="task_not_found")
        return task.snapshot()


@app.get("/api/task/{task_id}/transcript")
async def get_transcript(task_id: str):
    with _tasks_lock:
        task = _tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="task_not_found")
        return task.transcript_snapshot()


@app.get("/api/clips/{filename}")
async def get_clip(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="invalid_filename")

    clip_path = SETTINGS.clips_dir / filename
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="clip_not_found")

    return FileResponse(str(clip_path), media_type="video/mp4")


@app.get("/", include_in_schema=False)
async def root():
    index = SETTINGS.frontend_dir / "index.html"
    if not index.exists():
        raise HTTPException(status_code=500, detail="frontend/index.html missing")
    return FileResponse(str(index), media_type="text/html; charset=utf-8")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return JSONResponse(status_code=204, content=None)
