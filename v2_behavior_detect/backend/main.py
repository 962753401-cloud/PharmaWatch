"""FastAPI entrypoint for v2 behavior detection system."""
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
from scanner import VideoScanner
from slicer import slice_clip, get_video_duration

_CHUNK = 4 * 1024 * 1024  # 4MB streaming chunks


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("v2_behavior_detect")

app = FastAPI(title="v2 Behavior Detection", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()

SETTINGS.clips_dir.mkdir(parents=True, exist_ok=True)
SETTINGS.uploads_dir.mkdir(parents=True, exist_ok=True)
SETTINGS.frames_dir.mkdir(parents=True, exist_ok=True)


class AnalyzePathRequest(BaseModel):
    video_path: str


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "weights_exists": SETTINGS.weights_path.exists(),
        "classes": list(SETTINGS.classes),
        "triggers": {
            "crowd": {
                "threshold": SETTINGS.crowd_threshold,
                "consecutive": SETTINGS.crowd_consecutive,
            },
            "loiter": {
                "seconds": SETTINGS.loiter_seconds,
                "radius_px": SETTINGS.loiter_radius,
            },
            "carry": {
                "target_classes": list(SETTINGS.carry_target_classes),
                "sustain_frames": SETTINGS.carry_sustain_frames,
                "expand_px": SETTINGS.carry_expand,
            },
        },
        "slice": {
            "half_seconds": SETTINGS.clip_half,
            "max_seconds": SETTINGS.clip_max,
            "cooldown_seconds": SETTINGS.cooldown,
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
    with _tasks_lock:
        _tasks[task_id] = {
            "status": "scanning",
            "progress": 0.0,
            "current_time": 0.0,
            "total_time": 0.0,
            "triggers_found": 0,
            "clips": [],
            "events": [],
            "error": None,
            "video_path": video_path,
            "latest_frame": 0,
            "frame_dir": str(SETTINGS.frames_dir / task_id),
        }

    thread = threading.Thread(target=_run_scan, args=(task_id, video_path), daemon=True)
    thread.start()


def _run_scan(task_id: str, video_path: str):
    logger.info("Task %s: starting scan on %s", task_id, video_path)

    try:
        duration = get_video_duration(video_path)
    except Exception as e:
        with _tasks_lock:
            t = _tasks.get(task_id)
            if t:
                t["status"] = "error"
                t["error"] = f"cannot read video duration: {e}"
        return

    with _tasks_lock:
        t = _tasks.get(task_id)
        if t:
            t["total_time"] = round(duration, 1)

    def on_progress(frame_idx, total_frames, current_time, total_time):
        with _tasks_lock:
            t = _tasks.get(task_id)
            if t:
                t["progress"] = round(frame_idx / max(total_frames, 1) * 100, 1)
                t["current_time"] = round(current_time, 1)
                t["total_time"] = round(total_time, 1)

    def on_trigger(evt):
        logger.info(
            "Task %s: trigger %s at %.1fs - %s",
            task_id, evt.trigger_type, evt.timestamp, evt.details,
        )
        with _tasks_lock:
            t = _tasks.get(task_id)
            if t:
                t["events"].append({
                    "timestamp": round(evt.timestamp, 1),
                    "trigger_type": evt.trigger_type,
                    "details": evt.details,
                })
        def _do_slice():
            try:
                clip_info = slice_clip(video_path, evt.timestamp, evt.trigger_type, duration)
                if clip_info:
                    with _tasks_lock:
                        t = _tasks.get(task_id)
                        if t:
                            t["clips"].append(clip_info)
                            t["triggers_found"] += 1
            except Exception as e:
                logger.exception("Task %s: slice failed", task_id)
        threading.Thread(target=_do_slice, daemon=True).start()

    # on_frame callback: save annotated JPEG for live preview
    frame_dir = SETTINGS.frames_dir / task_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    def on_frame(fa):
        # Save JPEG to disk
        jpeg_path = frame_dir / f"{fa.frame_idx:06d}.jpg"
        try:
            jpeg_path.write_bytes(fa.jpeg_bytes)
        except Exception:
            pass
        # Update latest_frame index
        with _tasks_lock:
            t = _tasks.get(task_id)
            if t:
                t["latest_frame"] = fa.frame_idx

    scanner = VideoScanner(video_path, on_progress=on_progress, on_trigger=on_trigger, on_frame=on_frame)

    try:
        scanner.scan()
        with _tasks_lock:
            t = _tasks.get(task_id)
            if t and t["status"] != "error":
                t["status"] = "done"
                t["progress"] = 100.0
        logger.info("Task %s: scan complete", task_id)
    except Exception as e:
        logger.exception("Task %s: scan failed", task_id)
        with _tasks_lock:
            t = _tasks.get(task_id)
            if t:
                t["status"] = "error"
                t["error"] = str(e)


@app.get("/api/task/{task_id}")
async def get_task(task_id: str):
    with _tasks_lock:
        t = _tasks.get(task_id)
        if not t:
            raise HTTPException(status_code=404, detail="task_not_found")
        return {
            "status": t["status"],
            "progress": t["progress"],
            "current_time": t["current_time"],
            "total_time": t["total_time"],
            "triggers_found": t["triggers_found"],
            "clips": list(t["clips"]),
            "events": list(t.get("events", [])),
            "latest_frame": t.get("latest_frame", 0),
            "error": t["error"],
        }


@app.get("/api/task/{task_id}/frame/{frame_idx}")
async def get_frame(task_id: str, frame_idx: str):
    with _tasks_lock:
        t = _tasks.get(task_id)
        if not t:
            raise HTTPException(status_code=404, detail="task_not_found")
        frame_dir = SETTINGS.frames_dir / task_id

    if frame_idx == "latest":
        with _tasks_lock:
            t = _tasks.get(task_id)
            if t:
                frame_idx = str(t.get("latest_frame", 0))
            else:
                raise HTTPException(status_code=404, detail="task_not_found")

    jpeg_path = frame_dir / f"{int(frame_idx):06d}.jpg"
    if not jpeg_path.exists():
        raise HTTPException(status_code=404, detail="frame_not_found")

    return FileResponse(str(jpeg_path), media_type="image/jpeg")


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
