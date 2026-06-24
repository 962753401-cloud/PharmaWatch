"""FastAPI entrypoint for the v1 smoke test (qwen3-vl-plus video Q&A)."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from config import SETTINGS
from qwen_client import QwenVLClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("v1_smoke_test")

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"

app = FastAPI(title="药店监控 AI v1 烟雾测试", version="0.1.0")

# Same-origin in production; keep CORS open here so localhost dev tools work.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

qwen_client = QwenVLClient(SETTINGS)

ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi"}


def _normalize_mime(upload: UploadFile) -> str:
    declared = (upload.content_type or "").lower()
    if declared in SETTINGS.allowed_mime:
        return declared
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix == ".mp4":
        return "video/mp4"
    if suffix in {".mov", ".m4v"}:
        return "video/quicktime"
    if suffix == ".avi":
        return "video/x-msvideo"
    return declared


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "has_api_key": SETTINGS.has_api_key,
        "max_video_mb": SETTINGS.max_video_mb,
        "fps": SETTINGS.fps,
    }


@app.post("/api/analyze")
async def analyze(
    video: UploadFile = File(..., description="药店监控视频文件 (mp4/mov/avi)"),
    question: str | None = Form(default=None),
):
    if not SETTINGS.has_api_key:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error_code": "api_key_missing",
                "message": "后端未配置 DASHSCOPE_API_KEY，请在 .env 中填写后重启服务。",
            },
        )

    declared_mime = (video.content_type or "").lower()
    suffix = Path(video.filename or "").suffix.lower()
    if declared_mime and declared_mime not in SETTINGS.allowed_mime and suffix not in ALLOWED_SUFFIXES:
        return JSONResponse(
            status_code=415,
            content={
                "ok": False,
                "error_code": "unsupported_media",
                "message": f"仅支持 mp4/mov/avi 视频，当前类型：{declared_mime or suffix or 'unknown'}",
            },
        )

    data = await video.read()
    size_mb = len(data) / (1024 * 1024)
    if len(data) == 0:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error_code": "empty_file", "message": "上传的视频为空。"},
        )
    if size_mb > SETTINGS.max_video_mb:
        return JSONResponse(
            status_code=413,
            content={
                "ok": False,
                "error_code": "file_too_large",
                "message": f"视频体积 {size_mb:.1f}MB 超过上限 {SETTINGS.max_video_mb}MB。",
            },
        )

    mime = _normalize_mime(video) or "video/mp4"
    q = (question or "").strip()
    logger.info(
        "analyze start filename=%s size=%.2fMB mime=%s question_len=%d",
        video.filename,
        size_mb,
        mime,
        len(q),
    )

    result = qwen_client.analyze_video(data, mime, q)
    payload = result.to_dict()
    logger.info(
        "analyze done ok=%s elapsed_ms=%s tokens=%s",
        result.ok,
        result.elapsed_ms,
        payload.get("tokens") if result.ok else None,
    )
    status_code = 200 if result.ok else 502
    if not result.ok and result.error_code == "api_key_missing":
        status_code = 503
    return JSONResponse(status_code=status_code, content=payload)


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=500, detail="frontend/index.html 缺失")
    return FileResponse(INDEX_HTML, media_type="text/html; charset=utf-8")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return JSONResponse(status_code=204, content=None)