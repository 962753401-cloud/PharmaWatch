"""Orchestrator: drives the audio-first scan pipeline for a single task.

Stage flow:
  extract audio (ffmpeg) -> VAD windowing (silero) -> slice 5-min WAV chunks
  -> ASR (fun-asr-realtime) -> keyword match (pypinyin) -> risk LLM (qwen3.5-flash)
  -> boundary-clamped video slice (ffmpeg)

All progress is pushed into a shared task-state dict (thread-safe via a lock).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import SETTINGS
from audio_pipeline import (
    extract_audio,
    get_video_duration,
    iter_chunks,
    slice_audio_chunk,
    vad_speech_segments,
)
from asr_client import transcribe, mock_transcribe
from keyword_matcher import Word, scan_transcript, build_context
from risk_llm import RiskLLMClient
from slicer import slice_video_clip

logger = logging.getLogger("v3_scanner")

STAGE_EXTRACT = "extract"
STAGE_VAD = "vad"
STAGE_ASR = "asr"
STAGE_RISK = "risk"
STAGE_DONE = "done"


class ScanRunner:
    """Runs the full pipeline for one task, writing results into `state`."""

    def __init__(
        self,
        task_id: str,
        video_path: str,
        state: dict,
        state_lock: threading.Lock,
    ) -> None:
        self.task_id = task_id
        self.video_path = video_path
        self.state = state
        self.state_lock = state_lock
        self._cancel = threading.Event()
        self._risk = RiskLLMClient()

    def cancel(self) -> None:
        self._cancel.set()

    # ---- state helpers -------------------------------------------------
    def _set(self, **kw: Any) -> None:
        with self.state_lock:
            for k, v in kw.items():
                self.state[k] = v

    def _add_event(self, evt: dict) -> None:
        with self.state_lock:
            self.state["events"].append(evt)

    def _add_clip(self, clip: dict) -> None:
        with self.state_lock:
            self.state["clips"].append(clip)
            self.state["triggers_found"] = len(self.state["clips"])

    def _push_transcript(self, text: str, chunk_label: str, words_text: str = "") -> None:
        with self.state_lock:
            self.state["transcript"] = text
            self.state["transcript_chunk"] = chunk_label
            self.state["transcript_words"] = words_text or text

    # ---- main ----------------------------------------------------------
    def run(self) -> None:
        try:
            duration = get_video_duration(self.video_path)
            if duration <= 0:
                raise RuntimeError("cannot read video duration (ffprobe returned 0)")
            self._set(total_time=round(duration, 1))

            # Stage 1: extract audio
            self._set(stage=STAGE_EXTRACT, stage_label="提取音轨")
            work = SETTINGS.work_dir / self.task_id
            work.mkdir(parents=True, exist_ok=True)
            big_wav = work / "full_track.wav"
            ok = extract_audio(self.video_path, big_wav)
            if not ok or self._cancel.is_set():
                if not ok:
                    raise RuntimeError("ffmpeg audio extraction produced no output")
                return self._finish_cancelled()

            # Stage 2: VAD windowing + per-chunk processing
            self._set(stage=STAGE_VAD, stage_label="VAD 预筛")
            chunks = list(iter_chunks(duration, SETTINGS.chunk_seconds))
            total_chunks = len(chunks)
            self._set(total_chunks=total_chunks)
            logger.info("Task %s: %d chunks total", self.task_id, total_chunks)

            if total_chunks == 0:
                self._push_transcript("未检测到音频片段", "-")

            # First pass: VAD to count speech chunks for progress baseline.
            speech_chunks: list[tuple[int, float, float]] = []
            for idx, cstart, cend in chunks:
                if self._cancel.is_set():
                    return self._finish_cancelled()
                segs, speech_sec = vad_speech_segments(big_wav, cstart, cend)
                self._set(
                    stage=STAGE_VAD,
                    stage_label=f"VAD 预筛 {idx + 1}/{total_chunks}",
                    current_chunk_index=idx,
                    current_chunk_start=round(cstart, 1),
                    current_chunk_end=round(cend, 1),
                    vad_speech_sec=round(speech_sec, 1),
                    progress=round((idx + 1) / max(total_chunks, 1) * 40, 1),
                )
                if speech_sec >= SETTINGS.vad_min_speech:
                    speech_chunks.append((idx, cstart, cend))

            self._set(total_chunks=total_chunks, speech_chunks=len(speech_chunks))

            # Stage 3 + 4: ASR + risk for each speech chunk.
            n = len(speech_chunks)
            for i, (idx, cstart, cend) in enumerate(speech_chunks):
                if self._cancel.is_set():
                    return self._finish_cancelled()
                self._set(
                    stage=STAGE_ASR,
                    stage_label=f"ASR 转写 {i + 1}/{n}",
                    current_chunk_index=idx,
                    current_chunk_start=round(cstart, 1),
                    current_chunk_end=round(cend, 1),
                    progress=round(40 + (i / max(n, 1)) * 55, 1),
                )
                self._process_window(work, big_wav, cstart, cend, duration)

            self._set(status="done", stage=STAGE_DONE, stage_label="完成", progress=100.0)
        except Exception as e:
            logger.exception("Task %s: scan failed", self.task_id)
            self._set(status="error", error=str(e))

    # ---- per-window ----------------------------------------------------
    def _process_window(
        self, work: Path, big_wav: Path, cstart: float, cend: float, duration: float
    ) -> None:
        chunk_name = f"vid_{self.task_id}_{int(cstart)}_{int(cend)}.wav"
        chunk_path = work / chunk_name
        if not slice_audio_chunk(big_wav, cstart, cend, chunk_path):
            logger.warning("Task %s: audio chunk slice failed for %s", self.task_id, chunk_name)
            return

        chunk_label = f"{int(cstart)}-{int(cend)}s"
        self._push_transcript("（转写中...）", chunk_label)

        if SETTINGS.mock_asr:
            asr = mock_transcribe(chunk_path, cstart)
        else:
            asr = transcribe(
                chunk_path,
                SETTINGS.api_key,
                SETTINGS.asr_model,
                send_interval=SETTINGS.asr_send_interval,
                timeout=300,
            )
        if not asr.ok:
            self._push_transcript(f"（ASR 失败：{asr.error}）", chunk_label)
            logger.warning("Task %s: ASR failed for %s: %s", self.task_id, chunk_name, asr.error)
            return

        # Build word list with absolute timestamps for the whole chunk.
        all_words: list[Word] = []
        full_text_parts: list[str] = []
        sentence_texts: list[tuple[float, float, str]] = []
        for s in asr.sentences:
            s_text = s.text or ""
            s_abs_begin = cstart + (s.begin_time_ms or 0) / 1000.0
            s_abs_end = cstart + (s.end_time_ms or s.begin_time_ms or 0) / 1000.0
            sentence_texts.append((s_abs_begin, s_abs_end, s_text))
            full_text_parts.append(s_text)
            for w in s.words:
                all_words.append(Word(
                    text=w.text,
                    abs_begin=cstart + w.begin_time_ms / 1000.0,
                    abs_end=cstart + w.end_time_ms / 1000.0,
                ))

        full_text = "".join(full_text_parts)
        self._push_transcript(full_text or "（无人声内容）", chunk_label, full_text)

        if not all_words:
            return

        # Stage 1 keyword matching.
        hits = scan_transcript(all_words, demo_mode=SETTINGS.demo_mode)
        if not hits:
            return

        # Stage 2 risk LLM + video slicing per hit, with cooldown dedup.
        # Hits within CLIP_COOLDOWN seconds of the last sliced clip are merged
        # into the existing clip (only the first hit in a cooldown window slices).
        clip_cooldown = 10.0  # seconds - merge hits within this window
        last_clip_ts = -999.0
        for hit in hits:
            if self._cancel.is_set():
                return
            self._set(stage=STAGE_RISK, stage_label="风控判定")
            context = self._build_context(all_words, hit.abs_timestamp)
            rr = self._risk.judge(context, keyword=hit.keyword)
            verdict = rr.verdict if rr.ok else "合规"
            reason = rr.reason if rr.ok else (rr.message or "风控判定失败，默认合规")
            transcript_snippet = self._snippet_around(sentence_texts, hit.abs_timestamp)
            self._add_event({
                "timestamp": round(hit.abs_timestamp, 1),
                "keyword": hit.keyword,
                "violation_type": hit.violation_type,
                "match_mode": hit.match_mode,
                "source": hit.source,
                "verdict": verdict,
                "reason": reason,
                "stage2_ok": rr.ok,
                "chunk": chunk_label,
            })
            # Cooldown dedup: skip slicing if within cooldown of last clip.
            if hit.abs_timestamp - last_clip_ts < clip_cooldown:
                continue
            clip = slice_video_clip(
                self.video_path,
                hit.abs_timestamp,
                self.task_id,
                duration,
                keyword=hit.keyword,
                violation_type=hit.violation_type,
                transcript=transcript_snippet,
                verdict=verdict,
                reason=reason,
            )
            if clip:
                self._add_clip(clip)
                last_clip_ts = hit.abs_timestamp

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _snippet_around(
        sentences: list[tuple[float, float, str]], abs_ts: float
    ) -> str:
        parts = []
        for abs_begin, abs_end, text in sentences:
            if abs_end >= abs_ts - 30 and abs_begin <= abs_ts + 30:
                parts.append(f"[{abs_begin:.1f}s] {text}")
        return " ".join(parts) if parts else ""

    @staticmethod
    def _build_context(words: list[Word], abs_ts: float, span: int = 5) -> str:
        idx = 0
        for i, w in enumerate(words):
            if w.abs_begin <= abs_ts <= w.abs_end:
                idx = i
                break
            idx = i
        return build_context(words, idx, span) or "（无上下文）"

    def _finish_cancelled(self) -> None:
        self._set(status="error", error="任务已取消")


def _initial_state() -> dict:
    return {
        "status": "pending",
        "stage": "",
        "stage_label": "",
        "progress": 0.0,
        "current_time": 0.0,
        "total_time": 0.0,
        "total_chunks": 0,
        "speech_chunks": 0,
        "current_chunk_index": 0,
        "current_chunk_start": 0.0,
        "current_chunk_end": 0.0,
        "vad_speech_sec": 0.0,
        "triggers_found": 0,
        "clips": [],
        "events": [],
        "transcript": "",
        "transcript_chunk": "",
        "transcript_words": "",
        "error": None,
        "video_path": "",
    }


class AudioScanTask:
    """Owns task state + worker thread; exposes snapshot() for FastAPI."""

    def __init__(self, task_id: str, video_path: str) -> None:
        self.task_id = task_id
        self.video_path = video_path
        self._lock = threading.Lock()
        self._state = _initial_state()
        self._state["video_path"] = video_path
        self._runner = ScanRunner(task_id, video_path, self._state, self._lock)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            self._state["status"] = "scanning"
        self._thread = threading.Thread(
            target=self._runner.run, name=f"scan-{self.task_id}", daemon=True
        )
        self._thread.start()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)

    def transcript_snapshot(self) -> dict:
        with self._lock:
            return {
                "transcript": self._state.get("transcript", ""),
                "transcript_chunk": self._state.get("transcript_chunk", ""),
                "transcript_words": self._state.get("transcript_words", ""),
                "stage": self._state.get("stage", ""),
                "stage_label": self._state.get("stage_label", ""),
            }

