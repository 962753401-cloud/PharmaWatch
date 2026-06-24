"""Fun-ASR realtime client over the DashScope WebSocket protocol.

Streams a 16kHz mono 16-bit WAV chunk to fun-asr-realtime and collects
sentence-level + word-level timestamps (ms, relative to chunk start).
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

import websocket

logger = logging.getLogger("v3_asr")

_WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference/"
_CHUNK_BYTES = 3200  # 100ms @ 16kHz 16-bit mono


@dataclass
class Word:
    text: str
    begin_time_ms: int
    end_time_ms: int


@dataclass
class Sentence:
    text: str
    begin_time_ms: int
    end_time_ms: int
    words: list[Word] = field(default_factory=list)
    pending: bool = False


@dataclass
class ASRResult:
    ok: bool
    sentences: list[Sentence] = field(default_factory=list)
    error: str | None = None
    elapsed_ms: int = 0

    def full_text(self) -> str:
        return "".join(s.text for s in self.sentences)

    def all_words(self) -> list[Word]:
        out: list[Word] = []
        for s in self.sentences:
            out.extend(s.words)
        return out


def _transcribe_once(
    wav_path: str,
    api_key: str,
    model: str = "fun-asr-realtime",
    send_interval: float = 0.0,
    timeout: float = 120.0,
) -> ASRResult:
    if not api_key:
        return ASRResult(ok=False, error="api_key_missing")

    task_id = uuid.uuid4().hex[:32]
    started = time.perf_counter()
    state = {
        "sentences": [],
        "error": None,
        "finished": False,
        "started": False,
    }
    state_lock = threading.Lock()

    def send_run_task(ws):
        msg = {
            "header": {
                "action": "run-task",
                "task_id": task_id,
                "streaming": "duplex",
            },
            "payload": {
                "task_group": "audio",
                "task": "asr",
                "function": "recognition",
                "model": model,
                "parameters": {"sample_rate": 16000, "format": "wav"},
                "input": {},
            },
        }
        ws.send(json.dumps(msg))

    def send_finish_task(ws):
        msg = {
            "header": {
                "action": "finish-task",
                "task_id": task_id,
                "streaming": "duplex",
            },
            "payload": {"input": {}},
        }
        ws.send(json.dumps(msg))

    def send_audio_stream(ws):
        chunk_size = _CHUNK_BYTES
        try:
            with open(wav_path, "rb") as f:
                f.read(44)
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    ws.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)
                    if send_interval > 0:
                        time.sleep(send_interval)
            send_finish_task(ws)
        except Exception as e:
            with state_lock:
                state["error"] = "audio_stream_error: " + str(e)
            logger.exception("ASR audio stream failed")
            try:
                ws.close()
            except Exception:
                pass

    def on_open(ws):
        logger.info("ASR ws connected, sending run-task")
        send_run_task(ws)

    def on_message(ws, data):
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            return
        header = message.get("header", {})
        event = header.get("event")
        if event == "task-started":
            with state_lock:
                state["started"] = True
            threading.Thread(target=send_audio_stream, args=(ws,), daemon=True).start()
        elif event == "result-generated":
            payload = message.get("payload", {})
            output = payload.get("output", {})
            sentence = output.get("sentence", {}) or {}
            text = sentence.get("text", "") or ""
            is_end = sentence.get("sentence_end", False)
            if not text:
                return
            words_raw = sentence.get("words", []) or []
            words = [
                Word(
                    text=str(w.get("text", "")),
                    begin_time_ms=int(w.get("begin_time", 0) or 0),
                    end_time_ms=int(w.get("end_time", 0) or 0),
                )
                for w in words_raw
                if str(w.get("text", "")).strip()
            ]
            sent = Sentence(
                text=text,
                begin_time_ms=int(sentence.get("begin_time", 0) or 0),
                end_time_ms=int(sentence.get("end_time", 0) or 0),
                words=words,
                pending=not is_end,
            )
            with state_lock:
                if is_end:
                    state["sentences"].append(sent)
                else:
                    if state["sentences"] and state["sentences"][-1].pending:
                        state["sentences"][-1] = sent
                    else:
                        state["sentences"].append(sent)
        elif event == "task-finished":
            with state_lock:
                state["finished"] = True
            ws.close()
        elif event == "task-failed":
            err = header.get("error_message") or "task_failed"
            with state_lock:
                state["error"] = err
                state["finished"] = True
            ws.close()

    def on_error(ws, error):
        with state_lock:
            state["error"] = str(error)
            state["finished"] = True
        logger.error("ASR ws error: %s", error)

    def on_close(ws, code, msg):
        with state_lock:
            state["finished"] = True

    headers = [
        "Authorization: bearer %s" % api_key,
        "X-DashScope-DataInspection: enable",
    ]

    ws_app = websocket.WebSocketApp(
        _WS_URL,
        header=headers,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    runner = threading.Thread(
        target=ws_app.run_forever,
        kwargs={"ping_timeout": timeout},
        daemon=True,
    )
    runner.start()
    runner.join(timeout=timeout)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    with state_lock:
        sentences = [s for s in state["sentences"] if not s.pending]
        err = state["error"]
    if err and not sentences:
        return ASRResult(ok=False, error=err, elapsed_ms=elapsed_ms)
    return ASRResult(ok=True, sentences=sentences, error=err, elapsed_ms=elapsed_ms)

def transcribe(
    wav_path: str,
    api_key: str,
    model: str = "fun-asr-realtime",
    send_interval: float = 0.0,
    timeout: float = 120.0,
    retries: int = 3,
) -> ASRResult:
    """Transcribe with automatic retry on transient network errors."""
    import time as _time
    last = ASRResult(ok=False, error="not_attempted")
    for attempt in range(retries):
        result = _transcribe_once(wav_path, api_key, model, send_interval, timeout)
        if result.ok and result.sentences:
            return result
        # If we got some sentences despite an error, return them.
        if result.sentences:
            return result
        err = result.error or ""
        logger.warning("ASR attempt %d/%d failed: %s", attempt + 1, retries, err)
        last = result
        if attempt < retries - 1:
            _time.sleep(2 * (attempt + 1))  # backoff
    return last



def mock_transcribe(wav_path, chunk_start_sec=0.0):
    """Mock ASR for testing without API quota. Returns canned transcript."""
    import os
    script = (
        "你好我想把这几盒感冒药换成那个红参礼盒。"
        "可以的，价格差不多，我在电脑上给你打成感冒药。"
        "另外能不能帮我套个现，弄点现金出来。"
        "没问题，多开点就行，先存着。"
    )
    chars = list(script)
    n = len(chars)
    dur = 18.0
    words = []
    for i, ch in enumerate(chars):
        t0 = chunk_start_sec + (i / n) * dur
        t1 = chunk_start_sec + ((i + 1) / n) * dur
        words.append(Word(text=ch, begin_time_ms=int(t0 * 1000), end_time_ms=int(t1 * 1000)))
    s = Sentence(text=script, begin_time_ms=0, end_time_ms=int(dur * 1000), words=words)
    return ASRResult(ok=True, sentences=[s], elapsed_ms=0)