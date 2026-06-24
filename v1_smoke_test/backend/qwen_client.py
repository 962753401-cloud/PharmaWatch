"""Thin wrapper around qwen3-vl-plus via the OpenAI-compatible DashScope endpoint."""
from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from config import Settings

logger = logging.getLogger("qwen_client")

DEFAULT_QUESTION = "请描述这段药店监控视频里发生了什么，并指出是否存在医保违规嫌疑。"

SYSTEM_PROMPT = (
    "你是医保基金稽查 AI 助手，专门解读药店监控视频画面。"
    "面对每一段药店监控视频，请基于实际画面，按以下要求输出：\n"
    "1) scene_summary：用 3-6 句中文描述场景，覆盖人员数量、所处区域（如收银台/货架/出入口）、"
    "关键动作（如递医保卡、递现金、扫码、拿药、换商品、点货）、可疑迹象（如药物串换、商家返现、"
    "多卡同刷、超量配药）。\n"
    "2) answer：针对用户问题，用中文给出贴合画面的回答；若画面信息不足以回答某点，要明确说明。\n"
    "3) risk_hint：对是否存在医保违规嫌疑给出粗略判断，取值仅限 \"none\"（无明显风险）、"
    "\"suspect\"（存在疑点需复核）、\"violation\"（明显违规迹象）。\n"
    "4) raw_observations：列出 3-8 条帧级线索短句，例如\"约第 6 秒，柜台上出现一张卡片\"。\n"
    "严禁臆造画面中没有的内容；如视频不清晰或证据不足，请如实说明。"
    "必须严格输出一个 JSON 对象，键为 scene_summary、answer、risk_hint、raw_observations，"
    "其中 raw_observations 为字符串数组，其它为字符串。"
)


@dataclass
class AnalyzeResult:
    ok: bool
    scene_summary: str = ""
    answer: str = ""
    risk_hint: str = "none"
    raw_observations: list | None = None
    raw_text: str = ""
    elapsed_ms: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    error_code: str | None = None
    message: str | None = None

    def to_dict(self) -> dict:
        if self.ok:
            return {
                "ok": True,
                "scene_summary": self.scene_summary,
                "answer": self.answer,
                "risk_hint": self.risk_hint,
                "raw_observations": self.raw_observations or [],
                "elapsed_ms": self.elapsed_ms,
                "tokens": {
                    "prompt": self.prompt_tokens,
                    "completion": self.completion_tokens,
                    "total": self.total_tokens,
                },
            }
        return {
            "ok": False,
            "error_code": self.error_code or "unknown_error",
            "message": self.message or "调用失败",
            "elapsed_ms": self.elapsed_ms,
        }


def _normalize_risk(value: Any) -> str:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"none", "suspect", "violation"}:
            return v
    return "none"


def _coerce_observations(value: Any) -> list:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _parse_model_text(text: str) -> dict:
    """Try hard to coax a JSON object out of the model output."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return {}


class QwenVLClient:
    """Encapsulates the qwen3-vl-plus call so the FastAPI layer stays thin."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                timeout=self.settings.timeout_seconds,
            )
        return self._client

    def analyze_video(
        self,
        video_bytes: bytes,
        mime: str,
        question: str | None,
    ) -> AnalyzeResult:
        if not self.settings.has_api_key:
            return AnalyzeResult(
                ok=False,
                error_code="api_key_missing",
                message="后端未配置 DASHSCOPE_API_KEY，请在 .env 中填写后重启服务。",
            )

        user_question = (question or "").strip() or DEFAULT_QUESTION
        b64 = base64.b64encode(video_bytes).decode("ascii")
        data_url = f"data:{mime or 'video/mp4'};base64,{b64}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": data_url},
                        "fps": self.settings.fps,
                    },
                    {
                        "type": "text",
                        "text": (
                            "用户问题：\n"
                            f"{user_question}\n\n"
                            "请严格输出一个 JSON 对象，包含 scene_summary、answer、"
                            "risk_hint(none|suspect|violation)、raw_observations(字符串数组) 四个字段。"
                        ),
                    },
                ],
            },
        ]

        client = self._get_client()
        started = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                model=self.settings.model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
        except AuthenticationError as exc:
            return self._fail("api_key_invalid", "API Key 无效或被拒绝，请检查 DASHSCOPE_API_KEY。", started, exc)
        except RateLimitError as exc:
            return self._fail("rate_limited", "DashScope 限流，请稍后再试。", started, exc)
        except APITimeoutError as exc:
            return self._fail("timeout", "调用 qwen3-vl-plus 超时，建议缩短视频或重试。", started, exc)
        except BadRequestError as exc:
            msg = self._extract_message(exc) or "请求被模型拒绝，请检查视频格式或大小。"
            return self._fail("bad_request", msg, started, exc)
        except APIStatusError as exc:
            msg = self._extract_message(exc) or f"DashScope 返回异常状态：{exc.status_code}"
            return self._fail("api_error", msg, started, exc)
        except APIConnectionError as exc:
            return self._fail("network_error", "无法连接 DashScope，请检查网络或 base_url。", started, exc)
        except Exception as exc:
            return self._fail("unknown_error", "调用失败，请查看后端日志。", started, exc)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        choice = completion.choices[0] if completion.choices else None
        text = ""
        if choice and choice.message and choice.message.content:
            content = choice.message.content
            text = content if isinstance(content, str) else str(content)

        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None

        parsed = _parse_model_text(text)
        if parsed:
            result = AnalyzeResult(
                ok=True,
                scene_summary=str(parsed.get("scene_summary", "")).strip(),
                answer=str(parsed.get("answer", "")).strip(),
                risk_hint=_normalize_risk(parsed.get("risk_hint")),
                raw_observations=_coerce_observations(parsed.get("raw_observations")),
                raw_text=text,
                elapsed_ms=elapsed_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        else:
            result = AnalyzeResult(
                ok=True,
                scene_summary="",
                answer=text.strip() or "模型未返回内容。",
                risk_hint="none",
                raw_observations=[],
                raw_text=text,
                elapsed_ms=elapsed_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        logger.info(
            "qwen3-vl-plus ok elapsed=%dms tokens=%s/%s/%s",
            elapsed_ms,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )
        return result

    @staticmethod
    def _extract_message(exc: Exception) -> str | None:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                msg = err.get("message")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
        msg = getattr(exc, "message", None)
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
        return None

    def _fail(
        self,
        code: str,
        message: str,
        started: float,
        exc: Exception,
    ) -> AnalyzeResult:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning("qwen3-vl-plus failed code=%s elapsed=%dms err=%s", code, elapsed_ms, exc)
        return AnalyzeResult(
            ok=False,
            error_code=code,
            message=message,
            elapsed_ms=elapsed_ms,
        )