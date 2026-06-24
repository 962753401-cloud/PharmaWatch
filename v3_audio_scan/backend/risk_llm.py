"""Stage 2 risk-control LLM (qwen3.5-flash) via OpenAI-compatible DashScope."""
from __future__ import annotations

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

from config import SETTINGS

logger = logging.getLogger("risk_llm")

RISK_SYSTEM_PROMPT = (
    "你现在是音频内容审核助手，负责两个任务："
    "第一，系统已通过拼音精准匹配在转写文本中检测到触发关键词，请结合上下文复核该关键词是否确实出现（排除同音字、谐音导致的误匹配）。"
    "第二，请判断对话中是否存在暗语、谐音、变体表达等可疑模式（例如故意用代称、隐晦说法规避识别）。"
    "请分析给出的对话片段，综合以上两点给出结论。"
    "必须严格输出一个 JSON 对象，键为 verdict 和 reason。"
    "verdict 取值仅限 \"确认\"、\"暗语\" 或 \"误报\"；reason 为一句话中文理由。"
)


@dataclass
class RiskResult:
    ok: bool
    verdict: str = "合规"
    reason: str = ""
    model_used: str = ""
    elapsed_ms: int = 0
    error_code: str | None = None
    message: str | None = None

    def to_dict(self) -> dict:
        if self.ok:
            return {
                "ok": True,
                "verdict": self.verdict,
                "reason": self.reason,
                "model_used": self.model_used,
                "elapsed_ms": self.elapsed_ms,
            }
        return {
            "ok": False,
            "verdict": "合规",
            "reason": self.message or "风控判定失败，默认合规",
            "error_code": self.error_code or "unknown_error",
            "elapsed_ms": self.elapsed_ms,
        }


def _build_user_prompt(context: str, keyword: str = "") -> str:
    kw_line = f"系统检测到的触发关键词为：{keyword}\n" if keyword else ""
    return (
        f"{kw_line}\n"
        "对话内容：\n"
        f"{context}\n\n"
        f"请分析以上对话，完成两项判断：\n"
        f"1. 触发关键词「{keyword}」是否真实出现在上下文中（排除同音字、谐音误匹配）。\n"
        f"2. 对话中是否存在暗语、谐音、变体表达等可疑模式。\n"
        f"判定规则：关键词真实出现且无暗语→\"确认\"；存在暗语/谐音/变体等可疑表达→\"暗语\"；"
        f"关键词未真实出现且无暗语→\"误报\"。\n"
        "仅输出 JSON：{\"verdict\": \"确认\"|\"暗语\"|\"误报\", \"reason\": \"一句话理由\"}。"
    )


def _parse_text(text: str) -> dict:
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


def _normalize_verdict(value: Any) -> str:
    if isinstance(value, str):
        v = value.strip()
        if "暗语" in v:
            return "暗语"
        if "确认" in v or "涉嫌" in v or "违规" in v or "suspect" in v.lower():
            return "确认"
        if "误报" in v or "合规" in v or "正常" in v or "ok" in v.lower() or "safe" in v.lower():
            return "误报"
    return "误报"


class RiskLLMClient:
    """Calls qwen3.5-flash (fallback qwen-plus) for Stage 2 context risk control."""

    def __init__(self) -> None:
        self._client: OpenAI | None = None
        self._resolved_model: str | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=SETTINGS.api_key,
                base_url=SETTINGS.base_url,
                timeout=SETTINGS.llm_timeout,
            )
        return self._client

    def _probe_model(self, candidate: str) -> bool:
        """Quick 1-token probe to confirm a model name is usable."""
        client = self._get_client()
        try:
            client.chat.completions.create(
                model=candidate,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0,
                timeout=20,
            )
            return True
        except BadRequestError as exc:
            body = getattr(exc, "body", None)
            msg = ""
            if isinstance(body, dict):
                err = body.get("error")
                if isinstance(err, dict):
                    msg = str(err.get("message", ""))
            logger.warning("probe %s failed (bad_request): %s", candidate, msg)
            return False
        except APIStatusError as exc:
            logger.warning("probe %s failed (status %s)", candidate, exc.status_code)
            return False
        except Exception as exc:
            logger.warning("probe %s failed: %s", candidate, exc)
            return False

    def _resolve_model(self) -> str:
        if self._resolved_model:
            return self._resolved_model
        primary = SETTINGS.llm_model
        if self._probe_model(primary):
            self._resolved_model = primary
            logger.info("LLM model resolved: %s", primary)
            return primary
        fallback = SETTINGS.llm_fallback
        logger.warning("model %s unavailable, trying fallback %s", primary, fallback)
        if self._probe_model(fallback):
            self._resolved_model = fallback
            logger.info("LLM model resolved (fallback): %s", fallback)
            return fallback
        # Last resort: assume primary works (probe may be overly strict); cache it.
        self._resolved_model = primary
        return primary

    def judge(self, context: str, keyword: str = "") -> RiskResult:
        if not SETTINGS.has_api_key:
            return RiskResult(
                ok=False,
                error_code="api_key_missing",
                message="未配置 DASHSCOPE_API_KEY",
            )
        context = (context or "").strip()
        if not context:
            return RiskResult(ok=True, verdict="合规", reason="无对话内容", model_used="")

        model = self._resolve_model()
        client = self._get_client()
        started = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": RISK_SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(context, keyword)},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
                timeout=SETTINGS.llm_timeout,
            )
        except AuthenticationError as exc:
            return self._fail("api_key_invalid", "API Key 无效", started, exc)
        except RateLimitError as exc:
            return self._fail("rate_limited", "DashScope 限流", started, exc)
        except APITimeoutError as exc:
            return self._fail("timeout", "调用 LLM 超时", started, exc)
        except BadRequestError as exc:
            msg = self._extract_message(exc) or "请求被拒绝"
            return self._fail("bad_request", msg, started, exc)
        except APIStatusError as exc:
            msg = self._extract_message(exc) or f"DashScope 异常: {exc.status_code}"
            return self._fail("api_error", msg, started, exc)
        except APIConnectionError as exc:
            return self._fail("network_error", "无法连接 DashScope", started, exc)
        except Exception as exc:
            return self._fail("unknown_error", "风控调用失败", started, exc)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        choice = completion.choices[0] if completion.choices else None
        text = ""
        if choice and choice.message and choice.message.content:
            content = choice.message.content
            text = content if isinstance(content, str) else str(content)

        parsed = _parse_text(text)
        if parsed:
            verdict = _normalize_verdict(parsed.get("verdict"))
            reason = str(parsed.get("reason", "")).strip() or ("关键词确认出现" if verdict == "确认" else "关键词未真实出现")
        else:
            low = text.strip()
            verdict = _normalize_verdict(low) if low else "合规"
            reason = low or "模型未返回内容"
            if not low:
                verdict = "误报"

        logger.info("risk judge verdict=%s model=%s elapsed=%dms", verdict, model, elapsed_ms)
        return RiskResult(
            ok=True,
            verdict=verdict,
            reason=reason,
            model_used=model,
            elapsed_ms=elapsed_ms,
        )

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

    def _fail(self, code: str, message: str, started: float, exc: Exception) -> RiskResult:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning("risk judge failed code=%s elapsed=%dms err=%s", code, elapsed_ms, exc)
        return RiskResult(
            ok=False,
            error_code=code,
            message=message,
            elapsed_ms=elapsed_ms,
        )
