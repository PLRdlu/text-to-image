"""OpenAI 兼容的大模型客户端封装（用于 DeepSeek）。"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar, cast

from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from text2image.core.errors import LLMError
from text2image.utils.logger import get_logger

TModel = TypeVar("TModel", bound=BaseModel)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LLMClient:
    """对 DeepSeek（OpenAI 兼容）的薄封装，支持 JSON 结构化输出。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        max_tokens: int = 4096,
    ) -> None:
        if not api_key:
            raise LLMError("缺少大模型 API Key")
        self._model = model
        self._max_tokens = max_tokens
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._log = get_logger()

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        """发起一次对话补全，返回文本内容。"""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": cast(Any, messages),
            "temperature": temperature,
            "max_tokens": max_tokens or self._max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            raise LLMError("调用大模型失败", detail=str(exc)) from exc

        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise LLMError("大模型返回了空内容")
        return content

    async def complete_model(
        self,
        messages: list[dict[str, Any]],
        schema: type[TModel],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> TModel:
        """要求模型返回 JSON，并校验成指定 Pydantic 模型。"""
        raw = await self.complete(
            messages,
            temperature=temperature,
            json_mode=True,
            max_tokens=max_tokens,
        )
        payload = _strip_fences(raw)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise LLMError("大模型返回的 JSON 无法解析", detail=f"{exc}; raw={raw[:500]}") from exc
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise LLMError(
                "大模型返回的 JSON 不符合预期结构", detail=f"{exc}; raw={raw[:500]}"
            ) from exc

    async def aclose(self) -> None:
        await self._client.close()


def _strip_fences(text: str) -> str:
    """去掉模型可能多输出的 Markdown 代码块围栏。"""
    return _FENCE_RE.sub("", text.strip()).strip()
