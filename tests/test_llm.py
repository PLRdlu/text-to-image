"""大模型客户端与提示词改写测试。

注意：openai v3 底层走的是 httpx2，respx 无法拦截，因此这里用注入假客户端的方式
测试我们自己的 JSON 解析与校验逻辑，避免任何真实网络访问。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from openai import OpenAIError
from pydantic import BaseModel

from text2image.core.errors import LLMError
from text2image.core.schemas import PromptRewrite
from text2image.llm.client import LLMClient
from text2image.llm.prompt_rewriter import PromptRewriter


class _Payload(BaseModel):
    value: int


class _FakeCompletions:
    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self._content = content
        self._error = error

    async def create(self, **kwargs: Any) -> Any:
        if self._error is not None:
            raise self._error
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeOpenAI:
    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(content, error))

    async def close(self) -> None:
        return None


def _client(content: str | None = None, error: Exception | None = None) -> LLMClient:
    client = LLMClient(
        api_key="sk-test", base_url="https://api.deepseek.com", model="deepseek-flash"
    )
    client._client = _FakeOpenAI(content, error)  # type: ignore[assignment]
    return client


async def test_parses_fenced_json() -> None:
    fenced = '\x60\x60\x60json\n{"value": 7}\n\x60\x60\x60'
    result = await _client(fenced).complete_model([{"role": "user", "content": "hi"}], _Payload)
    assert result.value == 7


async def test_parses_plain_json() -> None:
    result = await _client('{"value": 1}').complete_model([], _Payload)
    assert result.value == 1


async def test_bad_json_raises_llm_error() -> None:
    try:
        await _client("not json at all").complete_model([], _Payload)
    except LLMError as exc:
        assert "JSON" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("应当抛出 LLMError")


async def test_schema_mismatch_raises_llm_error() -> None:
    try:
        await _client('{"other": 1}').complete_model([], _Payload)
    except LLMError as exc:
        assert "结构" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("应当抛出 LLMError")


async def test_api_error_wrapped() -> None:
    try:
        await _client(error=OpenAIError("boom")).complete_model([], _Payload)
    except LLMError as exc:
        assert "调用大模型失败" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("应当抛出 LLMError")


async def test_empty_content_raises() -> None:
    try:
        await _client(None).complete_model([], _Payload)
    except LLMError as exc:
        assert "空内容" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("应当抛出 LLMError")


class _StubRewriterClient:
    """鸭子类型的假客户端，用于测试 PromptRewriter 的组装逻辑。"""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.last_messages: list[dict[str, Any]] = []

    async def complete_model(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Any:
        self.last_messages = messages
        return schema.model_validate(self._payload)

    async def aclose(self) -> None:
        return None


async def test_prompt_rewriter_fills_original() -> None:
    stub = _StubRewriterClient(
        {"rewritten": "a cat", "negative_prompt": "blurry", "tags": ["cat", "pet"]}
    )
    rewriter = PromptRewriter(stub)  # type: ignore[arg-type]
    result = await rewriter.rewrite("一只猫", style="photo")
    assert isinstance(result, PromptRewrite)
    assert result.original == "一只猫"
    assert result.rewritten == "a cat"
    assert result.negative_prompt == "blurry"
    assert result.tags == ["cat", "pet"]
    assert result.changed is True
    assert "一只猫" in stub.last_messages[1]["content"]


async def test_prompt_rewriter_marks_unchanged() -> None:
    stub = _StubRewriterClient({"rewritten": "一只猫", "tags": []})
    rewriter = PromptRewriter(stub)  # type: ignore[arg-type]
    result = await rewriter.rewrite("一只猫")
    assert result.changed is False
    assert result.negative_prompt is None
