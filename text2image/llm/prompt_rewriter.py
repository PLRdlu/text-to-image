"""用 DeepSeek 把用户输入改写为高质量文生图提示词。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from text2image.core.schemas import PromptRewrite
from text2image.llm.client import LLMClient
from text2image.llm.prompts import REWRITE_SYSTEM, REWRITE_USER_TEMPLATE
from text2image.utils.logger import get_logger


class _RewritePayload(BaseModel):
    """模型实际返回的字段（不含 original）。"""

    rewritten: str = Field(min_length=1)
    negative_prompt: str | None = None
    tags: list[str] = Field(default_factory=list)


class PromptRewriter:
    """提示词改写器。"""

    def __init__(self, client: LLMClient, *, temperature: float = 0.7) -> None:
        self._client = client
        self._temperature = temperature
        self._log = get_logger()

    async def rewrite(self, prompt: str, style: str | None = None) -> PromptRewrite:
        """返回改写后的提示词；失败时由调用方决定是否回退到原始提示词。"""
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM},
            {
                "role": "user",
                "content": REWRITE_USER_TEMPLATE.format(
                    prompt=prompt, style=style or "不限定，由你判断最合适的风格"
                ),
            },
        ]
        payload = await self._client.complete_model(
            messages, _RewritePayload, temperature=self._temperature
        )
        rewritten = payload.rewritten.strip()
        result = PromptRewrite(
            original=prompt,
            rewritten=rewritten,
            negative_prompt=(payload.negative_prompt or "").strip() or None,
            tags=payload.tags,
            changed=rewritten != prompt,
        )
        self._log.info("提示词改写完成: {} -> {}", prompt[:40], rewritten[:80])
        return result

    async def aclose(self) -> None:
        await self._client.aclose()
