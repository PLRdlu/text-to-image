"""大模型能力：提示词改写与图片质检（DeepSeek，OpenAI 兼容）。"""

from __future__ import annotations

from text2image.config import Settings
from text2image.core.errors import ConfigError
from text2image.llm.client import LLMClient
from text2image.llm.image_reviewer import ImageReviewer
from text2image.llm.prompt_rewriter import PromptRewriter


def create_llm_client(settings: Settings) -> LLMClient:
    """用配置里的 DeepSeek 凭据创建 LLM 客户端。"""
    if not settings.deepseek_api_key:
        raise ConfigError("缺少 DEEPSEEK_API_KEY，无法调用大模型")
    return LLMClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        timeout=settings.deepseek_timeout,
        max_tokens=settings.llm_max_tokens,
    )


def create_prompt_rewriter(settings: Settings) -> PromptRewriter:
    return PromptRewriter(create_llm_client(settings))


def create_image_reviewer(settings: Settings) -> ImageReviewer:
    return ImageReviewer(create_llm_client(settings), threshold=settings.review_threshold)


__all__ = [
    "ImageReviewer",
    "LLMClient",
    "PromptRewriter",
    "create_image_reviewer",
    "create_llm_client",
    "create_prompt_rewriter",
]
