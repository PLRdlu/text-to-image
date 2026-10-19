"""用 DeepSeek 的多模态能力对生成图片做质检。"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from text2image.core.errors import LLMError
from text2image.core.schemas import ImageReview
from text2image.llm.client import LLMClient
from text2image.llm.prompts import REVIEW_SYSTEM, REVIEW_USER_TEMPLATE
from text2image.utils.logger import get_logger

_MIME_BY_FORMAT = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}


def to_data_url(image_path: Path, *, max_side: int = 1024) -> str:
    """把本地图片编码为 data URL（必要时等比缩小以节省 token）。"""
    if not image_path.exists():
        raise LLMError("待质检的图片不存在", detail=str(image_path))

    with Image.open(image_path) as opened:
        opened.load()
        image: Image.Image = opened
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        if max(image.size) > max_side:
            ratio = max_side / max(image.size)
            new_size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
            image = image.resize(new_size)
        fmt = "JPEG" if image.mode == "L" else "PNG"
        mime = _MIME_BY_FORMAT[fmt]
        buffer = BytesIO()
        image.save(buffer, format=fmt)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class ImageReviewer:
    """用视觉模型判断图片与提示词的贴合度与质量。"""

    def __init__(
        self,
        client: LLMClient,
        *,
        threshold: int = 60,
        max_side: int = 1024,
    ) -> None:
        self._client = client
        self._threshold = threshold
        self._max_side = max_side
        self._log = get_logger()

    async def review(self, image_path: Path, prompt: str) -> ImageReview:
        data_url = to_data_url(image_path, max_side=self._max_side)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": REVIEW_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": REVIEW_USER_TEMPLATE.format(prompt=prompt)},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
                ],
            },
        ]
        review = await self._client.complete_model(messages, ImageReview, temperature=0.2)
        review.passed = review.passed and review.score >= self._threshold
        self._log.info(
            "图片质检完成: score={} passed={} issues={}",
            review.score,
            review.passed,
            len(review.issues),
        )
        return review

    async def aclose(self) -> None:
        await self._client.aclose()
