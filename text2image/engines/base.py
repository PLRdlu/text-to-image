"""出图引擎抽象层。

上层只依赖 BaseEngine，具体后端（万相 / mock / 未来的 Qwen-Image）可替换，
方便在不改动业务代码的前提下切换模型。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from text2image.core.schemas import GenRequest, ImageResult


class BaseEngine(ABC):
    """所有出图后端的统一接口。"""

    name: str = "base"

    @abstractmethod
    async def generate(self, request: GenRequest) -> list[ImageResult]:
        """执行一次出图，返回图片 URL 级别的结果（尚未下载到本地）。"""

    async def aclose(self) -> None:
        """释放引擎持有的资源（如 HTTP 连接池）。"""
        return None
