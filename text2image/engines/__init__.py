"""出图引擎工厂。"""

from __future__ import annotations

from text2image.config import Settings
from text2image.core.errors import ConfigError
from text2image.engines.base import BaseEngine
from text2image.engines.mock import MockEngine
from text2image.engines.wanx import WanxEngine

_BACKENDS = {
    "wanx": WanxEngine,
    "mock": MockEngine,
}


def create_engine(settings: Settings) -> BaseEngine:
    """按配置创建出图引擎。"""
    backend = (settings.image_backend or "wanx").lower()
    factory = _BACKENDS.get(backend)
    if factory is None:
        raise ConfigError(
            f"未知的出图后端: {backend}",
            detail=f"可选值: {', '.join(sorted(_BACKENDS))}",
        )
    if backend == "wanx" and not settings.dashscope_api_key:
        raise ConfigError("缺少 DASHSCOPE_API_KEY，无法使用 wanx 后端")
    return factory(settings)


__all__ = ["BaseEngine", "MockEngine", "WanxEngine", "create_engine"]
