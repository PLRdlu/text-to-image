"""FastAPI 应用装配。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from text2image import __version__
from text2image.api.manager import GenerationManager
from text2image.api.routes import router
from text2image.config import get_settings
from text2image.services.generator import build_service
from text2image.utils.logger import get_logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动时装配服务，关闭时释放资源。"""
    settings = get_settings()
    settings.ensure_dirs()
    setup_logging(settings.log_dir, settings.log_level)
    logger = get_logger()

    service = build_service(settings)
    manager = GenerationManager(
        service,
        settings,
        max_concurrency=settings.image_max_concurrency,
    )
    app.state.service = service
    app.state.manager = manager
    logger.info(
        "服务启动: backend={} model={} max_concurrency={}",
        settings.image_backend,
        settings.image_model,
        settings.image_max_concurrency,
    )
    try:
        yield
    finally:
        await manager.shutdown()
        await service.aclose()
        logger.info("服务已关闭")


def create_app() -> FastAPI:
    """创建 FastAPI 应用。"""
    app = FastAPI(
        title="text2image",
        version=__version__,
        description="通义万相文生图服务：出图 + DeepSeek 提示词改写与质检",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
