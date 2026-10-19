"""HTTP 路由定义。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse

from text2image.api.manager import GenerationManager
from text2image.api.schemas import (
    GenerateImageRequest,
    HealthResponse,
    TaskCreated,
    TaskDetail,
)
from text2image.config import get_settings

router = APIRouter()


def get_manager(request: Request) -> GenerationManager:
    """从应用状态中取出任务管理器。"""
    manager: GenerationManager = request.app.state.manager
    return manager


ManagerDep = Annotated[GenerationManager, Depends(get_manager)]


@router.get("/healthz", response_model=HealthResponse, tags=["system"])
async def healthz() -> HealthResponse:
    """健康检查：返回当前后端与模型配置。"""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        backend=settings.image_backend,
        model=settings.image_model,
        api_key_configured=bool(settings.dashscope_api_key),
    )


@router.post(
    "/api/v1/images",
    response_model=TaskCreated,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["images"],
)
async def create_image(payload: GenerateImageRequest, manager: ManagerDep) -> TaskCreated:
    """提交一次文生图任务，立即返回 task_id（后台执行）。"""
    task_id = manager.submit(payload)
    return TaskCreated(task_id=task_id, status="PENDING")


@router.get("/api/v1/tasks", response_model=list[TaskDetail], tags=["tasks"])
async def list_tasks(
    manager: ManagerDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[TaskDetail]:
    """列出最近的任务。"""
    return [record.to_detail() for record in manager.recent(limit)]


@router.get("/api/v1/tasks/{task_id}", response_model=TaskDetail, tags=["tasks"])
async def get_task(task_id: str, manager: ManagerDep) -> TaskDetail:
    """查询任务状态与结果。"""
    record = manager.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return record.to_detail()


@router.get("/api/v1/images/{task_id}/{index}", tags=["images"])
async def get_image(task_id: str, index: int, manager: ManagerDep) -> FileResponse:
    """按索引获取生成好的图片文件。"""
    record = manager.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if index < 0 or index >= len(record.results):
        raise HTTPException(status_code=404, detail="图片索引越界")
    local_path = record.results[index].local_path
    if local_path is None or not local_path.exists():
        raise HTTPException(status_code=404, detail="图片尚未就绪")
    return FileResponse(local_path, media_type="image/png", filename=local_path.name)
