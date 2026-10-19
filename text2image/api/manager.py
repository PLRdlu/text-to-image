"""任务管理：接收提交、后台执行、维护内存状态。"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from text2image.api.schemas import GenerateImageRequest, ImageInfo, TaskDetail
from text2image.config import Settings
from text2image.core.errors import Text2ImageError
from text2image.core.schemas import GenParams, GenRequest, ImageResult, ImageReview, PromptRewrite
from text2image.services.generator import GeneratorService
from text2image.utils.logger import get_logger

TaskStateName = Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED"]

_TERMINAL_STATES = {"SUCCEEDED", "FAILED"}


@dataclass
class TaskRecord:
    """单个任务的运行状态（内存态，进程重启即丢失）。"""

    task_id: str
    status: TaskStateName
    request: GenerateImageRequest
    created_at: datetime
    finished_at: datetime | None = None
    final_prompt: str | None = None
    rewrite: PromptRewrite | None = None
    results: list[ImageResult] = field(default_factory=list)
    reviews: list[ImageReview] = field(default_factory=list)
    error: str | None = None

    @property
    def finished(self) -> bool:
        return self.status in _TERMINAL_STATES

    def to_detail(self) -> TaskDetail:
        images = [
            ImageInfo(
                index=index,
                url=f"/api/v1/images/{self.task_id}/{index}",
                local_path=str(item.local_path),
                review=item.review,
            )
            for index, item in enumerate(self.results)
            if item.local_path is not None
        ]
        return TaskDetail(
            task_id=self.task_id,
            status=self.status,
            created_at=self.created_at,
            finished_at=self.finished_at,
            error=self.error,
            prompt=self.request.prompt,
            final_prompt=self.final_prompt,
            rewrite=self.rewrite,
            images=images,
            reviews=self.reviews,
        )


class GenerationManager:
    """管理后台生成任务与并发上限。"""

    def __init__(
        self,
        service: GeneratorService,
        settings: Settings,
        *,
        max_concurrency: int = 3,
    ) -> None:
        self._service = service
        self._settings = settings
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._records: dict[str, TaskRecord] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._log = get_logger()

    # ---------- 对外接口 ----------

    def submit(self, payload: GenerateImageRequest) -> str:
        """登记任务并立即返回 task_id，实际生成在后台执行。"""
        task_id = uuid.uuid4().hex
        record = TaskRecord(
            task_id=task_id,
            status="PENDING",
            request=payload,
            created_at=datetime.now(),
        )
        self._records[task_id] = record
        task = asyncio.create_task(self._run(task_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        self._log.info("已提交任务 {}: {}", task_id, payload.prompt[:40])
        return task_id

    def get(self, task_id: str) -> TaskRecord | None:
        return self._records.get(task_id)

    def recent(self, limit: int = 20) -> list[TaskRecord]:
        records = sorted(self._records.values(), key=lambda item: item.created_at, reverse=True)
        return records[:limit]

    async def shutdown(self) -> None:
        """取消仍在运行的任务并等待收尾。"""
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    # ---------- 内部实现 ----------

    def _build_request(self, payload: GenerateImageRequest) -> GenRequest:
        return GenRequest(
            prompt=payload.prompt,
            style=payload.style,
            params=GenParams(
                model=payload.model or self._settings.image_model,
                size=payload.size,
                n=payload.n,
                negative_prompt=payload.negative_prompt,
                seed=payload.seed,
                prompt_extend=payload.prompt_extend,
                watermark=payload.watermark,
            ),
        )

    async def _run(self, task_id: str) -> None:
        record = self._records[task_id]
        record.status = "RUNNING"
        request = self._build_request(record.request)
        try:
            async with self._semaphore:
                outcome = await self._service.generate(
                    request,
                    rewrite=record.request.rewrite,
                    review=record.request.review,
                )
            record.final_prompt = outcome.final_prompt
            record.rewrite = outcome.rewrite
            record.results = list(outcome.results)
            record.reviews = list(outcome.reviews)
            record.status = "SUCCEEDED"
            self._log.info("任务 {} 完成，共 {} 张", task_id, len(record.results))
        except Text2ImageError as exc:
            record.status = "FAILED"
            record.error = str(exc)
            self._log.warning("任务 {} 失败: {}", task_id, exc)
        except asyncio.CancelledError:
            record.status = "FAILED"
            record.error = "任务被取消"
            raise
        except Exception as exc:  # noqa: BLE001 - 兜底，避免后台任务静默退出
            record.status = "FAILED"
            record.error = f"未预期错误: {exc}"
            self._log.exception("任务 {} 出现未预期错误", task_id)
        finally:
            record.finished_at = datetime.now()
