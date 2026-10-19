"""请求、响应与任务状态的 Pydantic 数据模型。

尺寸格式约定：
    * DashScope 协议使用星号，例如 "1024*1024"；
    * OpenAI 兼容协议使用字母 x，例如 "1024x1024"。
本模块统一以星号作为内部表示，兼容读取字母 x 形式。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# 任务状态：与 DashScope 异步任务返回的取值保持一致
TaskState = Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELED"]

_SIZE_RE = re.compile(r"^\s*(\d{2,5})\s*[*xX]\s*(\d{2,5})\s*$")


def normalize_size(value: str) -> str:
    """把 1024x1024 / 1024*1024 统一规范成 1024*1024。"""
    match = _SIZE_RE.match(value)
    if not match:
        raise ValueError(f"尺寸格式非法: {value!r}，应形如 '1024*1024' 或 '1024x1024'")
    width, height = int(match.group(1)), int(match.group(2))
    pixels = width * height
    if not (512 * 512 <= pixels <= 2048 * 2048):
        raise ValueError(f"像素面积需在 512*512 ~ 2048*2048 之间，当前为 {width}x{height}")
    long_side, short_side = max(width, height), min(width, height)
    if long_side > short_side * 8:
        raise ValueError(f"宽高比不得超过 8:1，当前为 {width}x{height}")
    return f"{width}*{height}"


class GenParams(BaseModel):
    """单次出图的生成参数。"""

    model: str = "wan2.6-t2i"
    size: str = "1024*1024"
    n: int = Field(default=1, ge=1, le=6, description="输出图片张数")
    negative_prompt: str | None = None
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    prompt_extend: bool = True
    watermark: bool = False

    @field_validator("size")
    @classmethod
    def _validate_size(cls, value: str) -> str:
        return normalize_size(value)


class GenRequest(BaseModel):
    """上层提交的一次文生图请求。"""

    prompt: str = Field(min_length=1, description="原始提示词")
    style: str | None = Field(default=None, description="风格预设名称，如 cyberpunk")
    params: GenParams = Field(default_factory=GenParams)

    @field_validator("prompt")
    @classmethod
    def _strip_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt 不能为空字符串")
        return value


class PromptRewrite(BaseModel):
    """DeepSeek 提示词改写结果。"""

    original: str
    rewritten: str
    negative_prompt: str | None = None
    tags: list[str] = Field(default_factory=list)
    changed: bool = True


class ImageReview(BaseModel):
    """图片质检结果（由视觉模型给出）。"""

    score: int = Field(default=0, ge=0, le=100)
    passed: bool = False
    issues: list[str] = Field(default_factory=list)
    suggestion: str | None = None


class ImageResult(BaseModel):
    """单张生成图片的结果与元数据。"""

    task_id: str
    image_url: str
    local_path: Path | None = None
    prompt: str = ""
    params: GenParams = Field(default_factory=GenParams)
    seed: int | None = None
    elapsed_ms: int | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    review: ImageReview | None = None


class TaskStatus(BaseModel):
    """任务状态查询结果。"""

    task_id: str
    status: TaskState
    progress: int | None = Field(default=None, ge=0, le=100)
    results: list[ImageResult] = Field(default_factory=list)
    error: str | None = None

    @property
    def finished(self) -> bool:
        return self.status in ("SUCCEEDED", "FAILED", "CANCELED")


class GenerationOutcome(BaseModel):
    """一次完整生成（改写 → 出图 → 下载 → 质检）的结果汇总。"""

    user_prompt: str
    final_prompt: str
    rewrite: PromptRewrite | None = None
    results: list[ImageResult] = Field(default_factory=list)
    reviews: list[ImageReview] = Field(default_factory=list)

    @property
    def images(self) -> list[Path]:
        return [item.local_path for item in self.results if item.local_path is not None]
