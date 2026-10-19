"""HTTP 接口的请求/响应模型。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from text2image.core.schemas import ImageReview, PromptRewrite, normalize_size


class GenerateImageRequest(BaseModel):
    """提交一次文生图任务。"""

    prompt: str = Field(min_length=1, description="提示词")
    style: str | None = Field(default=None, description="风格提示")
    model: str | None = Field(default=None, description="覆盖默认出图模型")
    size: str = Field(default="1024*1024", description="尺寸，如 1024*1024")
    n: int = Field(default=1, ge=1, le=6, description="图片张数")
    negative_prompt: str | None = None
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    prompt_extend: bool = True
    watermark: bool = False
    rewrite: bool | None = Field(default=None, description="覆盖是否改写提示词")
    review: bool | None = Field(default=None, description="覆盖是否做图片质检")

    @field_validator("prompt")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt 不能为空")
        return value

    @field_validator("size")
    @classmethod
    def _size(cls, value: str) -> str:
        return normalize_size(value)


class TaskCreated(BaseModel):
    task_id: str
    status: str


class ImageInfo(BaseModel):
    index: int
    url: str = Field(description="通过本服务获取图片的相对地址")
    local_path: str
    review: ImageReview | None = None


class TaskDetail(BaseModel):
    task_id: str
    status: str
    created_at: datetime
    finished_at: datetime | None = None
    error: str | None = None
    prompt: str
    final_prompt: str | None = None
    rewrite: PromptRewrite | None = None
    images: list[ImageInfo] = Field(default_factory=list)
    reviews: list[ImageReview] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    backend: str
    model: str
    api_key_configured: bool
