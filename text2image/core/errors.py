"""项目统一异常体系。

所有业务异常继承自 Text2ImageError，方便在 FastAPI 层统一捕获并映射为
合适的 HTTP 状态码。
"""

from __future__ import annotations


class Text2ImageError(Exception):
    """项目所有异常的共同基类。"""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __str__(self) -> str:
        if self.detail:
            return f"{self.message} ({self.detail})"
        return self.message


class ConfigError(Text2ImageError):
    """配置缺失或非法，例如未设置 DASHSCOPE_API_KEY。"""


class EngineError(Text2ImageError):
    """推理引擎相关的错误基类。"""


class SubmissionError(EngineError):
    """提交出图任务失败。"""


class TaskFailedError(EngineError):
    """任务已提交，但服务端返回失败状态。"""


class PollingTimeoutError(EngineError):
    """轮询任务状态超时。"""


class DownloadError(EngineError):
    """下载生成图片失败。"""


class LLMError(Text2ImageError):
    """调用 DeepSeek / 质检模型失败。"""


class RateLimitError(Text2ImageError):
    """触发了本地或服务端的限流。"""
