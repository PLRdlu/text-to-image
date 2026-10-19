"""通义万相出图引擎（DashScope HTTP 同步接口）。

以 `wan2.6-t2i` 为例，接口为：

    POST {base}/api/v1/services/aigc/multimodal-generation/generation
    {
      "model": "wan2.6-t2i",
      "input": {"messages": [{"role": "user", "content": [{"text": "<prompt>"}]}]},
      "parameters": {...}
    }

该接口一次请求直接返回图片 URL（无需轮询），URL 有效期 24 小时。
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from text2image.config import Settings
from text2image.core.errors import SubmissionError, TaskFailedError
from text2image.core.schemas import GenRequest, ImageResult
from text2image.engines.base import BaseEngine
from text2image.utils.logger import get_logger
from text2image.utils.rate_limit import AsyncRateLimiter

GENERATION_PATH = "/api/v1/services/aigc/multimodal-generation/generation"
_FAILED_STATES = {"FAILED", "CANCELED", "UNKNOWN"}


class WanxEngine(BaseEngine):
    """调用阿里云百炼通义万相文生图接口。"""

    name = "wanx"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(settings.image_timeout))
        self._limiter = AsyncRateLimiter(settings.image_rpm)
        self._log = get_logger()

    # ---------- 请求构造 ----------

    @staticmethod
    def _endpoint(base_url: str) -> str:
        """由 base_url 拼出生成接口地址，兼容是否已带 /api/v1。"""
        base = base_url.rstrip("/")
        if base.endswith("/api/v1"):
            base = base[: -len("/api/v1")]
        return f"{base}{GENERATION_PATH}"

    def _build_payload(self, request: GenRequest) -> dict[str, Any]:
        params = request.params
        parameters: dict[str, Any] = {
            "negative_prompt": params.negative_prompt or "",
            "prompt_extend": params.prompt_extend,
            "watermark": params.watermark,
            "n": params.n,
            "size": params.size,
        }
        if params.seed is not None:
            parameters["seed"] = params.seed
        return {
            "model": params.model,
            "input": {
                "messages": [
                    {"role": "user", "content": [{"text": request.prompt}]},
                ]
            },
            "parameters": parameters,
        }

    # ---------- 主流程 ----------

    async def generate(self, request: GenRequest) -> list[ImageResult]:
        await self._limiter.acquire()
        started = time.perf_counter()
        url = self._endpoint(self._settings.dashscope_base_url)
        headers = {
            "Authorization": f"Bearer {self._settings.dashscope_api_key}",
            "Content-Type": "application/json",
        }
        self._log.info(
            "调用万相出图: model={} size={} n={}",
            request.params.model,
            request.params.size,
            request.params.n,
        )

        try:
            response = await self._client.post(
                url, json=self._build_payload(request), headers=headers
            )
        except httpx.HTTPError as exc:
            raise SubmissionError("请求万相接口失败", detail=str(exc)) from exc

        data = self._decode(response)
        results = self._parse(data, request, int((time.perf_counter() - started) * 1000))
        self._log.info("万相出图成功: {} 张", len(results))
        return results

    # ---------- 响应解析 ----------

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise SubmissionError("万相返回内容不是合法 JSON", detail=response.text[:500]) from exc
        if not isinstance(data, dict):
            raise SubmissionError("万相返回结构异常", detail=str(data)[:500])

        if response.is_error or data.get("code"):
            code = data.get("code") or response.status_code
            message = data.get("message") or "未知错误"
            raise SubmissionError(f"万相接口返回错误 {code}", detail=str(message))
        return data

    @staticmethod
    def _extract_urls(data: dict[str, Any]) -> list[str]:
        output = data.get("output") or {}
        status = output.get("task_status")
        if status in _FAILED_STATES:
            raise TaskFailedError(
                "万相任务失败",
                detail=str(output.get("message") or output.get("code") or status),
            )

        urls: list[str] = []
        for choice in output.get("choices") or []:
            message = choice.get("message") or {}
            for item in message.get("content") or []:
                image = item.get("image")
                if image:
                    urls.append(str(image))

        # 兼容早期 V1 的 results[].url 结构
        for item in output.get("results") or []:
            image = item.get("url") or item.get("image")
            if image:
                urls.append(str(image))

        if not urls:
            raise TaskFailedError("万相未返回任何图片", detail=str(data)[:500])
        return urls

    def _parse(
        self,
        data: dict[str, Any],
        request: GenRequest,
        elapsed_ms: int,
    ) -> list[ImageResult]:
        output = data.get("output") or {}
        task_id = str(output.get("task_id") or data.get("request_id") or "")
        return [
            ImageResult(
                task_id=task_id,
                image_url=image_url,
                prompt=request.prompt,
                params=request.params,
                seed=request.params.seed,
                elapsed_ms=elapsed_ms,
            )
            for image_url in self._extract_urls(data)
        ]

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
