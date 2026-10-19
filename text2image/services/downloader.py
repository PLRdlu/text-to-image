"""异步下载万相生成的图片。

万相返回的图像 URL 仅保留 24 小时，必须尽快转存到本地。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiofiles
import httpx

from text2image.core.errors import DownloadError
from text2image.utils.logger import get_logger


class ImageDownloader:
    """把远程图片下载到本地路径，带简单重试。"""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        retries: int = 3,
        timeout: float = 120.0,
    ) -> None:
        self._client = client
        self._retries = max(1, retries)
        self._timeout = timeout
        self._log = get_logger()

    async def download(self, url: str, dest: Path) -> Path:
        """下载 url 到 dest，返回 dest。失败抛 DownloadError。"""
        dest.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None

        for attempt in range(1, self._retries + 1):
            try:
                response = await self._client.get(url, timeout=self._timeout, follow_redirects=True)
                response.raise_for_status()
                content = response.content
                if not content:
                    raise DownloadError("下载到的图片内容为空", detail=url)
                async with aiofiles.open(dest, "wb") as handle:
                    await handle.write(content)
                self._log.info("图片已保存: {} ({} bytes)", dest, len(content))
                return dest
            except (httpx.HTTPError, OSError, DownloadError) as exc:
                last_error = exc
                if attempt < self._retries:
                    await asyncio.sleep(min(2**attempt * 0.5, 8.0))

        raise DownloadError(
            "下载图片失败",
            detail=f"{url} -> {last_error}",
        ) from last_error
