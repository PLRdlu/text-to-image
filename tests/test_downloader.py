"""图片下载器测试（打桩）。"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from text2image.core.errors import DownloadError
from text2image.services.downloader import ImageDownloader

URL = "https://example.com/a.png"


@respx.mock
async def test_download_writes_file(tmp_path: Path) -> None:
    respx.get(URL).mock(return_value=httpx.Response(200, content=b"PNGDATA"))
    async with httpx.AsyncClient() as client:
        downloader = ImageDownloader(client, retries=1)
        dest = tmp_path / "nested" / "a.png"
        result = await downloader.download(URL, dest)
    assert result == dest
    assert dest.read_bytes() == b"PNGDATA"


@respx.mock
async def test_download_raises_after_retries(tmp_path: Path) -> None:
    respx.get(URL).mock(return_value=httpx.Response(500))
    async with httpx.AsyncClient() as client:
        downloader = ImageDownloader(client, retries=1)
        with pytest.raises(DownloadError):
            await downloader.download(URL, tmp_path / "a.png")


@respx.mock
async def test_download_rejects_empty_body(tmp_path: Path) -> None:
    respx.get(URL).mock(return_value=httpx.Response(200, content=b""))
    async with httpx.AsyncClient() as client:
        downloader = ImageDownloader(client, retries=1)
        with pytest.raises(DownloadError):
            await downloader.download(URL, tmp_path / "a.png")
