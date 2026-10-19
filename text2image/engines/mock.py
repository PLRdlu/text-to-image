"""离线模拟引擎：不联网，用 Pillow 生成占位图。

用于本地开发、单元测试和余额不足时验证链路。
"""

from __future__ import annotations

import time

from PIL import Image, ImageDraw

from text2image.config import Settings
from text2image.core.schemas import GenRequest, ImageResult
from text2image.engines.base import BaseEngine
from text2image.storage.paths import build_image_path


class MockEngine(BaseEngine):
    """生成带序号的占位图，接口与真实引擎完全一致。"""

    name = "mock"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def generate(self, request: GenRequest) -> list[ImageResult]:
        started = time.perf_counter()
        results: list[ImageResult] = []
        for index in range(request.params.n):
            dest = build_image_path(
                self._settings.output_dir,
                prompt=request.prompt,
                model="mock",
                index=index,
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            self._render(dest, index)
            results.append(
                ImageResult(
                    task_id=f"mock-{int(time.time())}",
                    image_url=dest.as_uri(),
                    local_path=dest,
                    prompt=request.prompt,
                    params=request.params,
                    seed=request.params.seed,
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                )
            )
        return results

    @staticmethod
    def _render(dest, index: int) -> None:
        image = Image.new("RGB", (512, 512), (28, 38, 58))
        draw = ImageDraw.Draw(image)
        draw.rectangle((16, 16, 496, 496), outline=(120, 160, 220), width=3)
        draw.text((40, 220), f"MOCK IMAGE #{index}", fill=(235, 240, 250))
        draw.text((40, 250), "no API call was made", fill=(160, 190, 220))
        image.save(dest)
