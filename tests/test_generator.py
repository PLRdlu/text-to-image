"""编排层测试：用假引擎/假下载器验证五步流程。"""

from __future__ import annotations

from pathlib import Path

from text2image.config import Settings
from text2image.core.schemas import (
    GenParams,
    GenRequest,
    ImageResult,
    ImageReview,
    PromptRewrite,
)
from text2image.engines.base import BaseEngine
from text2image.services.generator import GeneratorService
from text2image.storage.repository import JsonlRepository


class FakeEngine(BaseEngine):
    name = "fake"

    def __init__(self, results: list[ImageResult]) -> None:
        self._results = results
        self.calls: list[GenRequest] = []

    async def generate(self, request: GenRequest) -> list[ImageResult]:
        self.calls.append(request)
        return list(self._results)


class FakeDownloader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    async def download(self, url: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"data")
        self.calls.append((url, dest))
        return dest


class FakeRewriter:
    def __init__(self, rewritten: str = "rewritten-en", negative: str | None = "bad") -> None:
        self.rewritten = rewritten
        self.negative = negative

    async def rewrite(self, prompt: str, style: str | None = None) -> PromptRewrite:
        return PromptRewrite(
            original=prompt, rewritten=self.rewritten, negative_prompt=self.negative
        )


class FakeReviewer:
    async def review(self, image_path: Path, prompt: str) -> ImageReview:
        return ImageReview(score=90, passed=True, issues=["小问题"], suggestion="再亮一点")


def _build(
    settings: Settings,
    results: list[ImageResult],
    *,
    rewriter: FakeRewriter | None = None,
    reviewer: FakeReviewer | None = None,
    repository: JsonlRepository | None = None,
) -> tuple[GeneratorService, FakeEngine, FakeDownloader]:
    engine = FakeEngine(results)
    downloader = FakeDownloader()
    service = GeneratorService(
        settings,
        engine=engine,
        downloader=downloader,  # type: ignore[arg-type]
        repository=repository,
        rewriter=rewriter,  # type: ignore[arg-type]
        reviewer=reviewer,  # type: ignore[arg-type]
    )
    return service, engine, downloader


async def test_generate_downloads_results(settings: Settings) -> None:
    results = [ImageResult(task_id="t", image_url="https://example.com/a.png")]
    service, engine, downloader = _build(settings, results)
    outcome = await service.generate(GenRequest(prompt="cat", params=GenParams(n=1)))
    await service.aclose()

    assert len(outcome.results) == 1
    local_path = outcome.results[0].local_path
    assert local_path is not None and local_path.exists()
    assert downloader.calls and downloader.calls[0][0] == "https://example.com/a.png"
    assert engine.calls[0].prompt == "cat"


async def test_generate_skips_download_for_local_results(
    settings: Settings, tmp_path: Path
) -> None:
    existing = tmp_path / "already.png"
    existing.write_bytes(b"png")
    results = [
        ImageResult(task_id="t", image_url="file:///x.png", local_path=existing),
    ]
    service, _, downloader = _build(settings, results)
    outcome = await service.generate(GenRequest(prompt="cat"))
    await service.aclose()

    assert outcome.results[0].local_path == existing
    assert downloader.calls == []


async def test_rewrite_applied(settings: Settings) -> None:
    results = [ImageResult(task_id="t", image_url="https://example.com/a.png")]
    service, engine, _ = _build(settings, results, rewriter=FakeRewriter())
    outcome = await service.generate(GenRequest(prompt="一只猫"), rewrite=True)
    await service.aclose()

    assert outcome.final_prompt == "rewritten-en"
    assert outcome.rewrite is not None
    assert engine.calls[0].prompt == "rewritten-en"
    assert engine.calls[0].params.negative_prompt == "bad"


async def test_review_attached(settings: Settings) -> None:
    results = [ImageResult(task_id="t", image_url="https://example.com/a.png")]
    service, _, _ = _build(settings, results, reviewer=FakeReviewer())
    outcome = await service.generate(GenRequest(prompt="cat"), review=True)
    await service.aclose()

    assert outcome.reviews and outcome.reviews[0].score == 90
    assert outcome.results[0].review is not None
    assert outcome.results[0].review.issues == ["小问题"]


async def test_repository_records_each_image(settings: Settings, tmp_path: Path) -> None:
    results = [ImageResult(task_id="t", image_url="https://example.com/a.png")]
    repo = JsonlRepository(tmp_path / "metadata.jsonl")
    service, _, _ = _build(settings, results, repository=repo)
    await service.generate(GenRequest(prompt="cat"))
    await service.aclose()

    lines = repo.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert "local_path" in lines[0]
