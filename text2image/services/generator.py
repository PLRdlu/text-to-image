"""编排层：把改写、出图、下载、质检、落库串成一次完整生成。"""

from __future__ import annotations

import httpx

from text2image.config import Settings
from text2image.core.errors import Text2ImageError
from text2image.core.schemas import GenerationOutcome, GenRequest, ImageResult
from text2image.engines import create_engine
from text2image.engines.base import BaseEngine
from text2image.llm import create_llm_client
from text2image.llm.client import LLMClient
from text2image.llm.image_reviewer import ImageReviewer
from text2image.llm.prompt_rewriter import PromptRewriter
from text2image.services.downloader import ImageDownloader
from text2image.storage.paths import build_image_path
from text2image.storage.repository import JsonlRepository
from text2image.utils.logger import get_logger


class GeneratorService:
    """文生图主流程编排器。"""

    def __init__(
        self,
        settings: Settings,
        *,
        engine: BaseEngine,
        downloader: ImageDownloader,
        repository: JsonlRepository | None = None,
        rewriter: PromptRewriter | None = None,
        reviewer: ImageReviewer | None = None,
        http_client: httpx.AsyncClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._settings = settings
        self._engine = engine
        self._downloader = downloader
        self._repository = repository
        self._rewriter = rewriter
        self._reviewer = reviewer
        self._http_client = http_client
        self._llm_client = llm_client
        self._log = get_logger()

    # ---------- 对外接口 ----------

    async def generate(
        self,
        request: GenRequest,
        *,
        rewrite: bool | None = None,
        review: bool | None = None,
    ) -> GenerationOutcome:
        """执行一次完整生成。rewrite/review 为 None 时取配置默认值。"""
        settings = self._settings
        do_rewrite = settings.enable_prompt_rewrite if rewrite is None else rewrite
        do_review = settings.enable_image_review if review is None else review

        outcome = GenerationOutcome(user_prompt=request.prompt, final_prompt=request.prompt)
        effective = request

        # 1) 提示词改写（失败则回退到原始提示词，不阻断出图）
        if do_rewrite and self._rewriter is not None:
            try:
                rewritten = await self._rewriter.rewrite(request.prompt, style=request.style)
                outcome.rewrite = rewritten
                outcome.final_prompt = rewritten.rewritten
                params = request.params.model_copy(
                    update={
                        "negative_prompt": rewritten.negative_prompt
                        or request.params.negative_prompt
                    }
                )
                effective = request.model_copy(
                    update={"prompt": rewritten.rewritten, "params": params}
                )
            except Text2ImageError as exc:
                self._log.warning("提示词改写失败，回退原始提示词: {}", exc)

        # 2) 出图（引擎只负责拿到图片 URL）
        raw_results = await self._engine.generate(effective)

        # 3) 下载落盘（引擎若已给出本地文件则跳过下载，如 mock 引擎）
        results: list[ImageResult] = []
        for index, item in enumerate(raw_results):
            if item.local_path is not None and item.local_path.exists():
                results.append(item.model_copy(update={"prompt": effective.prompt}))
                continue
            dest = build_image_path(
                settings.output_dir,
                prompt=effective.prompt,
                model=effective.params.model,
                index=index,
            )
            await self._downloader.download(item.image_url, dest)
            results.append(item.model_copy(update={"local_path": dest, "prompt": effective.prompt}))
        outcome.results = results

        # 4) 图片质检（逐张进行，失败不阻断）
        if do_review and self._reviewer is not None:
            for item in results:
                if item.local_path is None:
                    continue
                try:
                    item.review = await self._reviewer.review(item.local_path, effective.prompt)
                    outcome.reviews.append(item.review)
                except Text2ImageError as exc:
                    self._log.warning("图片质检失败: {}", exc)

        # 5) 元数据落库
        if self._repository is not None:
            for item in results:
                self._repository.append(item)

        return outcome

    async def generate_batch(
        self,
        requests: list[GenRequest],
        *,
        rewrite: bool | None = None,
        review: bool | None = None,
    ) -> list[GenerationOutcome]:
        """顺序批量生成，避免触发服务端限流。"""
        outcomes: list[GenerationOutcome] = []
        for request in requests:
            outcomes.append(await self.generate(request, rewrite=rewrite, review=review))
        return outcomes

    async def aclose(self) -> None:
        """释放引擎、HTTP 连接与 LLM 客户端。"""
        await self._engine.aclose()
        if self._http_client is not None:
            await self._http_client.aclose()
        if self._llm_client is not None:
            await self._llm_client.aclose()


def build_service(settings: Settings) -> GeneratorService:
    """按配置装配出一个可直接使用的 GeneratorService。"""
    engine = create_engine(settings)
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(settings.image_timeout))
    downloader = ImageDownloader(http_client, retries=settings.max_retries)
    repository = JsonlRepository(settings.output_dir / "metadata.jsonl")

    llm_client: LLMClient | None = None
    rewriter: PromptRewriter | None = None
    reviewer: ImageReviewer | None = None
    if settings.deepseek_api_key:
        llm_client = create_llm_client(settings)
        rewriter = PromptRewriter(llm_client)
        reviewer = ImageReviewer(llm_client, threshold=settings.review_threshold)

    return GeneratorService(
        settings,
        engine=engine,
        downloader=downloader,
        repository=repository,
        rewriter=rewriter,
        reviewer=reviewer,
        http_client=http_client,
        llm_client=llm_client,
    )
