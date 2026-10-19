"""命令行入口。

用法示例：
    uv run python -m text2image.cli gen "一只在太空的橘猫" --n 2 --review
    uv run python -m text2image.cli gen "赛博朋克城市" --backend mock --no-rewrite
    uv run python -m text2image.cli batch prompts.txt
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from text2image.config import Settings, get_settings
from text2image.core.schemas import GenerationOutcome, GenParams, GenRequest
from text2image.services.generator import build_service
from text2image.utils.logger import setup_logging

app = typer.Typer(add_completion=False, help="通义万相文生图命令行工具")
console = Console()


def _resolve_settings(
    backend: str | None,
    out: str | None,
    model: str | None,
    size: str | None,
    count: int | None,
) -> Settings:
    """在全局配置基础上应用命令行覆盖。"""
    updates: dict[str, object] = {}
    if backend:
        updates["image_backend"] = backend
    if out:
        updates["output_dir"] = Path(out)
    if model:
        updates["image_model"] = model
    if size:
        updates["image_size"] = size
    if count is not None:
        updates["image_count"] = count

    settings = get_settings().model_copy(update=updates)
    settings.ensure_dirs()
    setup_logging(settings.log_dir, settings.log_level)
    return settings


def _build_request(settings: Settings, prompt: str, style: str | None) -> GenRequest:
    return GenRequest(
        prompt=prompt,
        style=style,
        params=GenParams(
            model=settings.image_model,
            size=settings.image_size,
            n=settings.image_count,
        ),
    )


def _render(outcome: GenerationOutcome) -> None:
    if outcome.rewrite is not None:
        console.print("[bold]改写后提示词：[/bold]")
        console.print(f"  {outcome.final_prompt}")
        console.print()

    table = Table(title="生成结果")
    table.add_column("#", justify="right")
    table.add_column("文件")
    table.add_column("大小", justify="right")
    table.add_column("质检", justify="center")
    for index, item in enumerate(outcome.results, start=1):
        size_text = "-"
        if item.local_path and item.local_path.exists():
            size_text = f"{item.local_path.stat().st_size / 1024:.0f} KB"
        review_text = "-"
        if item.review is not None:
            verdict = "PASS" if item.review.passed else "FAIL"
            review_text = f"{item.review.score} {verdict}"
        table.add_row(str(index), str(item.local_path), size_text, review_text)
    console.print(table)

    for item in outcome.results:
        if item.review and item.review.issues:
            label = item.local_path.name if item.local_path else "图片"
            console.print(f"[yellow]{label} 的问题：[/yellow]")
            for issue in item.review.issues:
                console.print(f"  - {issue}")
            if item.review.suggestion:
                console.print(f"  [dim]建议：{item.review.suggestion}[/dim]")


async def _run(
    settings: Settings,
    requests: list[GenRequest],
    rewrite: bool | None,
    review: bool | None,
) -> list[GenerationOutcome]:
    service = build_service(settings)
    try:
        return await service.generate_batch(requests, rewrite=rewrite, review=review)
    finally:
        await service.aclose()


def _report(outcomes: list[GenerationOutcome], as_json: bool) -> int:
    if as_json:
        console.print_json(data=[item.model_dump(mode="json") for item in outcomes])
    else:
        for outcome in outcomes:
            _render(outcome)
    all_passed = all(
        result.review is None or result.review.passed
        for outcome in outcomes
        for result in outcome.results
    )
    return 0 if all_passed else 2


@app.command()
def gen(
    prompt: Annotated[str, typer.Argument(help="提示词，建议用引号包裹")],
    size: Annotated[str | None, typer.Option("--size", "-s", help="尺寸，如 1024*1024")] = None,
    count: Annotated[int | None, typer.Option("--n", min=1, max=6, help="生成张数")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="出图模型")] = None,
    style: Annotated[str | None, typer.Option("--style", help="风格提示")] = None,
    backend: Annotated[str | None, typer.Option("--backend", help="wanx | mock")] = None,
    out: Annotated[str | None, typer.Option("--out", "-o", help="输出目录")] = None,
    rewrite: Annotated[
        bool | None, typer.Option("--rewrite/--no-rewrite", help="是否用 DeepSeek 改写提示词")
    ] = None,
    review: Annotated[
        bool | None, typer.Option("--review/--no-review", help="是否用 DeepSeek 做图片质检")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="以 JSON 输出结果")] = False,
) -> None:
    """生成一张或多张图片。"""
    settings = _resolve_settings(backend, out, model, size, count)
    requests = [_build_request(settings, prompt, style)]
    outcomes = asyncio.run(_run(settings, requests, rewrite, review))
    raise typer.Exit(_report(outcomes, as_json))


@app.command()
def batch(
    file: Annotated[Path, typer.Argument(help="提示词文件，每行一条，# 开头为注释")],
    size: Annotated[str | None, typer.Option("--size", "-s")] = None,
    count: Annotated[int | None, typer.Option("--n", min=1, max=6)] = None,
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    style: Annotated[str | None, typer.Option("--style")] = None,
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    out: Annotated[str | None, typer.Option("--out", "-o")] = None,
    rewrite: Annotated[bool | None, typer.Option("--rewrite/--no-rewrite")] = None,
    review: Annotated[bool | None, typer.Option("--review/--no-review")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """从文件批量读取提示词并逐条生成。"""
    if not file.exists():
        console.print(f"[red]文件不存在：{file}[/red]")
        raise typer.Exit(1)

    prompts = [
        line.strip()
        for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not prompts:
        console.print("[red]文件中没有有效提示词[/red]")
        raise typer.Exit(1)

    settings = _resolve_settings(backend, out, model, size, count)
    requests = [_build_request(settings, item, style) for item in prompts]
    console.print(f"共 {len(requests)} 条提示词，开始逐条生成……")
    outcomes = asyncio.run(_run(settings, requests, rewrite, review))
    raise typer.Exit(_report(outcomes, as_json))


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="监听地址")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="监听端口")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="代码热重载（开发用）")] = False,
    backend: Annotated[str | None, typer.Option("--backend", help="wanx | mock")] = None,
) -> None:
    """启动 FastAPI 服务。"""
    if backend:
        os.environ["IMAGE_BACKEND"] = backend
    import uvicorn

    console.print(f"[green]启动服务：http://{host}:{port}/docs[/green]")
    uvicorn.run("text2image.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
