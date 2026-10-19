"""生成元数据的持久化（JSONL 追加写）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from text2image.core.schemas import ImageResult
from text2image.utils.logger import get_logger


class JsonlRepository:
    """把每次生成的结果以一行 JSON 的形式追加到文件。

    选 JSONL 而非数据库，是为了零依赖、易追加、方便直接用文本工具查看。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._log = get_logger()

    def append(self, record: ImageResult | BaseModel | dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as exc:  # 元数据写入失败不应影响出图主流程
            self._log.warning("写入元数据失败: {}", exc)
