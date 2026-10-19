"""生成图片的落盘路径规则。"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path

_UNSAFE = re.compile(r"[^0-9A-Za-z_-]+")


def safe_slug(text: str, max_len: int = 32) -> str:
    """把任意文本压成文件名安全片段。"""
    slug = _UNSAFE.sub("-", text).strip("-")
    return slug[:max_len] or "image"


def build_image_path(
    output_dir: Path,
    *,
    prompt: str,
    model: str,
    index: int,
    suffix: str = ".png",
) -> Path:
    """按 输出目录/日期/时间_模型_提示词片段_序号_随机 生成文件路径。"""
    base = output_dir.expanduser().resolve()
    now = datetime.now()
    day = now.strftime("%Y%m%d")
    stamp = now.strftime("%H%M%S")
    token = uuid.uuid4().hex[:8]
    filename = f"{stamp}_{safe_slug(model)}_{safe_slug(prompt)}_{index}_{token}{suffix}"
    return base / day / filename
