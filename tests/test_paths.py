"""落盘路径规则测试。"""

from __future__ import annotations

from pathlib import Path

from text2image.storage.paths import build_image_path, safe_slug


def test_safe_slug_strips_unsafe_chars() -> None:
    assert safe_slug("a cat in space!") == "a-cat-in-space"
    assert safe_slug("中文提示词") == "image"
    assert safe_slug("") == "image"


def test_build_image_path_is_absolute_and_unique(tmp_path: Path) -> None:
    first = build_image_path(tmp_path, prompt="a cat", model="wan2.6-t2i", index=0)
    second = build_image_path(tmp_path, prompt="a cat", model="wan2.6-t2i", index=1)
    assert first.is_absolute()
    assert first.suffix == ".png"
    assert first.parent == second.parent
    assert first != second
    assert first.parent.parent == tmp_path.resolve()
