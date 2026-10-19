"""pytest 公共夹具。"""

from __future__ import annotations

from pathlib import Path

import pytest

from text2image.config import Settings


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """构造一份完全隔离、不触碰真实网络的配置。"""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-dashscope")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    monkeypatch.setenv("IMAGE_BACKEND", "mock")
    monkeypatch.setenv("IMAGE_RPM", "6000")
    monkeypatch.setenv("IMAGE_SIZE", "1024*1024")
    monkeypatch.setenv("T2I_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("T2I_LOG_DIR", str(tmp_path / "logs"))
    return Settings(_env_file=None)  # type: ignore[call-arg]
