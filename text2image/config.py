"""集中式配置：从环境变量 / .env 读取。

环境变量（大小写不敏感）：
    DASHSCOPE_API_KEY      必填，阿里云百炼 API Key
    DASHSCOPE_BASE_URL     默认 https://dashscope.aliyuncs.com
    IMAGE_MODEL            默认 wan2.6-t2i
    IMAGE_SIZE             默认 1024*1024
    DEEPSEEK_API_KEY       可选，用于提示词改写
    T2I_OUTPUT_DIR         默认 ./output
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def read_windows_env(name: str) -> str | None:
    """从 Windows 注册表读取用户/系统环境变量。

    某些进程（IDE、沙箱、守护进程）启动时环境块被裁剪，看不到系统级变量。
    这里作为兜底：先查 HKCU\Environment，再查 HKLM 的系统环境变量。
    """
    if sys.platform != "win32":
        return None
    try:
        import winreg
    except ImportError:  # pragma: no cover - 非 Windows 平台
        return None

    locations = (
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
    )
    for hive, subkey in locations:
        try:
            with winreg.OpenKey(hive, subkey) as key:
                value, _ = winreg.QueryValueEx(key, name)
        except OSError:
            continue
        if value:
            return str(value)
    return None


def resolve_env(name: str) -> str | None:
    """先取进程环境变量，取不到再回退到 Windows 注册表。"""
    value = os.environ.get(name)
    if value:
        return value
    return read_windows_env(name)


class Settings(BaseSettings):
    """项目运行配置。字段名与同名环境变量自动匹配（大小写不敏感）。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # ---------- 阿里云百炼 / 通义万相 ----------
    dashscope_api_key: str = Field(default="", description="百炼 API Key")
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com",
        description="百炼服务地址（国内版）",
    )
    image_backend: str = Field(default="wanx", description="出图后端: wanx | mock")
    image_model: str = Field(default="wan2.6-t2i", description="出图模型")
    image_size: str = Field(default="1024*1024", description="默认尺寸")
    image_count: int = Field(default=1, ge=1, le=6, description="默认出图张数")
    image_timeout: float = Field(default=600.0, description="出图请求超时（秒）")
    image_poll_interval: float = Field(default=2.0, description="轮询间隔（秒）")
    image_poll_timeout: float = Field(default=300.0, description="轮询总超时（秒）")
    image_rpm: int = Field(default=5, description="本地限流：每分钟最多请求数")
    image_max_concurrency: int = Field(default=3, description="最大并发出图数")
    max_retries: int = Field(default=3, description="网络请求重试次数")

    # ---------- DeepSeek（提示词改写 / 文本质检） ----------
    deepseek_api_key: str | None = Field(default=None)
    deepseek_base_url: str = Field(default="https://api.deepseek.com")
    deepseek_model: str = Field(
        default="deepseek-flash",
        description="DeepSeek 模型；deepseek-flash 同时支持文本与图片输入",
    )
    deepseek_timeout: float = Field(default=60.0)
    llm_max_tokens: int = Field(default=4096, description="大模型单次最大输出 token 数")
    enable_prompt_rewrite: bool = Field(default=False, description="是否默认开启提示词改写")
    enable_image_review: bool = Field(default=False, description="是否默认开启图片质检")
    review_threshold: int = Field(default=60, ge=0, le=100, description="质检通过分数线")

    # ---------- 输出与日志 ----------
    output_dir: Path = Field(
        default=Path("output"),
        validation_alias=AliasChoices("T2I_OUTPUT_DIR", "OUTPUT_DIR"),
    )
    log_dir: Path = Field(
        default=Path("logs"),
        validation_alias=AliasChoices("T2I_LOG_DIR", "LOG_DIR"),
    )
    log_level: str = Field(default="INFO")

    def ensure_dirs(self) -> None:
        """确保输出与日志目录存在。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


_FALLBACK_ENV_NAMES = ("DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回全局唯一的配置实例（带缓存）。

    构造前会把缺失的关键环境变量从注册表补回 os.environ，保证
    pydantic-settings 能读到系统级配置。
    """
    for name in _FALLBACK_ENV_NAMES:
        if not os.environ.get(name):
            value = read_windows_env(name)
            if value:
                os.environ[name] = value
    return Settings()
