"""本地限流器：控制出图请求的最小间隔。"""

from __future__ import annotations

import asyncio
import time


class AsyncRateLimiter:
    """按"每分钟请求数"平滑放行的异步限流器。

    万相 RPM 较低（如 qwen-image-3.0-pro 仅 5），这里把请求在时间上摊开，
    避免触发服务端 429。
    """

    def __init__(self, per_minute: int) -> None:
        self._interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def acquire(self) -> None:
        if self._interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self._interval
