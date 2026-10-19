"""万相引擎的请求构造与响应解析测试（全程打桩，不发真实请求）。"""

from __future__ import annotations

import httpx
import pytest
import respx

from text2image.config import Settings
from text2image.core.errors import SubmissionError, TaskFailedError
from text2image.core.schemas import GenParams, GenRequest
from text2image.engines.wanx import WanxEngine

BASE = "https://dashscope.aliyuncs.com"
URL = f"{BASE}/api/v1/services/aigc/multimodal-generation/generation"

SUCCESS_BODY = {
    "output": {
        "task_id": "task-1",
        "task_status": "SUCCEEDED",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "image", "image": "https://example.com/a.png"},
                        {"type": "image", "image": "https://example.com/b.png"},
                    ],
                },
            }
        ],
    },
    "usage": {"image_count": 2},
    "request_id": "req-1",
}


def test_endpoint_building() -> None:
    assert WanxEngine._endpoint(BASE) == URL
    assert WanxEngine._endpoint(f"{BASE}/") == URL
    assert WanxEngine._endpoint(f"{BASE}/api/v1") == URL


async def test_build_payload(settings: Settings) -> None:
    engine = WanxEngine(settings)
    try:
        payload = engine._build_payload(
            GenRequest(
                prompt="一只猫",
                params=GenParams(model="wan2.6-t2i", size="1024x1024", n=2, negative_prompt="模糊"),
            )
        )
    finally:
        await engine.aclose()

    assert payload["model"] == "wan2.6-t2i"
    assert payload["input"]["messages"][0]["content"][0]["text"] == "一只猫"
    assert payload["parameters"]["size"] == "1024*1024"
    assert payload["parameters"]["n"] == 2
    assert payload["parameters"]["negative_prompt"] == "模糊"
    assert "seed" not in payload["parameters"]


@respx.mock
async def test_generate_success(settings: Settings) -> None:
    engine = WanxEngine(settings)
    try:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=SUCCESS_BODY))
        results = await engine.generate(GenRequest(prompt="cat", params=GenParams(n=2)))
        assert route.called
        assert [item.image_url for item in results] == [
            "https://example.com/a.png",
            "https://example.com/b.png",
        ]
        assert results[0].task_id == "task-1"
        request = route.calls[0].request
        assert request.headers["authorization"] == "Bearer sk-test-dashscope"
    finally:
        await engine.aclose()


@respx.mock
async def test_generate_http_error(settings: Settings) -> None:
    engine = WanxEngine(settings)
    try:
        body = {"code": "InvalidApiKey", "message": "No API-key provided."}
        respx.post(URL).mock(return_value=httpx.Response(401, json=body))
        with pytest.raises(SubmissionError) as excinfo:
            await engine.generate(GenRequest(prompt="cat"))
        assert "InvalidApiKey" in str(excinfo.value)
    finally:
        await engine.aclose()


@respx.mock
async def test_generate_task_failed(settings: Settings) -> None:
    engine = WanxEngine(settings)
    try:
        body = {"output": {"task_status": "FAILED", "message": "boom"}}
        respx.post(URL).mock(return_value=httpx.Response(200, json=body))
        with pytest.raises(TaskFailedError):
            await engine.generate(GenRequest(prompt="cat"))
    finally:
        await engine.aclose()


@respx.mock
async def test_generate_no_images(settings: Settings) -> None:
    engine = WanxEngine(settings)
    try:
        body = {"output": {"task_status": "SUCCEEDED", "choices": []}}
        respx.post(URL).mock(return_value=httpx.Response(200, json=body))
        with pytest.raises(TaskFailedError):
            await engine.generate(GenRequest(prompt="cat"))
    finally:
        await engine.aclose()


@respx.mock
async def test_generate_invalid_json(settings: Settings) -> None:
    engine = WanxEngine(settings)
    try:
        respx.post(URL).mock(return_value=httpx.Response(200, text="<html>oops</html>"))
        with pytest.raises(SubmissionError):
            await engine.generate(GenRequest(prompt="cat"))
    finally:
        await engine.aclose()
