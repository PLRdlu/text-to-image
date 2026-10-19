"""FastAPI 接口测试（mock 后端，不联网、不花钱）。"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from text2image.config import get_settings


def _prepare_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("IMAGE_BACKEND", "mock")
    monkeypatch.setenv("T2I_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("T2I_LOG_DIR", str(tmp_path / "logs"))
    get_settings.cache_clear()


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    _prepare_env(tmp_path, monkeypatch)
    from text2image.api.app import create_app

    return TestClient(create_app())


def test_healthz_reports_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(tmp_path, monkeypatch) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["backend"] == "mock"
    assert body["api_key_configured"] is True


def test_generation_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(tmp_path, monkeypatch) as client:
        created = client.post("/api/v1/images", json={"prompt": "一只猫", "n": 2})
        assert created.status_code == 202
        task_id = created.json()["task_id"]

        detail: dict = {}
        for _ in range(100):
            detail = client.get(f"/api/v1/tasks/{task_id}").json()
            if detail["status"] in ("SUCCEEDED", "FAILED"):
                break
            time.sleep(0.1)

        assert detail["status"] == "SUCCEEDED"
        assert len(detail["images"]) == 2

        image = client.get(detail["images"][0]["url"])
        assert image.status_code == 200
        assert image.content

        listing = client.get("/api/v1/tasks?limit=5")
        assert listing.status_code == 200
        assert len(listing.json()) == 1


def test_not_found_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(tmp_path, monkeypatch) as client:
        assert client.get("/api/v1/tasks/nope").status_code == 404
        assert client.get("/api/v1/images/nope/0").status_code == 404


def test_invalid_payload_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(tmp_path, monkeypatch) as client:
        assert client.post("/api/v1/images", json={"prompt": "   "}).status_code == 422
        bad_size = {"prompt": "x", "size": "100x100"}
        assert client.post("/api/v1/images", json=bad_size).status_code == 422
        bad_n = {"prompt": "x", "n": 99}
        assert client.post("/api/v1/images", json=bad_n).status_code == 422
