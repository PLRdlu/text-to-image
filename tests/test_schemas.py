"""数据模型与尺寸规范化测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from text2image.core.schemas import (
    GenParams,
    GenRequest,
    ImageReview,
    TaskStatus,
    normalize_size,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1024*1024", "1024*1024"),
        ("1024x1024", "1024*1024"),
        ("1024X1024", "1024*1024"),
        (" 768 * 1152 ", "768*1152"),
        ("720x1280", "720*1280"),
    ],
)
def test_normalize_size_ok(raw: str, expected: str) -> None:
    assert normalize_size(raw) == expected


@pytest.mark.parametrize("raw", ["100x100", "4096*4096", "2048*100", "abc", ""])
def test_normalize_size_rejects(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_size(raw)


def test_gen_params_defaults() -> None:
    params = GenParams()
    assert params.model == "wan2.6-t2i"
    assert params.size == "1024*1024"
    assert params.n == 1
    assert params.prompt_extend is True


@pytest.mark.parametrize("bad_n", [0, 7, -1])
def test_gen_params_n_bounds(bad_n: int) -> None:
    with pytest.raises(ValidationError):
        GenParams(n=bad_n)


def test_gen_params_seed_bounds() -> None:
    assert GenParams(seed=0).seed == 0
    with pytest.raises(ValidationError):
        GenParams(seed=-1)
    with pytest.raises(ValidationError):
        GenParams(seed=2_147_483_648)


def test_gen_request_strips_prompt() -> None:
    request = GenRequest(prompt="  一只猫  ")
    assert request.prompt == "一只猫"


def test_gen_request_rejects_blank_prompt() -> None:
    with pytest.raises(ValidationError):
        GenRequest(prompt="   ")


def test_task_status_finished() -> None:
    assert TaskStatus(task_id="t", status="SUCCEEDED").finished is True
    assert TaskStatus(task_id="t", status="RUNNING").finished is False


def test_image_review_score_bounds() -> None:
    assert ImageReview(score=100, passed=True).score == 100
    with pytest.raises(ValidationError):
        ImageReview(score=101)
