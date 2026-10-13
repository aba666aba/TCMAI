# Copyright (C) 2026 aba666aba
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# 本文件是中医药智能问答系统 (TCMAI) 的一部分，按 AGPL-3.0-or-later 分发。

"""桌面服务层冒烟测试：query() 的返回契约。

需要模型权重（~127MB，不入 git）。缺失时 skip——裸 clone 未下载权重属预期状态。
"""

from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT / "tcm_recommender" / "best_model.pth"
DB_PATH = PROJECT / "data" / "tcm_knowledge.db"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (MODEL_PATH.exists() and DB_PATH.exists()),
        reason="缺少模型权重或知识库，请先运行 python scripts/download_weights.py",
    ),
]

REQUIRED_KEYS = {"success", "mode", "answer", "disclaimer", "has_warning", "knowledge"}


@pytest.fixture(scope="module")
def service():
    from src.desktop.service import TCMDesktopService

    return TCMDesktopService()


def test_query_returns_expected_contract(service):
    result = service.query("最近睡不着，白天没力气，头晕", ["失眠", "乏力", "头晕"])
    assert isinstance(result, dict)
    missing = REQUIRED_KEYS - set(result)
    assert not missing, f"返回缺少字段: {missing}"
    assert result["success"] is True
    assert result["mode"] == "ai"
    assert isinstance(result["answer"], str) and len(result["answer"]) > 50


def test_query_answer_contains_core_sections(service):
    result = service.query("失眠 乏力 头晕", ["失眠", "乏力", "头晕"])
    answer = result["answer"]
    for section in ("症状分析", "推荐方剂", "推荐药材"):
        assert section in answer, f"回答缺少「{section}」段落"


def test_query_rejects_invalid_input(service):
    with pytest.raises(ValueError):
        service.query("", [])
    with pytest.raises(ValueError):
        service.query("x" * 1001, [])
    with pytest.raises(ValueError):
        service.query("正常输入", ["症状"] * 21)


def test_query_desensitizes_contact_information(service):
    """手机号等个人信息不得进入下游检索与历史记录。"""
    result = service.query("失眠 乏力 我的电话是13800138000", ["失眠", "乏力"])
    assert "13800138000" not in result["answer"], "回答中泄漏了手机号"
