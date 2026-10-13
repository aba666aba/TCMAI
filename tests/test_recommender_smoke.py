# Copyright (C) 2026 aba666aba
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# 本文件是中医药智能问答系统 (TCMAI) 的一部分，按 AGPL-3.0-or-later 分发。

"""推荐器冒烟测试：确认真实权重能被加载并产出结构正确的推荐。

需要 tcm_recommender/best_model.pth（~127MB，不入 git）。
缺失时整体 skip，而不是让测试失败——裸 clone 未执行下载脚本属预期状态。
"""

from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT / "tcm_recommender" / "best_model.pth"
DB_PATH = PROJECT / "data" / "tcm_knowledge.db"

pytestmark = pytest.mark.slow

requires_weights = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="缺少 best_model.pth，请先运行 python scripts/download_weights.py",
)
requires_db = pytest.mark.skipif(
    not DB_PATH.exists(),
    reason="缺少 data/tcm_knowledge.db",
)


@pytest.fixture(scope="module")
def recommender():
    torch = pytest.importorskip("torch")
    assert torch is not None
    from tcm_recommender.recommend import TCMRecommender

    return TCMRecommender(
        model_path=str(MODEL_PATH), db_path=str(DB_PATH), device="cpu"
    )


@requires_weights
@requires_db
def test_symptom_parsing_uses_standard_vocabulary(recommender):
    ids, valid, unknown = recommender.parse_symptoms("失眠 乏力 头晕")
    assert ids, "标准症状未被解析出来"
    assert len(ids) == len(valid)
    from tcm_recommender.extended_config import EXTENDED_SYMPTOM_LIST

    vocab = set(EXTENDED_SYMPTOM_LIST)
    assert set(valid) <= vocab, f"解析出词表外症状: {set(valid) - vocab}"


@requires_weights
@requires_db
def test_recommend_returns_documented_contract(recommender):
    """recommend() 的返回契约：症状 / 证型 / 药材 / 方剂 / 被拦截的毒性药材。

    注意：`safety` 不在本方法返回值内——安全检查是独立方法 `check_safety()`，
    由上层（service.py）拼装。此前的测试误以为它内联返回，已修正。
    """
    ids, _, _ = recommender.parse_symptoms("失眠 乏力 头晕")
    result = recommender.recommend(ids, top_k_herb=10, top_k_formula=5)

    expected = {"symptoms", "syndrome", "herbs", "formulas", "blocked_toxic_herbs"}
    assert expected <= set(result), f"返回缺少字段: {expected - set(result)}"
    assert isinstance(result["syndrome"], dict)
    assert {"id", "name", "confidence"} <= set(result["syndrome"])


@requires_weights
@requires_db
def test_check_safety_is_callable_and_structured(recommender):
    """安全检查作为独立能力必须可用，且返回结构化警告。"""
    ids, _, _ = recommender.parse_symptoms("失眠 乏力")
    result = recommender.recommend(ids, top_k_herb=10, top_k_formula=5)
    names = [h["name"] for h in result["herbs"]]
    assert names, "没有药材可做安全检查"

    safety = recommender.check_safety(names[:3])
    assert isinstance(safety, dict), "check_safety 应返回 dict"
    assert "warnings" in safety, "check_safety 缺少 warnings 字段"
    assert isinstance(safety["warnings"], list)


@requires_weights
@requires_db
def test_toxic_herbs_are_blocked(recommender):
    """毒性药材必须被拦截，不得出现在推荐结果中。

    推荐器使用的是 `extended_config.is_toxic_herb()`（EXTENDED_TOXIC_HERBS，52 味），
    与 `src/core/tcm_config.TOXIC_HERBS`（7 味）不是同一份清单，此处按实际调用方校验。
    """
    from tcm_recommender.extended_config import is_toxic_herb

    ids, _, _ = recommender.parse_symptoms("失眠 乏力 头晕")
    result = recommender.recommend(ids, top_k_herb=20, top_k_formula=5)
    names = [h["name"] for h in result["herbs"]]
    leaked = [n for n in names if is_toxic_herb(n)]
    assert not leaked, f"推荐结果中出现毒性药材: {leaked}"
    assert isinstance(result["blocked_toxic_herbs"], list)


@requires_weights
@requires_db
def test_recommend_is_deterministic(recommender):
    """同一输入的推荐结果必须可复现（评测指标依赖这一点）。"""
    ids, _, _ = recommender.parse_symptoms("恶寒 发热 咳嗽 鼻塞 头痛 无汗")
    first = [h["name"] for h in recommender.recommend(ids, 10, 5)["herbs"]]
    second = [h["name"] for h in recommender.recommend(ids, 10, 5)["herbs"]]
    assert first == second, "相同输入的推荐顺序不一致"


@requires_weights
@requires_db
def test_herb_and_formula_entries_are_well_formed(recommender):
    ids, _, _ = recommender.parse_symptoms("失眠 乏力 头晕")
    result = recommender.recommend(ids, top_k_herb=10, top_k_formula=5)

    herbs = result.get("herbs", [])
    formulas = result.get("formulas", [])
    assert herbs, "未返回任何药材推荐"
    assert formulas, "未返回任何方剂推荐"

    for h in herbs:
        assert set(h) >= {"name", "score"}, f"药材字段缺失: {h}"
        assert isinstance(h["name"], str) and h["name"]
    for f in formulas:
        assert set(f) >= {"name", "score"}, f"方剂字段缺失: {f}"
        assert isinstance(f["name"], str) and f["name"]
