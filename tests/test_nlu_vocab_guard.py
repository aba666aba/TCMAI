# Copyright (C) 2026 aba666aba
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# 本文件是中医药智能问答系统 (TCMAI) 的一部分，按 AGPL-3.0-or-later 分发。

"""NLU 输出必须被约束在标准症状词表内。

背景：提示词里的"只输出列表内症状"是软约束，实测 0.5B 模型经常不遵守
（输入"睡不着、没力气、头晕"会输出词表外的 疲劳/乏累）。
src/nlu/engine.py::_normalize_symptoms 是这道硬约束，本模块锁住它的行为。

这里不加载模型，只验证归一化逻辑本身。
"""

import pytest

from src.nlu.engine import NLUEngine
from tcm_recommender.extended_config import EXTENDED_SYMPTOM_LIST


@pytest.fixture(scope="module")
def engine():
    """不触发 load()，仅用于调用词表与归一化逻辑。"""
    return NLUEngine(device="cpu")


def test_vocabulary_loaded_from_model_config(engine):
    vocab, alias_map = engine._load_vocabulary()
    assert vocab, "症状词表为空"
    assert vocab == set(EXTENDED_SYMPTOM_LIST), "NLU 词表应与模型词表同源"
    assert alias_map, "别名表为空，归一化将退化为仅精确匹配"


@pytest.mark.parametrize(
    "raw,expected",
    [
        (["失眠", "乏力", "头晕"], ["失眠", "乏力", "头晕"]),
        (["疲劳", "乏累"], ["乏力"]),
        (["睡不着", "睡不好", "没力气"], ["失眠", "乏力"]),
        (["外星人症状", "乏力"], ["乏力"]),
        (["失眠", "失眠", "乏力"], ["失眠", "乏力"]),
        ([], []),
    ],
)
def test_normalize_maps_or_drops(engine, raw, expected):
    assert engine._normalize_symptoms(raw) == expected


def test_normalize_never_emits_out_of_vocab(engine):
    """核心不变量：无论输入什么，输出都必须落在词表内。"""
    vocab, _ = engine._load_vocabulary()
    messy = [
        "疲劳",
        "乏累",
        "睡不着",
        "没力气",
        "some english word",
        "12345",
        "  腹胀  ",
        "不存在的症状名",
        "怕冷",
        "手脚冰凉",
    ]
    out = engine._normalize_symptoms(messy)
    assert isinstance(out, list)
    offenders = [s for s in out if s not in vocab]
    assert not offenders, f"输出含词表外症状: {offenders}"


def test_normalize_preserves_ordering(engine):
    """归一化后应保持原始相对顺序，便于上游按序展示。"""
    out = engine._normalize_symptoms(["睡不着", "乏力", "头晕"])
    assert out == ["失眠", "乏力", "头晕"]
