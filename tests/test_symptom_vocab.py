# Copyright (C) 2026 aba666aba
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# 本文件是中医药智能问答系统 (TCMAI) 的一部分，按 AGPL-3.0-or-later 分发。

"""症状词表与别名表的一致性约束。

这些不变量此前没有任何东西在守，导致过两类真实缺陷：
  1. src/nlu/engine.py 输出的症状名落在 229 词表之外（下游无法编码）
  2. tcm_recommender 与 src/core 各自维护别名表，目标值可能指向不存在的症状

本模块把这些约束固化为测试，任何一侧漂移都会在提交时失败。
"""

import pytest

from src.core.tcm_config import (
    SYMPTOM_ALIAS_MAP as CORE_ALIAS_MAP,
    SYMPTOM_LIST as CORE_SYMPTOM_LIST,
    SYNDROME_DEFINITIONS,
    TCM_SYNONYM_MAP,
)
from tcm_recommender.extended_config import (
    EXTENDED_SYMPTOM_LIST,
    SYMPTOM_ALIAS_MAP as MODEL_ALIAS_MAP,
)


def test_vocabularies_are_non_trivial():
    """两份词表都应达到设计规模，避免被误删成空表。"""
    assert len(CORE_SYMPTOM_LIST) >= 50, "core 症状词表异常偏小"
    assert len(EXTENDED_SYMPTOM_LIST) >= 200, "模型侧症状词表异常偏小"


def test_vocab_length_matches_checkpoint():
    """词表长度必须与已发布权重的嵌入形状一致——这是一条硬约束。

    `best_model.pth` 的 `symptom_encoder.embedding.weight` 形状为 [230, 384]，
    即 229 个条目 + 1 个填充位，与 EXTENDED_SYMPTOM_LIST **按下标逐行对应**。

    曾经有一次改动删除了列表中重复的 `手足心热`（使长度 229 → 228），
    结果权重加载直接失败：
        size mismatch for symptom_encoder.embedding.weight:
        copying a param with shape torch.Size([230, 384]), the shape in current model is [229, 384]
    本测试就是为拦住这类"看起来是清理、实则破坏兼容"的改动。

    若确实需要修改词表，必须先重新训练权重并同步更新
    data/tcm_recommender 侧发布物，不能只改列表。
    """
    assert len(EXTENDED_SYMPTOM_LIST) == 229, (
        f"词表长度被改为 {len(EXTENDED_SYMPTOM_LIST)}，与已发布权重的 [230, 384] 嵌入不再匹配"
    )


def test_model_vocab_known_duplicate_is_documented():
    """已知重复项必须保持"恰好重复一次"：字典去重后 ID 空间仍等于设计值。

    这是有意的历史遗留（权重按下标对应），不是待修的数据错误。
    若有人把它"修"成 0 次或 2 次，长度断言与权重兼容性都会被破坏。
    """
    duplicates = {
        name: EXTENDED_SYMPTOM_LIST.count(name)
        for name in set(EXTENDED_SYMPTOM_LIST)
        if EXTENDED_SYMPTOM_LIST.count(name) > 1
    }
    assert duplicates == {"手足心热": 2}, f"重复项集合发生变化: {duplicates}"


def test_model_alias_targets_stay_inside_model_vocab():
    """模型侧别名表的每个目标值都必须是模型词表内的合法症状。

    否则 NLU 归一化会产出模型无法编码的名字（下游按索引取症状，词表外即失效）。
    """
    vocab = set(EXTENDED_SYMPTOM_LIST)
    invalid = {k: v for k, v in MODEL_ALIAS_MAP.items() if v not in vocab}
    assert not invalid, f"模型侧别名表指向词表外症状: {invalid}"


def test_core_alias_targets_are_resolvable():
    """core 侧别名表的目标值必须能被下游继续解析，不能指向"无人认领"的字符串。

    合法目标集合为三层并集：
      1. CORE_SYMPTOM_LIST      —— 标准症状
      2. SYNDROME_DEFINITIONS   —— 已定义证型（如 肝郁气滞）
      3. TCM_SYNONYM_MAP.keys() —— 口语证型伞词（如 肾虚/脾虚），
                                  它们再由同义表展开为正式证型，属设计如此

    第 3 层此前被遗漏，导致"阳痿→肾虚"看起来像指向了不存在的目标。
    """
    allowed = (
        set(CORE_SYMPTOM_LIST)
        | set(SYNDROME_DEFINITIONS)
        | set(TCM_SYNONYM_MAP)
    )
    invalid = {k: v for k, v in CORE_ALIAS_MAP.items() if v not in allowed}
    assert not invalid, f"core 侧别名表指向下游无法解析的目标: {invalid}"


def test_colloquial_syndrome_targets_have_synonym_expansion():
    """口语证型伞词必须能在同义表里找到，否则该路径在检索时断链。"""
    targets = set(CORE_ALIAS_MAP.values()) - set(CORE_SYMPTOM_LIST)
    unexpanded = {t for t in targets if t not in TCM_SYNONYM_MAP}
    assert not unexpanded, f"口语证型未在同义表中定义，检索路径会断链: {unexpanded}"


def test_alias_maps_are_deliberately_distinct():
    """两份别名表服务不同层次，不应被误当成重复定义而合并。

    - core  ：口语 → 粗粒度症状/证型，服务于智能搜索与证型推断
    - model ：口语 → 229 词表内具体症状，服务于模型编码

    若将来有人把它们合并，二者会退化为同一份，本测试会失败并提示重新评估语义。
    """
    assert CORE_ALIAS_MAP is not MODEL_ALIAS_MAP
    assert len(CORE_ALIAS_MAP) > len(MODEL_ALIAS_MAP), (
        "core 别名表应覆盖比模型侧更粗粒度的语义映射"
    )


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("疲劳", "乏力"),
        ("睡不着", "失眠"),
        ("没力气", "乏力"),
    ],
)
def test_model_alias_map_has_expected_entries(alias, expected):
    """抽样验证模型侧别名表的关键条目，防止批量编辑时被误改。"""
    assert MODEL_ALIAS_MAP.get(alias) == expected
