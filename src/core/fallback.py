# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0
# See LICENSE file for details

"""
降级回答模块

当 AI 模型无法正常工作时，使用基于规则的中医药知识提供回答。
所有知识数据从 tcm_config 统一读取。
"""

import logging
import re
from typing import Optional, List, Dict, Any

from core.tcm_config import (
    FALLBACK_HERBS,
    FALLBACK_FORMULAS,
    FALLBACK_SYMPTOMS,
    SYNDROME_DEFINITIONS,
    SYMPTOM_KEYWORDS,
    TCM_KEYWORDS,
    generate_health_tips,
)

# 向后兼容：旧代码可能直接引用这些名称
TCM_KNOWLEDGE = {
    "herbs": FALLBACK_HERBS,
    "formulas": FALLBACK_FORMULAS,
    "symptoms": FALLBACK_SYMPTOMS,
}
SYNDROMES = SYNDROME_DEFINITIONS

logger = logging.getLogger(__name__)


def detect_symptoms(query: str) -> List[str]:
    """从用户问题中检测症状（使用正则避免子串误匹配）"""
    detected = []

    for symptom, keywords in SYMPTOM_KEYWORDS.items():
        for keyword in keywords:
            if re.search(keyword, query):
                if symptom not in detected:
                    detected.append(symptom)
                break

    return detected


def search_herbs(keyword: str) -> List[Dict]:
    """搜索药材"""
    results = []
    for herb in FALLBACK_HERBS:
        if (
            keyword in herb["name"]
            or keyword in herb.get("alias", "")
            or keyword in herb.get("efficacy", "")
        ):
            results.append(herb)
    return results[:5]


def search_formulas(keyword: str) -> List[Dict]:
    """搜索方剂"""
    results = []
    for formula in FALLBACK_FORMULAS:
        if (
            keyword in formula["name"]
            or keyword in formula.get("efficacy", "")
            or keyword in formula.get("indications", "")
        ):
            results.append(formula)
    return results[:5]


def generate_fallback_answer(query: str, kb_result: Optional[Dict] = None) -> Dict[str, Any]:
    """生成降级回答（基于规则的中医药知识回答）"""
    logger.info("使用降级方案生成回答：%s", query[:50])

    symptoms = detect_symptoms(query)
    answer_parts = []

    if symptoms:
        answer_parts.append("【症状分析】")
        for symptom in symptoms:
            if symptom in FALLBACK_SYMPTOMS:
                info = FALLBACK_SYMPTOMS[symptom]
                answer_parts.append(f"• {symptom}：{info['description']}")
                answer_parts.append(f"  可能证型：{info['syndromes']}")
                answer_parts.append(f"  建议：{info['recommendation']}")
    else:
        answer_parts.append("【分析】")
        answer_parts.append("根据您的描述，我来提供相关的中医药知识参考。")

    if symptoms:
        answer_parts.append("\n【推荐方剂】")
        for symptom in symptoms:
            if symptom in FALLBACK_SYMPTOMS:
                info = FALLBACK_SYMPTOMS[symptom]
                answer_parts.append(f"针对{symptom}，可考虑：{info['recommendation']}")

    answer_parts.append("\n【相关知识】")
    keywords = extract_keywords(query)
    for keyword in keywords[:3]:
        herbs = search_herbs(keyword)
        formulas = search_formulas(keyword)

        if herbs:
            answer_parts.append(f"• 相关药材（{keyword}）：")
            for herb in herbs[:2]:
                answer_parts.append(f"  - {herb['name']}：{herb['efficacy']}")

        if formulas:
            answer_parts.append(f"• 相关方剂（{keyword}）：")
            for formula in formulas[:2]:
                answer_parts.append(f"  - {formula['name']}：{formula['efficacy']}")
                answer_parts.append(f"    组成：{formula['composition']}")

    answer_parts.append("\n【养生建议】")
    sym_set = set(symptoms)
    answer_parts.extend(generate_health_tips(sym_set))

    answer = "\n".join(answer_parts)
    disclaimer = (
        "温馨提示：本系统仅提供中医药知识科普和参考信息，不构成医疗诊断或治疗建议。"
        "用户如有健康问题，请咨询专业医疗机构或执业医师。"
    )

    return {
        "answer": answer,
        "knowledge": {"symptoms": symptoms, "source": "fallback_mode"},
        "disclaimer": disclaimer,
        "mode": "fallback",
    }


def extract_keywords(query: str) -> List[str]:
    """从查询中提取中医药领域关键词（使用正则匹配）"""
    keywords = []
    for keyword in TCM_KEYWORDS:
        if re.search(keyword, query):
            keywords.append(keyword)
    return keywords


def get_fallback_info(topic: str) -> Dict[str, Any]:
    """获取特定主题的降级信息"""
    result: Dict[str, Any] = {"found": False, "type": None, "data": None}

    for herb in FALLBACK_HERBS:
        if topic in herb["name"] or topic in herb.get("alias", ""):
            result["found"] = True
            result["type"] = "herb"
            result["data"] = herb
            return result

    for formula in FALLBACK_FORMULAS:
        if topic in formula["name"]:
            result["found"] = True
            result["type"] = "formula"
            result["data"] = formula
            return result

    if topic in FALLBACK_SYMPTOMS:
        result["found"] = True
        result["type"] = "symptom"
        result["data"] = FALLBACK_SYMPTOMS[topic]
        return result

    if topic in SYNDROME_DEFINITIONS:
        result["found"] = True
        result["type"] = "syndrome"
        result["data"] = SYNDROME_DEFINITIONS[topic]
        return result

    return result
