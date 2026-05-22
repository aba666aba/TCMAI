# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0
# See LICENSE file for details

"""
症状分析模块

原创功能：症状组合分析、证型判断、禁忌校验
所有数据定义从 tcm_config 统一读取。
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from core.tcm_config import (
    SYMPTOM_ALIAS_MAP,
    SYMPTOM_CATEGORIES,
    SYMPTOM_INFO,
    SYNDROME_RULES,
    CONTRAINDICATION_RULES,
)

logger = logging.getLogger(__name__)


@dataclass
class SymptomInfo:
    """症状信息"""
    name: str
    standardized_name: str
    category: str
    severity: int


@dataclass
class SyndromeResult:
    """证型判断结果"""
    syndrome_name: str
    confidence: float
    symptoms: List[str]
    explanation: str


class SymptomAnalyzer:
    """症状分析器，基于中医辨证论治理论进行症状分析"""

    def __init__(self):
        self.symptom_map = dict(SYMPTOM_ALIAS_MAP)
        self.symptom_categories = dict(SYMPTOM_CATEGORIES)
        self.syndrome_rules = dict(SYNDROME_RULES)
        self.contraindications = dict(CONTRAINDICATION_RULES)

    def analyze(self, symptoms: List[str]) -> Dict:
        """分析症状"""
        if not symptoms:
            return {
                "standardized": [],
                "syndromes": [],
                "contraindications": [],
                "suggestions": [],
            }

        standardized = self._standardize_symptoms(symptoms)
        syndromes = self._identify_syndromes(standardized)
        contraindications = self._check_contraindications(syndromes)
        suggestions = self._generate_suggestions(syndromes)

        return {
            "standardized": standardized,
            "syndromes": syndromes,
            "contraindications": contraindications,
            "suggestions": suggestions,
        }

    def _standardize_symptoms(self, symptoms: List[str]) -> List[SymptomInfo]:
        """将日常用语映射到中医专业术语"""
        standardized = []
        for symptom in symptoms:
            std_name = self.symptom_map.get(symptom, symptom)
            category = self.symptom_categories.get(std_name, "其他")
            symptom_info = SymptomInfo(
                name=symptom,
                standardized_name=std_name,
                category=category,
                severity=3,
            )
            standardized.append(symptom_info)

        logger.info("症状标准化完成：%s", [s.standardized_name for s in standardized])
        return standardized

    def _identify_syndromes(self, symptoms: List[SymptomInfo]) -> List[SyndromeResult]:
        """证型判断"""
        symptom_names = [s.standardized_name for s in symptoms]
        results = []

        for syndrome_name, rule in self.syndrome_rules.items():
            matched = [s for s in rule["symptoms"] if s in symptom_names]

            if not rule["symptoms"]:
                logger.warning("证型 '%s' 的症状列表为空，跳过该证型", syndrome_name)
                continue

            confidence = len(matched) / len(rule["symptoms"])

            if confidence >= 0.5:
                result = SyndromeResult(
                    syndrome_name=syndrome_name,
                    confidence=confidence,
                    symptoms=matched,
                    explanation=rule["explanation"],
                )
                results.append(result)

        results.sort(key=lambda x: x.confidence, reverse=True)
        logger.info("证型判断完成：%s", [r.syndrome_name for r in results])
        return results

    def _check_contraindications(self, syndromes: List[SyndromeResult]) -> List[str]:
        """禁忌检查"""
        contraindications = []
        for syndrome in syndromes:
            if syndrome.syndrome_name == "脾胃虚弱":
                contraindications.append("脾胃虚寒者慎用寒凉药物")

        logger.info("禁忌检查完成：%s", contraindications)
        return contraindications

    def _generate_suggestions(self, syndromes: List[SyndromeResult]) -> List[str]:
        """生成调理建议"""
        suggestions = []

        if not syndromes:
            return ["建议咨询专业中医师进行辨证施治"]

        top_syndrome = syndromes[0]

        if top_syndrome.syndrome_name == "脾胃虚弱":
            suggestions.extend([
                "饮食宜清淡易消化",
                "避免生冷油腻食物",
                "可适当食用山药、薏米等健脾食材",
            ])
        elif top_syndrome.syndrome_name == "肝郁气滞":
            suggestions.extend([
                "保持心情舒畅",
                "适当运动疏解压力",
                "可饮用玫瑰花茶疏肝理气",
            ])

        return suggestions

    def get_symptom_info(self, symptom_name: str) -> Optional[Dict]:
        """获取症状信息（从统一配置读取）"""
        info = SYMPTOM_INFO.get(symptom_name)
        if info:
            return {
                "name": symptom_name,
                "category": info["category"],
                "description": info["description"],
                "related_syndromes": info["related_syndromes"],
            }
        return None
