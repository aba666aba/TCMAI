# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0
# See LICENSE file for details

"""
症状分析模块

原创功能：症状组合分析、证型判断、禁忌校验
所有数据定义从 tcm_config 统一读取（SYNDROME_DEFINITIONS 9 证型）。
"""

import logging
from dataclasses import dataclass
from typing import Optional

from core.tcm_config import (
    CONTRAINDICATION_RULES,
    SYMPTOM_ALIAS_MAP,
    SYMPTOM_CATEGORIES,
    SYMPTOM_INFO,
    SYNDROME_DEFINITIONS,
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
    symptoms: list[str]
    explanation: str
    treatment: str = ""
    formula: str = ""


class SymptomAnalyzer:
    """症状分析器，基于中医辨证论治理论进行症状分析"""

    def __init__(self):
        self.symptom_map = dict(SYMPTOM_ALIAS_MAP)
        self.symptom_categories = dict(SYMPTOM_CATEGORIES)
        # 使用 SYNDROME_DEFINITIONS（9 证型）替代 SYNDROME_RULES（3 证型）
        self.syndrome_definitions = dict(SYNDROME_DEFINITIONS)
        self.contraindications = dict(CONTRAINDICATION_RULES)

    def analyze(self, symptoms: list[str]) -> dict:
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

    def _standardize_symptoms(self, symptoms: list[str]) -> list[SymptomInfo]:
        """将日常用语映射到中医专业术语"""
        standardized = []
        for symptom in symptoms:
            std_name = self.symptom_map.get(symptom, symptom)
            category = self.symptom_categories.get(std_name, "其他")
            # severity 从 SYMPTOM_INFO 推断：伴随便秘/出血/高热等症状时权重更高
            info = SYMPTOM_INFO.get(std_name, {})
            serious_keywords = ["出血", "剧烈", "高热", "神昏", "剧痛", "昏迷", "休克"]
            desc = info.get("description", "")
            severity = 5 if any(k in desc for k in serious_keywords) else 3
            symptom_info = SymptomInfo(
                name=symptom,
                standardized_name=std_name,
                category=category,
                severity=severity,
            )
            standardized.append(symptom_info)

        logger.info("症状标准化完成：%s", [s.standardized_name for s in standardized])
        return standardized

    def _identify_syndromes(self, symptoms: list[SymptomInfo]) -> list[SyndromeResult]:
        """证型判断（基于 SYNDROME_DEFINITIONS 9 证型）"""
        symptom_names: set[str] = {s.standardized_name for s in symptoms}
        results = []

        for syndrome_name, definition in self.syndrome_definitions.items():
            required = definition.get("symptoms", [])
            if not required:
                logger.warning("证型 '%s' 的症状列表为空，跳过", syndrome_name)
                continue

            matched = [s for s in required if s in symptom_names]
            confidence = len(matched) / len(required)

            if confidence >= 0.4:
                result = SyndromeResult(
                    syndrome_name=syndrome_name,
                    confidence=confidence,
                    symptoms=matched,
                    explanation=definition.get("description", ""),
                    treatment=definition.get("treatment", ""),
                    formula=definition.get("formula", ""),
                )
                results.append(result)

        results.sort(key=lambda x: x.confidence, reverse=True)
        logger.info("证型判断完成：%s", [r.syndrome_name for r in results])
        return results

    def _check_contraindications(self, syndromes: list[SyndromeResult]) -> list[str]:
        """禁忌检查（基于 CONTRAINDICATION_RULES）"""
        warnings = []
        for syndrome in syndromes:
            name = syndrome.syndrome_name
            # 通用禁忌规则
            if name in ("脾胃虚弱", "心脾两虚"):
                warnings.append("脾胃虚寒者慎用寒凉药物")
            if name in ("肾阳虚", "风寒感冒"):
                warnings.append("阳虚者慎用清热泻火类药物")
            if name in ("阴虚火旺", "肾阴虚"):
                warnings.append("阴虚者慎用温燥药物")

        logger.info("禁忌检查完成：%s", warnings)
        return warnings

    def _generate_suggestions(self, syndromes: list[SyndromeResult]) -> list[str]:
        """生成调理建议（基于9种证型）"""
        suggestions = []

        if not syndromes:
            return ["建议咨询专业中医师进行辨证施治"]

        top_syndrome = syndromes[0]
        name = top_syndrome.syndrome_name

        # 证型-建议映射表
        suggestion_map = {
            "脾胃虚弱": [
                "饮食宜清淡易消化",
                "避免生冷油腻食物",
                "可适当食用山药、薏米等健脾食材",
            ],
            "气血两虚": [
                "宜食补气养血食物，如红枣、桂圆、瘦肉",
                "避免过度劳累，保证充足睡眠",
                "可适当食用当归、黄芪药膳",
            ],
            "肝郁气滞": [
                "保持心情舒畅，避免情绪激动",
                "适当运动疏解压力",
                "可饮用玫瑰花茶疏肝理气",
            ],
            "肾阳虚": [
                "注意保暖，避免受寒",
                "可适量食用羊肉、韭菜、核桃等温阳食物",
                "避免生冷饮食和过度劳累",
            ],
            "阴虚火旺": [
                "饮食宜清淡，少食辛辣温燥之品",
                "保持充足睡眠，避免熬夜",
                "可适当食用百合、银耳、鸭肉等滋阴食物",
            ],
            "风寒感冒": [
                "注意保暖，多饮温水",
                "可食用生姜红糖水发汗散寒",
                "适当休息，避免劳累",
            ],
            "风热感冒": [
                "多饮温水，少食辛辣",
                "可用金银花、菊花泡水饮用",
                "保持空气流通，注意休息",
            ],
            "心脾两虚": [
                "调养心神，避免过度思虑",
                "宜食养心安神之品，如红枣、龙眼肉",
                "规律作息，适度运动",
            ],
            "肾阴虚": [
                "避免熬夜，节制房事",
                "可适当食用黑芝麻、枸杞、桑葚等补肾之品",
                "避免过度劳累和精神紧张",
            ],
        }

        if name in suggestion_map:
            suggestions.extend(suggestion_map[name])
        else:
            suggestions.append("建议咨询专业中医师进行辨证施治")

        return suggestions

    def get_symptom_info(self, symptom_name: str) -> Optional[dict]:
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
