# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0

"""
统一症状处理管线

合并 fallback.detect_symptoms() 和 SymptomAnalyzer 的功能，
提供单一入口：症状检测 → 标准化 → 证型推断 → 禁忌检查。
"""

import logging
from typing import Dict, Optional

from core.symptom_analysis import SymptomAnalyzer
from core.tcm_config import (
    FALLBACK_SYMPTOMS,
    SYMPTOM_KEYWORDS,
    SYNDROME_DEFINITIONS,
)

logger = logging.getLogger(__name__)


class SymptomPipeline:
    """统一症状处理管线"""

    def __init__(self):
        self.analyzer = SymptomAnalyzer()

    def detect(self, query: str) -> list[str]:
        """从自然语言查询中检测症状（关键词匹配）"""
        detected = []
        query_lower = query.lower()
        for symptom, keywords in SYMPTOM_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    if symptom not in detected:
                        detected.append(symptom)
                    break
        return detected

    def analyze(self, symptoms: list[str]) -> dict:
        """分析症状列表：标准化 → 证型推断 → 禁忌检查"""
        return self.analyzer.analyze(symptoms)

    def process(self, query: str, symptoms: Optional[list[str]] = None) -> dict:
        """完整处理管线：检测 + 分析

        Returns:
            {
                "detected": [...],       # 从 query 检测到的症状
                "standardized": [...],    # 标准化后的 SymptomInfo
                "syndromes": [...],       # 证型推断结果
                "contraindications": [...], # 禁忌
                "suggestions": [...],     # 调理建议
                "info": {...},            # 症状详细信息
            }
        """
        result: dict = {
            "detected": [],
            "standardized": [],
            "syndromes": [],
            "contraindications": [],
            "suggestions": [],
            "info": {},
        }

        # 1. 关键词检测
        result["detected"] = self.detect(query)

        # 2. 合并外部传入的症状
        all_symptoms = list(result["detected"])
        if symptoms:
            for s in symptoms:
                if isinstance(s, str) and s.strip() and s not in all_symptoms:
                    all_symptoms.append(s.strip())

        if not all_symptoms:
            return result

        # 3. 症状分析（标准化 + 证型）
        analysis = self.analyzer.analyze(all_symptoms)
        result["standardized"] = analysis.get("standardized", [])
        result["syndromes"] = analysis.get("syndromes", [])
        result["contraindications"] = analysis.get("contraindications", [])
        result["suggestions"] = analysis.get("suggestions", [])

        # 4. 获取症状详情
        for symptom in all_symptoms:
            info = self.analyzer.get_symptom_info(symptom)
            if info:
                result["info"][symptom] = info

        return result

    def get_symptom_info(self, symptom_name: str) -> Optional[Dict]:
        """获取症状详细信息"""
        return self.analyzer.get_symptom_info(symptom_name)

    def get_fallback_symptom_info(self, symptom_name: str) -> Optional[Dict]:
        """获取降级症状信息（含推荐方剂）"""
        return FALLBACK_SYMPTOMS.get(symptom_name)

    def get_syndrome_info(self, syndrome_name: str) -> Optional[Dict]:
        """获取证型信息"""
        return SYNDROME_DEFINITIONS.get(syndrome_name)
