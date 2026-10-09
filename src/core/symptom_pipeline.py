# Copyright (C) 2026 中医药智能问答系统 (TCMAI)
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# 本文件是中医药智能问答系统 (TCMAI) 的一部分。
#
# 本程序是自由软件：你可以依据自由软件基金会发布的 GNU Affero 通用公共
# 许可证条款（第 3 版或你选择的任何更新版本）重新分发和/或修改本程序。
#
# 本程序按“希望有用”的原则分发，但不提供任何担保，甚至不提供适销性或
# 特定用途适用性的默示担保。详见 GNU Affero 通用公共许可证。
#
# 你应已随本程序收到 GNU Affero 通用公共许可证副本。如未收到，
# 请见 <https://www.gnu.org/licenses/>。
#
# 注意：以网络服务方式提供本程序的服务端修改版本，同样必须按 AGPL-3.0
# 第 13 条向所有用户提供对应源代码。

"""统一症状处理管线 — 症状检测 → 标准化 → 证型推断 → 禁忌检查"""

import json
import logging

from core.symptom_analysis import SymptomAnalyzer
from core.tcm_config import (
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

    def process(self, query: str, symptoms: list[str] | None = None) -> dict:
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

    def get_symptom_info(self, symptom_name: str) -> dict | None:
        """获取症状详细信息"""
        return self.analyzer.get_symptom_info(symptom_name)

    def get_syndrome_info(self, syndrome_name: str) -> dict | None:
        """获取证型信息"""
        return SYNDROME_DEFINITIONS.get(syndrome_name)
