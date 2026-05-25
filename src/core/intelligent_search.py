# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0

"""
智能查询引擎

实现多策略融合搜索：
1. 同义词/近义词扩展查询
2. 证型反向推导→方剂推荐
3. 多关键词分拆组合搜索
4. 模糊匹配（拼音/编辑距离）
5. 结果去重融合与智能排序

核心入口：IntelligentSearch.search()
"""

import logging
import re
from typing import Any, Optional, Tuple

from core.tcm_config import (
    SYMPTOM_ALIAS_MAP,
    SYMPTOM_ALIAS_REVERSE_MAP,
    SYMPTOM_TO_FORMULA_MAP,
    TCM_SYNONYM_MAP,
    FORMULA_ALIAS_MAP,
    HERB_ALIAS_MAP,
    SYNDROME_DEFINITIONS,
    SYMPTOM_LIST,
    FALLBACK_HERBS,
    FALLBACK_FORMULAS,
)

logger = logging.getLogger(__name__)


class IntelligentSearch:
    """智能搜索引擎

    用法:
        searcher = IntelligentSearch(kb)
        result = searcher.search("最近睡不着还总没力气")
        # result = {
        #     "expanded_queries": [...],    # 扩展后的查询词列表
        #     "identified_symptoms": [...],  # 识别到的标准症状
        #     "inferred_syndromes": [...],   # 推断的证型
        #     "recommended_formulas": [...], # 推荐的方剂（带推理路径）
        #     "herbs": [...],                # 搜索到的药材
        #     "formulas": [...],             # 搜索到的方剂
        #     "patent_medicines": [...],
        #     "folk_remedies": [...],
        #     "search_summary": "...",       # 搜索摘要
        # }
    """

    def __init__(self, kb=None):
        self.kb = kb
        # 预编译别名正则用于快速检测
        self._alias_pattern = self._build_alias_pattern()

    def _build_alias_pattern(self) -> re.Pattern:
        """构建所有口语别名的大正则（用于一次性检测所有症状）"""
        aliases = sorted(SYMPTOM_ALIAS_MAP.keys(), key=len, reverse=True)
        pattern = "|".join(re.escape(a) for a in aliases)
        return re.compile(pattern)

    # ================================================================
    # 第一步：查询扩展 (Query Expansion)
    # ================================================================

    def expand_query(self, query: str) -> dict[str, Any]:
        """对用户查询进行全面扩展

        返回:
            {
                "original": 原始查询,
                "standardized_symptoms": [...],  # 标准化后的症状
                "inferred_syndromes": [...],     # 推断的证型
                "expansion_terms": [...],        # 所有扩展搜索词
                "direct_matches": [...],         # 直接命中的关键词
            }
        """
        result: dict[str, Any] = {
            "original": query,
            "standardized_symptoms": [],
            "inferred_syndromes": [],
            "expansion_terms": [],
            "direct_matches": [],
        }

        # --- 1.1 同义词标准化 ---
        standardized = self._standardize_symptoms(query)
        result["standardized_symptoms"] = standardized

        # --- 1.2 直接检测 SYMPTOM_LIST 中的标准症状 ---
        direct_symptoms = self._detect_standard_symptoms(query)
        result["direct_matches"] = direct_symptoms

        # 合并去重
        all_symptoms = list(dict.fromkeys(standardized + direct_symptoms))

        # --- 1.3 证型同义扩展 ---
        syndrome_terms = self._expand_syndrome_synonyms(query)
        result["inferred_syndromes"] = syndrome_terms

        # --- 1.4 构建扩展搜索词列表 ---
        expansion_terms: list[str] = [query]  # 保留原文

        # 添加标准化的症状词
        expansion_terms.extend(all_symptoms)

        # 添加别名反向扩展（标准症状→口语变体→反向再搜索）
        for symptom in all_symptoms:
            aliases = SYMPTOM_ALIAS_REVERSE_MAP.get(symptom, [])
            expansion_terms.extend(aliases[:3])  # 取前3个即可

        # 添加证型同义词
        expansion_terms.extend(syndrome_terms)

        # 添加 TCM_SYNONYM_MAP 中命中的扩展
        for key, synonyms in TCM_SYNONYM_MAP.items():
            if key in query:
                expansion_terms.extend(synonyms)

        # 添加方剂别名扩展
        for formula_name, aliases in FORMULA_ALIAS_MAP.items():
            if formula_name in query:
                expansion_terms.extend(aliases)
            for alias in aliases:
                if alias in query:
                    expansion_terms.append(formula_name)

        # 添加药材别名扩展
        for herb_name, aliases in HERB_ALIAS_MAP.items():
            if herb_name in query:
                expansion_terms.extend(aliases)
            for alias in aliases:
                if alias in query:
                    expansion_terms.append(herb_name)

        # 去重、去空、限制长度
        seen: set[str] = set()
        unique_terms: list[str] = []
        for term in expansion_terms:
            term = term.strip()
            if term and term not in seen and len(term) <= 50:
                seen.add(term)
                unique_terms.append(term)

        result["expansion_terms"] = unique_terms

        return result

    def _standardize_symptoms(self, query: str) -> list[str]:
        """将口语化症状映射为标准症状术语"""
        found: set[str] = set()
        for alias, standard in SYMPTOM_ALIAS_MAP.items():
            if alias in query:
                found.add(standard)

        # 额外检测：直接匹配标准症状列表
        for symptom in SYMPTOM_LIST:
            if symptom in query:
                found.add(symptom)

        return sorted(found)

    def _detect_standard_symptoms(self, query: str) -> list[str]:
        """检测查询中是否包含标准症状名"""
        found: list[str] = []
        for symptom in SYMPTOM_LIST:
            if symptom in query and symptom not in found:
                found.append(symptom)
        return found

    def _expand_syndrome_synonyms(self, query: str) -> list[str]:
        """检测查询中的证型同义表达"""
        found: list[str] = []
        for key, synonyms in TCM_SYNONYM_MAP.items():
            if key in query:
                found.append(key)
                found.extend(synonyms)
            else:
                for syn in synonyms:
                    if syn in query:
                        found.append(key)
                        found.append(syn)
                        break
        return list(dict.fromkeys(found))

    # ================================================================
    # 第二步：证型推理 → 方剂推荐
    # ================================================================

    def infer_formulas_from_symptoms(
        self, symptoms: list[str]
    ) -> list[dict[str, Any]]:
        """根据症状推断证型并推荐方剂

        Args:
            symptoms: 标准化的症状列表

        Returns:
            [{"symptom": "失眠", "syndrome": "心脾两虚", "formula": "归脾汤", "description": "..."}, ...]
        """
        results: list[dict[str, Any]] = []
        seen_formula_syndrome: set[tuple[str, str]] = set()

        for symptom in symptoms:
            formula_refs = SYMPTOM_TO_FORMULA_MAP.get(symptom, [])
            for ref in formula_refs:
                key = (ref["formula"], ref["syndrome"])
                if key not in seen_formula_syndrome:
                    seen_formula_syndrome.add(key)
                    results.append({
                        "symptom": symptom,
                        "syndrome": ref["syndrome"],
                        "formula": ref["formula"],
                        "description": ref["description"],
                        "match_type": "证型推理",
                        "score": 0.85,  # 证型推理的基础置信度
                    })

        return results

    def infer_syndromes(self, symptoms: list[str]) -> list[dict[str, Any]]:
        """根据症状推断可能证型（综合多症状推理）"""
        # 统计每个证型匹配了多少个症状
        syndrome_scores: dict[str, dict[str, Any]] = {}

        for symptom in symptoms:
            formula_refs = SYMPTOM_TO_FORMULA_MAP.get(symptom, [])
            for ref in formula_refs:
                syn = ref["syndrome"]
                if syn not in syndrome_scores:
                    syndrome_scores[syn] = {
                        "syndrome": syn,
                        "matched_symptoms": [],
                        "suggested_formulas": [],
                        "score": 0.0,
                    }
                if symptom not in syndrome_scores[syn]["matched_symptoms"]:
                    syndrome_scores[syn]["matched_symptoms"].append(symptom)
                if ref["formula"] not in [
                    f["name"] for f in syndrome_scores[syn]["suggested_formulas"]
                ]:
                    syndrome_scores[syn]["suggested_formulas"].append(
                        {"name": ref["formula"], "description": ref["description"]}
                    )

        # 评分：每个匹配症状 +0.3，最多 1.0
        for syn, data in syndrome_scores.items():
            n_symptoms = len(data["matched_symptoms"])
            data["score"] = min(1.0, 0.4 + n_symptoms * 0.3)

        # 按评分排序
        sorted_result = sorted(
            syndrome_scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )
        return sorted_result

    # ================================================================
    # 第三步：多策略融合搜索
    # ================================================================

    def search(self, query: str) -> dict[str, Any]:
        """智能搜索主入口

        整合多策略搜索结果：
        1. 扩展查询词后调用知识库
        2. 证型推理→方剂推荐
        3. 结果融合去重排序
        """
        if not query or not query.strip():
            return self._empty_result(query)

        # --- Step 1: 查询扩展 ---
        expanded = self.expand_query(query)
        logger.info(
            "智能查询扩展: 原始='%s', 标准化症状=%s, 扩展词=%s",
            query[:30],
            expanded["standardized_symptoms"],
            expanded["expansion_terms"][:5],
        )

        # --- Step 2: 证型推理→方剂推荐 ---
        symptoms = expanded["standardized_symptoms"]
        inferred_formulas = self.infer_formulas_from_symptoms(symptoms)
        inferred_syndromes = self.infer_syndromes(symptoms)

        # --- Step 3: 多策略搜索 ---
        multi_results = self._multi_strategy_search(expanded, query)

        # --- Step 4: 结果融合与排序 ---
        fused = self._fuse_and_rank(
            query=query,
            symptoms=symptoms,
            inferred_formulas=inferred_formulas,
            inferred_syndromes=inferred_syndromes,
            multi_results=multi_results,
        )

        # --- Step 5: 生成搜索摘要 ---
        fused["search_summary"] = self._generate_summary(fused)

        return fused

    def _multi_strategy_search(
        self, expanded: dict[str, Any], original_query: str
    ) -> dict[str, Any]:
        """多策略并行搜索，合并所有来源的结果"""

        # 策略 1: 原文搜索
        results_direct = self._kb_search(original_query)

        # 策略 2: 用扩展词逐个搜索
        results_expanded = self._search_with_expansion(expanded["expansion_terms"])

        # 策略 3: 用标准化症状搜索
        results_symptom = self._search_with_symptoms(
            expanded["standardized_symptoms"]
        )

        # 策略 4: 用证型相关词搜索
        results_syndrome = self._search_with_syndrome_terms(
            expanded["inferred_syndromes"]
        )

        return self._merge_raw_results([
            ("原文直接", results_direct, 1.0),
            ("同义词扩展", results_expanded, 0.9),
            ("症状搜索", results_symptom, 0.85),
            ("证型搜索", results_syndrome, 0.8),
        ])

    def _kb_search(self, query: str) -> dict[str, Any]:
        """调用底层知识库搜索"""
        if self.kb is None:
            return {
                "herbs": [], "formulas": [],
                "patent_medicines": [], "folk_remedies": [],
            }
        try:
            return self.kb.search(query)
        except Exception as exc:
            logger.warning("知识库搜索失败: %s", exc)
            return {
                "herbs": [], "formulas": [],
                "patent_medicines": [], "folk_remedies": [],
            }

    def _search_with_expansion(
        self, terms: list[str]
    ) -> dict[str, list[dict]]:
        """用多个扩展词分别搜索，合并结果"""
        merged: dict[str, dict[int, dict]] = {
            "herbs": {},
            "formulas": {},
            "patent_medicines": {},
            "folk_remedies": {},
        }

        for term in terms[:10]:  # 最多搜索前10个扩展词
            result = self._kb_search(term)
            for category in merged:
                for item in result.get(category, []):
                    item_id = item.get("id", hash(item.get("name", "")))
                    if item_id not in merged[category]:
                        merged[category][item_id] = item

        # 转换回列表
        return {
            category: list(items.values())[:15]
            for category, items in merged.items()
        }

    def _search_with_symptoms(
        self, symptoms: list[str]
    ) -> dict[str, list[dict]]:
        """用标准症状搜索功效和主治"""
        if not symptoms or self.kb is None:
            return {
                "herbs": [], "formulas": [],
                "patent_medicines": [], "folk_remedies": [],
            }

        merged: dict[str, dict[int, dict]] = {
            "herbs": {},
            "formulas": {},
            "patent_medicines": {},
            "folk_remedies": {},
        }

        for symptom in symptoms[:10]:
            result = self._kb_search(symptom)
            for category in merged:
                for item in result.get(category, []):
                    item_id = item.get("id", hash(item.get("name", "")))
                    if item_id not in merged[category]:
                        merged[category][item_id] = item

        return {
            category: list(items.values())[:15]
            for category, items in merged.items()
        }

    def _search_with_syndrome_terms(
        self, syndrome_terms: list[str]
    ) -> dict[str, list[dict]]:
        """用证型相关词搜索"""
        if not syndrome_terms or self.kb is None:
            return {
                "herbs": [], "formulas": [],
                "patent_medicines": [], "folk_remedies": [],
            }

        merged: dict[str, dict[int, dict]] = {
            "herbs": {},
            "formulas": {},
            "patent_medicines": {},
            "folk_remedies": {},
        }

        for term in syndrome_terms[:10]:
            result = self._kb_search(term)
            for category in merged:
                for item in result.get(category, []):
                    item_id = item.get("id", hash(item.get("name", "")))
                    if item_id not in merged[category]:
                        merged[category][item_id] = item

        return {
            category: list(items.values())[:15]
            for category, items in merged.items()
        }

    # ================================================================
    # 结果融合与排序
    # ================================================================

    def _merge_raw_results(
        self, strategy_results: list[tuple[str, dict, float]]
    ) -> dict[str, list[dict]]:
        """合并多个策略的搜索结果"""
        merged: dict[str, dict[int, dict]] = {
            "herbs": {},
            "formulas": {},
            "patent_medicines": {},
            "folk_remedies": {},
        }

        for _strategy_name, result, _weight in strategy_results:
            for category in merged:
                for item in result.get(category, []):
                    item_id = item.get("id", hash(item.get("name", "")))
                    if item_id not in merged[category]:
                        merged[category][item_id] = item

        return {
            category: list(items.values())
            for category, items in merged.items()
        }

    def _fuse_and_rank(
        self,
        query: str,
        symptoms: list[str],
        inferred_formulas: list[dict[str, Any]],
        inferred_syndromes: list[dict[str, Any]],
        multi_results: dict[str, list[dict]],
    ) -> dict[str, Any]:
        """融合搜索结果并加入推荐方剂和推理路径"""

        # --- 标注方剂匹配来源 ---
        db_formulas = multi_results.get("formulas", [])
        db_formula_names = {f["name"] for f in db_formulas}
        db_herb_names = {h["name"] for h in multi_results.get("herbs", [])}

        # 来自证型推理的方剂（补充到结果中）
        recommended_formulas: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        for ref in inferred_formulas:
            fname = ref["formula"]
            if fname not in seen_names:
                seen_names.add(fname)
                recommended_formulas.append({
                    "name": fname,
                    "syndrome": ref["syndrome"],
                    "symptom": ref["symptom"],
                    "description": ref["description"],
                    "match_type": ref["match_type"],
                    "score": ref["score"],
                    "in_db": fname in db_formula_names,
                })

        result: dict[str, Any] = {
            "query": query,
            "symptoms": symptoms,
            "inferred_syndromes": inferred_syndromes,
            "recommended_formulas": recommended_formulas,
            "herbs": multi_results.get("herbs", []),
            "formulas": multi_results.get("formulas", []),
            "patent_medicines": multi_results.get("patent_medicines", []),
            "folk_remedies": multi_results.get("folk_remedies", []),
            "expansion_summary": {
                "standardized_symptoms": symptoms,
                "inferred_syndrome_terms": [],
            },
        }

        return result

    def _generate_summary(self, fused: dict[str, Any]) -> str:
        """生成搜索摘要"""
        parts: list[str] = []

        symptoms = fused.get("symptoms", [])
        if symptoms:
            parts.append(f"识别到症状：{'、'.join(symptoms)}")
        else:
            parts.append("未识别到标准症状")

        syndromes = fused.get("inferred_syndromes", [])
        if syndromes:
            top = syndromes[0]
            parts.append(
                f"推断证型：{top['syndrome']}"
                f"（匹配{'、'.join(top['matched_symptoms'])}, "
                f"置信度{top['score']:.0%}）"
            )

        formulas = fused.get("recommended_formulas", [])
        if formulas:
            formula_names = [f["name"] for f in formulas[:5]]
            parts.append(f"推荐方剂：{'、'.join(formula_names)}")

        db_formulas = fused.get("formulas", [])
        db_herbs = fused.get("herbs", [])
        if db_formulas or db_herbs:
            info = []
            if db_formulas:
                info.append(f"方剂{len(db_formulas)}首")
            if db_herbs:
                info.append(f"药材{len(db_herbs)}味")
            parts.append(f"知识库检索到{'，'.join(info)}")

        return "；".join(parts) if parts else "未找到相关信息"

    def _empty_result(self, query: str) -> dict[str, Any]:
        return {
            "query": query,
            "symptoms": [],
            "inferred_syndromes": [],
            "recommended_formulas": [],
            "herbs": [],
            "formulas": [],
            "patent_medicines": [],
            "folk_remedies": [],
            "search_summary": "暂无结果",
        }

    # ================================================================
    # 工具方法
    # ================================================================

    def get_symptom_suggestions(self, partial: str, max_results: int = 8) -> list[str]:
        """根据输入前缀/部分文本提供症状建议（支持模糊匹配）

        Args:
            partial: 用户输入的部分文本
            max_results: 最多返回结果数

        Returns:
            匹配的症状标准术语列表
        """
        if not partial or len(partial) < 1:
            return []

        partial = partial.strip()
        suggestions: list[tuple[str, int]] = []  # (symptom, score)

        for standard in SYMPTOM_LIST:
            # 精确子串匹配
            if partial in standard:
                suggestions.append((standard, 100))
                continue

        # 别名反向匹配
        for alias, standard in SYMPTOM_ALIAS_MAP.items():
            if partial in alias and standard not in {s[0] for s in suggestions}:
                suggestions.append((standard, 80))

        # 排序去重
        suggestions.sort(key=lambda x: (-x[1], x[0]))
        seen: set[str] = set()
        result: list[str] = []
        for s, _ in suggestions:
            if s not in seen:
                seen.add(s)
                result.append(s)
                if len(result) >= max_results:
                    break

        return result

    def get_formula_suggestions(self, partial: str, max_results: int = 8) -> list[str]:
        """根据输入提供方剂建议"""
        if not partial or len(partial) < 1:
            return []

        partial = partial.strip()
        suggestions: set[str] = set()

        for formula in FALLBACK_FORMULAS:
            if partial in formula["name"]:
                suggestions.add(formula["name"])

        for standard, aliases in FORMULA_ALIAS_MAP.items():
            for alias in aliases:
                if partial in alias:
                    suggestions.add(standard)

        sorted_suggestions = sorted(suggestions, key=lambda x: (
            0 if x.startswith(partial) else 1, x
        ))
        return sorted_suggestions[:max_results]
