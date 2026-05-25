"""
KG 增强推荐器

将 TCMNet 模型预测结果与知识图谱推理融合：
  1. 模型预测 → 药材/方剂/证型分数
  2. KG推理 → 症状→证型→推荐药材/方剂（含路径）
  3. 融合 → 加权合并 + 图谱验证 + 推理路径解释
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

_PROJECT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


class KGEnhancedRecommender:
    """KG 增强推荐器（包装 TCMRecommender + 知识图谱）"""

    def __init__(self, model_recommender, kg_query, kg_config: Optional[Dict] = None):
        """
        Args:
            model_recommender: TCMRecommender 实例
            kg_query: KnowledgeGraphQuery 实例
            kg_config: 融合参数
        """
        self.model = model_recommender
        self.kg = kg_query
        self.cfg = kg_config or {
            "model_weight": 0.6,  # 模型预测权重
            "kg_weight": 0.4,  # KG推理权重
            "kg_boost": 0.15,  # KG验证通过的额外加分
            "syndrome_weight": 0.3,  # 证型推荐路径权重
        }

    # ═══════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════

    def recommend(
        self, symptom_text: str, top_k_herb: int = 10, top_k_formula: int = 5
    ) -> Dict:
        """
        KG 增强推荐

        Args:
            symptom_text: 用户输入的症状文本（空格/逗号分隔）
            top_k_herb: 返回药材数
            top_k_formula: 返回方剂数

        Returns:
            {
                "symptoms": [...],          # 识别的症状
                "syndrome": {...},          # 证型推断
                "herbs": [...],             # 推荐药材（含路径）
                "formulas": [...],          # 推荐方剂（含路径）
                "reasoning_paths": [...],   # 推理路径
                "kg_insights": {...},       # 图谱洞察
            }
        """
        # ── 步骤1: 解析症状 ──
        symptom_ids, valid_symptoms, unknown = self.model.parse_symptoms(symptom_text)
        if not symptom_ids:
            return self._error_result("未识别到有效症状", symptom_text)

        # ── 步骤2: 模型预测 ──
        model_result = self.model.recommend(symptom_ids, top_k_herb, top_k_formula)

        # ── 步骤3: KG 推理（多条路径） ──
        kg_result = self._kg_reasoning(valid_symptoms)

        # ── 步骤4: 融合 ──
        fused = self._fusion(model_result, kg_result, top_k_herb, top_k_formula)

        # 修复症状显示（从解析结果取）
        fused["symptoms"] = valid_symptoms

        # ── 步骤5: 构建推理路径 ──
        fused["reasoning_paths"] = self._build_paths(valid_symptoms, kg_result)

        # ── 步骤6: 图谱洞察 ──
        fused["kg_insights"] = self._kg_insights(fused["herbs"], fused["formulas"])

        # ── 步骤7: 安全检查（KG增强） ──
        herb_names = [h["name"] for h in fused["herbs"]]
        safety = self.model.check_safety(herb_names)
        # 补充KG禁忌检查
        kg_safety = self._kg_safety_check(herb_names)
        if kg_safety:
            safety["kg_warnings"] = kg_safety
        fused["safety"] = safety

        return fused

    # ═══════════════════════════════════════════
    # KG 推理
    # ═══════════════════════════════════════════

    def _kg_reasoning(self, symptoms: list[str]) -> dict:
        """
        KG 三条推理路径：
          A. 症状 → 证型 → 推荐药材/方剂（辨证论治路径）
          B. 症状 → 直接关联药材/方剂（症状-药物关联路径）
          C. 多症状 → 综合征型分析
        """
        result = {
            "syndrome_analysis": {"matched_syndromes": []},
            "symptom_direct": {},
            "syndrome_paths": [],
        }

        # 路径A: 症状→证型→推荐
        syn_analysis = self.kg.syndrome_analysis(symptoms)
        result["syndrome_analysis"] = syn_analysis

        # 收集证型路径上的推荐药材/方剂
        path_herbs = {}
        path_formulas = {}
        n_syn = len(syn_analysis.get("matched_syndromes", []))
        for syn in syn_analysis.get("matched_syndromes", []):
            syn_name = syn["name"]
            match_ratio = syn["matched_symptom_count"] / max(len(symptoms), 1)
            # 证型→药材
            for h in syn.get("recommended_herbs", []):
                if h not in path_herbs:
                    path_herbs[h] = {"score": 0, "paths": []}
                path_herbs[h]["score"] += match_ratio
                path_herbs[h]["paths"].append(f"症状→{syn_name}→{h}")
            # 证型→方剂
            for f in syn.get("recommended_formulas", []):
                if f not in path_formulas:
                    path_formulas[f] = {"score": 0, "paths": []}
                path_formulas[f]["score"] += match_ratio
                path_formulas[f]["paths"].append(f"症状→{syn_name}→{f}")

        # 归一化分数到 [0, 1]
        for d in [path_herbs, path_formulas]:
            max_score = max((v["score"] for v in d.values()), default=1)
            if max_score > 0:
                for v in d.values():
                    v["score"] = v["score"] / max_score

        result["syndrome_paths"] = {
            "herbs": path_herbs,
            "formulas": path_formulas,
        }

        # 路径B: 症状→直接关联
        direct_herbs = {}
        direct_formulas = {}
        for sym in symptoms:
            rec = self.kg.symptom_to_recommendation(sym)
            for h in rec.get("herbs", []):
                name = h["name"]
                if name not in direct_herbs:
                    direct_herbs[name] = 0
                direct_herbs[name] += h.get("score", 0)
            for f in rec.get("formulas", []):
                name = f["name"]
                if name not in direct_formulas:
                    direct_formulas[name] = 0
                direct_formulas[name] += f.get("score", 0)

        result["symptom_direct"] = {
            "herbs": direct_herbs,
            "formulas": direct_formulas,
        }

        return result

    # ═══════════════════════════════════════════
    # 融合
    # ═══════════════════════════════════════════

    def _fusion(
        self, model_result: Dict, kg_result: Dict, top_k_herb: int, top_k_formula: int
    ) -> Dict:
        """融合模型预测 + KG推理结果"""

        # ── 药材融合 ──
        herb_scores = {}  # name -> {"model_score": x, "kg_score": x, "boosted": bool}

        # 模型结果
        for h in model_result.get("herbs", []):
            name = h["name"]
            herb_scores[name] = {
                "model_score": h["score"],
                "kg_score": 0,
                "path": "模型预测",
            }

        # KG 证型路径
        for name, info in kg_result["syndrome_paths"]["herbs"].items():
            if name not in herb_scores:
                herb_scores[name] = {"model_score": 0, "kg_score": 0, "path": ""}
            herb_scores[name]["kg_score"] = max(
                herb_scores[name]["kg_score"], info["score"]
            )
            herb_scores[name]["path"] = (
                info["paths"][0] if info["paths"] else "证型推荐"
            )

        # KG 直接关联
        for name, score in kg_result["symptom_direct"]["herbs"].items():
            if name not in herb_scores:
                herb_scores[name] = {"model_score": 0, "kg_score": 0, "path": ""}
            herb_scores[name]["kg_score"] = max(
                herb_scores[name]["kg_score"], score * 0.2
            )

        # 计算融合分数
        mw = self.cfg["model_weight"]
        kw = self.cfg["kg_weight"]
        boost = self.cfg["kg_boost"]

        fused_herbs = []
        for name, scores in herb_scores.items():
            ms = scores["model_score"]
            ks = scores["kg_score"]
            # 融合公式
            fused_score = ms * mw + ks * kw
            # 如果模型和KG都推荐 → 额外加分
            if ms > 0 and ks > 0:
                fused_score += boost
            fused_herbs.append(
                {
                    "name": name,
                    "score": round(fused_score, 4),
                    "model_score": round(ms, 4),
                    "kg_score": round(ks, 4),
                    "path": scores["path"],
                }
            )

        fused_herbs.sort(key=lambda x: -x["score"])

        # ── 方剂融合（类似逻辑） ──
        formula_scores = {}
        for f in model_result.get("formulas", []):
            name = f["name"]
            formula_scores[name] = {
                "model_score": f["score"],
                "kg_score": 0,
                "path": "模型预测",
            }

        for name, info in kg_result["syndrome_paths"]["formulas"].items():
            if name not in formula_scores:
                formula_scores[name] = {"model_score": 0, "kg_score": 0, "path": ""}
            formula_scores[name]["kg_score"] = max(
                formula_scores[name]["kg_score"], info["score"]
            )
            formula_scores[name]["path"] = (
                info["paths"][0] if info["paths"] else "证型推荐"
            )

        for name, score in kg_result["symptom_direct"]["formulas"].items():
            if name not in formula_scores:
                formula_scores[name] = {"model_score": 0, "kg_score": 0, "path": ""}
            formula_scores[name]["kg_score"] = max(
                formula_scores[name]["kg_score"], score * 0.2
            )

        fused_formulas = []
        for name, scores in formula_scores.items():
            ms = scores["model_score"]
            ks = scores["kg_score"]
            fused_score = ms * mw + ks * kw
            if ms > 0 and ks > 0:
                fused_score += boost
            fused_formulas.append(
                {
                    "name": name,
                    "score": round(fused_score, 4),
                    "model_score": round(ms, 4),
                    "kg_score": round(ks, 4),
                    "path": scores["path"],
                }
            )

        fused_formulas.sort(key=lambda x: -x["score"])

        return {
            "symptoms": model_result.get("symptoms", []),
            "syndrome": model_result.get("syndrome"),
            "herbs": fused_herbs[:top_k_herb],
            "formulas": fused_formulas[:top_k_formula],
        }

    # ═══════════════════════════════════════════
    # 可解释性：推理路径
    # ═══════════════════════════════════════════

    def _build_paths(self, symptoms: list[str], kg_result: dict) -> list[dict]:
        """构建可读的推理路径"""
        paths = []

        # 证型推理路径
        for syn in kg_result["syndrome_analysis"].get("matched_syndromes", [])[:3]:
            syn_name = syn["name"]
            matched = syn.get("matched_symptom_count", 0)
            desc = syn.get("description", "")
            paths.append(
                {
                    "type": "syndrome_reasoning",
                    "title": f"辨证: {syn_name}",
                    "detail": (
                        f"输入症状中 {matched} 个匹配「{syn_name}」证型"
                        + (f"（{desc}）" if desc else "")
                    ),
                    "path": " → ".join(symptoms[:4] + [syn_name]),
                }
            )

        # 药材推荐路径
        for syn in kg_result["syndrome_analysis"].get("matched_syndromes", [])[:2]:
            syn_name = syn["name"]
            herbs = syn.get("recommended_herbs", [])[:3]
            if herbs:
                paths.append(
                    {
                        "type": "herb_recommendation",
                        "title": f"「{syn_name}」推荐药材",
                        "detail": f"证型「{syn_name}」的推荐药材: {'、'.join(herbs)}",
                        "path": f"症状 → {syn_name} → {' → '.join(herbs)}",
                    }
                )
            formulas = syn.get("recommended_formulas", [])[:2]
            if formulas:
                paths.append(
                    {
                        "type": "formula_recommendation",
                        "title": f"「{syn_name}」推荐方剂",
                        "detail": f"证型「{syn_name}」的推荐方剂: {'、'.join(formulas)}",
                        "path": f"症状 → {syn_name} → {' → '.join(formulas)}",
                    }
                )

        return paths

    # ═══════════════════════════════════════════
    # KG 洞察
    # ═══════════════════════════════════════════

    def _kg_insights(self, herbs: list[dict], formulas: list[dict]) -> dict:
        """从 KG 提取额外洞察"""
        insights = {}

        # 毒性标记
        toxic_herbs = []
        for h in herbs:
            profile = self.kg.herb_profile(h["name"])
            if isinstance(profile, dict) and profile.get("toxic"):
                toxic_herbs.append(h["name"])
        if toxic_herbs:
            insights["toxic_warning"] = (
                f"注意：{'、'.join(toxic_herbs)} 有毒性，请在医师指导下使用"
            )

        # 相反相畏
        incompatible_pairs = []
        for i, h1 in enumerate(herbs):
            for h2 in herbs[i + 1 :]:
                profile = self.kg.herb_profile(h1["name"])
                if isinstance(profile, dict):
                    if h2["name"] in profile.get("incompatible_with", []):
                        incompatible_pairs.append((h1["name"], h2["name"]))
        if incompatible_pairs:
            insights["incompatible_warning"] = [
                f"{a} 与 {b} 相反相畏" for a, b in incompatible_pairs
            ]

        return insights

    def _kg_safety_check(self, herb_names: list[str]) -> list[str]:
        """KG 安全检查"""
        warnings = []
        for h in herb_names:
            profile = self.kg.herb_profile(h)
            if isinstance(profile, dict):
                if profile.get("toxic"):
                    warnings.append(f"{h}: 有毒药材，注意用量")
                if profile.get("pregnancy_forbidden"):
                    warnings.append(f"{h}: 孕妇禁用")
        return warnings

    # ═══════════════════════════════════════════
    # 工具
    # ═══════════════════════════════════════════

    def _error_result(self, message: str, symptom_text: str) -> Dict:
        return {
            "error": message,
            "symptoms": [symptom_text],
            "herbs": [],
            "formulas": [],
            "reasoning_paths": [],
            "kg_insights": {},
            "safety": {"warnings": []},
        }

    def format_output(self, result: Dict) -> str:
        """格式化输出（含推理路径）"""
        if "error" in result:
            return f"错误：{result['error']}"

        lines = []
        lines.append("=" * 60)
        lines.append("  TCM 知识增强推荐")
        lines.append("=" * 60)

        lines.append(f"\n📋 症状：{'、'.join(result.get('symptoms', []))}")

        if result.get("syndrome"):
            s = result["syndrome"]
            lines.append(
                f"\n🏥 证型：{s['name']} (模型概率: {s.get('confidence', 0) * 100:.1f}%)"
            )

        lines.append("\n🌿 推荐药材：")
        for i, h in enumerate(result.get("herbs", []), 1):
            score_str = f"{h['score'] * 100:.1f}%"
            tag = (
                " [模型+图谱]"
                if h.get("model_score", 0) > 0 and h.get("kg_score", 0) > 0
                else " [模型]"
                if h.get("model_score", 0) > 0
                else " [图谱]"
            )
            lines.append(f"  {i}. {h['name']} ({score_str}){tag}")
            if h.get("path"):
                lines.append(f"     路径: {h['path']}")

        lines.append("\n📜 推荐方剂：")
        for i, f in enumerate(result.get("formulas", []), 1):
            score_str = f"{f['score'] * 100:.1f}%"
            tag = (
                " [模型+图谱]"
                if f.get("model_score", 0) > 0 and f.get("kg_score", 0) > 0
                else " [模型]"
                if f.get("model_score", 0) > 0
                else " [图谱]"
            )
            lines.append(f"  {i}. {f['name']} ({score_str}){tag}")

        if result.get("reasoning_paths"):
            lines.append("\n🔍 推理路径：")
            for p in result["reasoning_paths"][:4]:
                lines.append(f"  • {p['detail']}")

        if result.get("kg_insights"):
            ins = result["kg_insights"]
            if "toxic_warning" in ins:
                lines.append(f"\n⚠️  {ins['toxic_warning']}")
            if "incompatible_warning" in ins:
                for w in ins["incompatible_warning"]:
                    lines.append(f"⚠️  {w}")

        if result.get("safety"):
            saf = result["safety"]
            kgw = saf.get("kg_warnings", [])
            if kgw:
                lines.append(f"\n🔒 安全提示：")
                for w in kgw:
                    lines.append(f"  {w}")
            if saf.get("warnings"):
                lines.append(f"\n📝 用药注意：")
                for w in saf["warnings"]:
                    lines.append(f"  {w}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
