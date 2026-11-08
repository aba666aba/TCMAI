# -*- coding: utf-8 -*-
"""确定性答案合成器（解读生成）。

把"智能搜索/推荐结果"合成为一段连贯、有据可查的中文解读。
- 全部内容来自知识库真实字段，不编造 → 零幻觉
- 纯离线、零外部大模型依赖
- 实体详情（组成/功用/功效）直接按名称查库获取，确定且正确
- 安全提示与免责声明内建，符合医学辅助工具定位

输入：
  query_text   : 用户原始问句
  symptoms     : 已识别症状列表
  intel_result : core.intelligent_search.IntelligentSearch.search() 的返回
  kg_paths     : 可选，KGEnhancedRecommender 的 reasoning_paths
  safety       : 可选，推荐模型 check_safety 的结果（含 incompatibilities/toxic_herbs）
"""
import os
import re
import sqlite3
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DB_PATH = os.path.join(ROOT, "data", "tcm_knowledge.db")

_CITE = re.compile(r"[①-⑩]")  # 典籍引文圈码 ①-⑩


def _clean(text: str, max_len: int = 46) -> str:
    """清理字段用于展示：去掉典籍引文（圈码起），截断到合理长度。"""
    if not text:
        return ""
    text = str(text)
    m = _CITE.search(text)
    if m:
        text = text[: m.start()]
    text = text.strip("；;。. ，,　 ")
    if len(text) > max_len:
        text = text[:max_len].rstrip("；;，,。. ，,　 ") + "…"
    return text


def _entity_fields(name: str, rtype: str) -> dict:
    """按名称直接从知识库取实体详情（组成/功用/功效等）。

    用精确查库而非 BM25，避免被知识图谱关系文档干扰，保证正确。
    """
    if not name or not os.path.exists(_DB_PATH):
        return {}
    try:
        con = sqlite3.connect(_DB_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        if rtype == "formula":
            row = cur.execute(
                "SELECT composition, functions, indications FROM formulas WHERE name=?",
                (name,),
            ).fetchone()
        elif rtype == "herb":
            row = cur.execute(
                "SELECT efficacy, indications, usage FROM herbs WHERE name=?",
                (name,),
            ).fetchone()
        elif rtype == "patent":
            row = cur.execute(
                "SELECT composition, functions, indications FROM patent_medicines WHERE name=?",
                (name,),
            ).fetchone()
            if row is None:
                # 别名解析：patent_aliases（同药异名/异剂型互通）
                std = cur.execute(
                    "SELECT standard FROM patent_aliases WHERE alias=?", (name,)
                ).fetchone()
                if std:
                    row = cur.execute(
                        "SELECT composition, functions, indications FROM patent_medicines WHERE name=?",
                        (std["standard"],),
                    ).fetchone()
        else:
            row = None
        con.close()
        return dict(row) if row else {}
    except Exception:
        return {}


def synthesize(
    query_text: str,
    symptoms: list[str],
    intel_result: dict[str, Any],
    kg_paths: list[dict] | None = None,
    safety: dict[str, Any] | None = None,
) -> str:
    """合成一段"AI 解读"。返回纯文本；调用方负责拼接到最终回答并保留免责声明。"""
    parts: list[str] = []
    syms = list(intel_result.get("symptoms") or symptoms or [])

    parts.append("【AI 解读】")
    if syms:
        parts.append(
            f"根据您描述的症状（{'、'.join(syms)}），系统从中医药知识库中"
            f"检索并整合了以下信息，供您参考。"
        )
    else:
        parts.append(
            "根据您的描述，系统从中医药知识库中检索并整合了以下信息，供您参考。"
        )
    parts.append("📌 药材/方剂的完整推荐清单与用量见「推荐结果」页；本页侧重辨证思路与组方草案。")

    # 证候倾向
    syndromes = intel_result.get("inferred_syndromes") or []
    if syndromes:
        top = syndromes[0]
        score = top.get("score")
        score_txt = f"（匹配度约 {score:.0%}）" if isinstance(score, (int, float)) else ""
        desc = (top.get("description") or "").strip()
        parts.append("\n【证候倾向】")
        parts.append(f"综合症状，较符合「{top['syndrome']}」的证候特点{score_txt}。")
        if desc:
            parts.append(f"{desc}")
        if len(syndromes) > 1:
            others = "、".join(s["syndrome"] for s in syndromes[1:3])
            parts.append(f"此外也可能涉及：{others}。")

    # 方剂/药材：点评式概述（完整清单在「推荐结果」页，此处不重复罗列）
    formulas = intel_result.get("recommended_formulas") or []
    if formulas:
        top = formulas[0]
        name = top.get("name", "")
        fld = _entity_fields(name, "formula")
        func = _clean(fld.get("functions"), 40)
        line = f"针对该证候，系统筛选出 {min(len(formulas), 5)} 首候选方剂"
        if name:
            line += f"，其中「{name}」与证候最为契合"
            if func:
                line += f"（{func}）"
        parts.append("\n【方剂要点】")
        parts.append(line + "。其余备选及组成详见「推荐结果」页。")

    herbs = intel_result.get("herbs") or []
    if herbs:
        names = [h.get("name", "") for h in herbs[:6] if h.get("name")]
        if names:
            core = "、".join(names[:3])
            parts.append("\n【药材要点】")
            parts.append(
                f"配伍思路围绕 {len(names)} 味核心药材展开（{core} 等）；"
                "逐味功效、用量与禁忌请见「推荐结果」页。"
            )

    patents = intel_result.get("patent_medicines") or []
    if patents:
        parts.append(
            f"\n【相关中成药】{'、'.join(p.get('name', '') for p in patents[:4])}。"
        )

    # 安全提示与辨证路径由「推荐结果」页与【知识图谱辨证】块负责，此处不再重复。

    # 日常调护
    try:
        from core.tcm_config import generate_health_tips

        tips = generate_health_tips(set(syms))
    except Exception:
        tips = []
    if tips:
        parts.append("\n【日常调护】")
        parts.extend(tips)

    parts.append(
        "\n⚠️ 以上内容不构成诊断或处方，用药请务必咨询执业中医师。"
    )
    return "\n".join(parts)


def main():
    import sys

    sys.path.insert(0, os.path.join(ROOT, "src"))
    from core.knowledge_base import KnowledgeBase
    from core.intelligent_search import IntelligentSearch

    q = sys.argv[1] if len(sys.argv) > 1 else "失眠 乏力 头晕"
    kb = KnowledgeBase()
    searcher = IntelligentSearch(kb)
    intel = searcher.search(q)
    print(synthesize(q, intel.get("symptoms", []), intel))


if __name__ == "__main__":
    main()
