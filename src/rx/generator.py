# -*- coding: utf-8 -*-
"""智能组方引擎 — 根据症状与药材药性生成「AI 草方」（君臣佐使）。

对应产品需求：
  1. 生成依据是药材的**药性数据**（四气五味/归经/功效/主治，全部来自知识库，
     零幻觉：不发明任何药材属性，方子的主治由组成药材的功效自然涌现）；
  2. 生成过程**规避全部严格配伍禁忌**（十八反/十九畏权威药对表，选药时即
     约束，组成后再次审计）；
  3. 毒性药材一律不进入草方（与推荐器安全硬拦截同标准）；
  4. 同步给出**已入库经典方剂**作为备选——医师判断草方不可用时直接取用；
  5. 输出始终标注「供执业中医师审核」，AI 不代定剂量。

不依赖 torch：候选分数由调用方（模型+KG 融合）传入，本引擎只负责
数据驱动的组方逻辑，可在任何环境独立运行。
"""

import logging
import re
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT = Path(__file__).resolve().parents[2]

# 常用调和类使药（惯用药；仅当其在候选中出现时才会被选为使药）
_ENVOY_HERBS = ("甘草", "大枣", "生姜")

# 组成规模：经典方剂常规 4-8 味，默认 6（1君 + ≤2臣 + ≤2佐 + ≤1使）
_DEFAULT_MAX_HERBS = 6


class FormulaGenerator:
    """药性驱动的组方引擎（纯 Python + SQLite，无 torch 依赖）。"""

    def __init__(self, db_path=None):
        self.db_path = str(db_path or (_PROJECT / "data" / "tcm_knowledge.db"))
        self._herbs = {}       # name -> {nature, flavor, meridian, efficacy, indications, usage}
        self._syndromes = {}   # name -> {description, treatment, formula}
        self._load()

    def _load(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        for r in cur.execute(
            "SELECT name, nature, flavor, meridian, efficacy, indications, usage "
            "FROM herbs"
        ):
            self._herbs[r["name"]] = dict(r)
        for r in cur.execute(
            "SELECT name, description, treatment, formula FROM syndrome_definitions"
        ):
            self._syndromes[r["name"]] = dict(r)
        con.close()
        logger.info(
            "组方引擎数据就绪: %d 味药材 / %d 条证型定义",
            len(self._herbs), len(self._syndromes),
        )

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------

    def generate(
        self,
        symptoms,
        herb_candidates,
        syndrome_name=None,
        matched_syndromes=None,
        alternative_formulas=None,
        max_herbs=_DEFAULT_MAX_HERBS,
    ) -> dict:
        """生成 AI 草方。

        Args:
            symptoms: list[str] 输入症状
            herb_candidates: list[{name, score}] 模型+KG 融合候选（带分数）
            syndrome_name: 主证型名（可选）
            matched_syndromes: KG 匹配证型 [{name, description, ...}]（可选）
            alternative_formulas: 已入库方剂名列表（可选，作为医师备选）
            max_herbs: 草方组成上限（含使药）

        Returns:
            {text, composition, roles, conflicts, skipped, dropped_toxic}
        """
        from tcm_recommender.extended_config import is_toxic_herb, pair_conflicts

        symptoms = [s for s in (symptoms or []) if s]

        # ---- 1) 候选清洗：毒性硬剔除 + 去重 + 库内存在性校验 ----
        dropped_toxic, cand = [], {}
        for c in herb_candidates or []:
            name = (c.get("name") or "").strip()
            if not name or name not in self._herbs:
                continue
            if is_toxic_herb(name):
                dropped_toxic.append(name)
                continue
            s = float(c.get("score") or 0.0)
            if name not in cand or s > cand[name]["score"]:
                cand[name] = {"name": name, "score": s}
        if not cand:
            return {
                "text": "", "composition": [], "roles": {}, "conflicts": [],
                "skipped": [], "dropped_toxic": sorted(set(dropped_toxic)),
                "error": "无有效候选药材",
            }

        # ---- 2) 证型/治法解析（库内定义优先 → KG 匹配回查 → 留空降级）----
        syn, syn_desc = self._resolve_syndrome(syndrome_name, matched_syndromes)
        treatment = syn.get("treatment") or ""
        kw = self._treatment_keywords(treatment)

        # ---- 3) 打分：融合分（0.8，max 归一）+ 治法功效匹配（0.3/词）+ 主治症状匹配（0.1/词）----
        max_s = max((c["score"] for c in cand.values()), default=1.0) or 1.0
        for c in cand.values():
            h = self._herbs[c["name"]]
            eff, ind = h.get("efficacy") or "", h.get("indications") or ""
            bonus = 0.0
            if kw:
                bonus += 0.30 * sum(1 for k in kw if k in eff)
            bonus += 0.10 * sum(1 for s in symptoms if s and s in ind)
            c["final"] = 0.8 * (c["score"] / max_s) + bonus
        ranked = sorted(cand.values(), key=lambda x: -x["final"])

        # ---- 4) 约束感知选药（预留 1 个使药名额；毒性已在步骤 1 剔除）----
        selected, skipped = [], []
        core_quota = max(max_herbs - 1, 1)
        for c in ranked:
            if len(selected) >= core_quota:
                break
            hits = pair_conflicts(c["name"], selected)
            if hits:
                skipped.append((c["name"], hits))
                continue
            selected.append(c["name"])
        if not selected:
            return {
                "text": "", "composition": [], "roles": {}, "conflicts": [],
                "skipped": skipped, "dropped_toxic": sorted(set(dropped_toxic)),
                "error": "候选药材均存在配伍冲突，无法组方",
            }

        # ---- 5) 君臣佐使分派（君=最高分；臣=同功效家族辅君；佐=异家族兼顾；使=调和药）----
        roles = {"君": [], "臣": [], "佐": [], "使": []}
        roles["君"].append(selected[0])
        jun_fam = self._eff_family(self._herbs[selected[0]].get("efficacy"))
        for name in selected[1:]:
            fam = self._eff_family(self._herbs[name].get("efficacy"))
            if fam == jun_fam and len(roles["臣"]) < 2:
                roles["臣"].append(name)
            else:
                roles["佐"].append(name)
        for c in ranked:  # 使药：惯用调和药优先
            if c["name"] in selected or len(selected) >= max_herbs:
                continue
            if c["name"] in _ENVOY_HERBS and not pair_conflicts(c["name"], selected):
                selected.append(c["name"])
                roles["使"].append(c["name"])
                break

        # ---- 6) 组成审计（应恒为空，双重保险）----
        final_conflicts = []
        for a in selected:
            for o in pair_conflicts(a, [x for x in selected if x != a]):
                pair = " + ".join(sorted([a, o]))
                if pair not in final_conflicts:
                    final_conflicts.append(pair)

        text = self._format(
            symptoms, syn, syn_desc, treatment, roles, selected,
            final_conflicts, skipped, dropped_toxic, alternative_formulas,
        )
        return {
            "text": text,
            "composition": selected,
            "roles": roles,
            "conflicts": final_conflicts,
            "skipped": skipped,
            "dropped_toxic": sorted(set(dropped_toxic)),
        }

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _resolve_syndrome(self, syndrome_name, matched_syndromes):
        names = [syndrome_name] + [
            (m.get("name") if isinstance(m, dict) else m)
            for m in (matched_syndromes or [])
        ]
        for name in names:
            if name and name in self._syndromes:
                d = self._syndromes[name]
                return dict(d), (d.get("description") or "")
        for m in (matched_syndromes or []):
            if isinstance(m, dict) and m.get("name"):
                return (
                    {"name": m["name"], "treatment": "", "formula": ""},
                    (m.get("description") or ""),
                )
        if syndrome_name:
            return {"name": syndrome_name, "treatment": "", "formula": ""}, ""
        return {"name": "", "treatment": "", "formula": ""}, ""

    @staticmethod
    def _treatment_keywords(treatment: str) -> list[str]:
        t = (treatment or "").replace("，", "").replace(",", "").strip()
        if not t:
            return []
        kws = [t] if len(t) >= 4 else []
        for i in range(0, len(t) - 1, 2):
            kws.append(t[i : i + 2])
        return [k for k in dict.fromkeys(kws) if k]

    @staticmethod
    def _eff_family(efficacy: str) -> str:
        return (efficacy or "").strip()[:2]

    @staticmethod
    def _short(text: str, n: int = 24) -> str:
        """截短药性文本；剥离 ①②③ 等文献引文段，只留正文。"""
        text = re.split(r"[①②③④⑤⑥⑦⑧⑨⑩]", (text or ""))[0].strip().rstrip("。；;，,")
        return text[:n] + ("…" if len(text) > n else "")

    @staticmethod
    def _clean_prop(text: str) -> str:
        """清理药性属性字段（性味/归经）：去引文段与首尾标点。"""
        t = re.split(r"[①②③④⑤⑥⑦⑧⑨⑩]", (text or ""))[0].strip()
        return t.strip("。，,;；、 ")

    def _build_fangjie(self, roles, treatment) -> str:
        parts = []
        jun = roles["君"][0]
        parts.append(
            f"以{jun}为君，取其{self._short(self._herbs[jun].get('efficacy'), 14)}，直指主证"
        )
        for n in roles["臣"]:
            parts.append(f"{n}为臣，助君攻邪补正")
        for n in roles["佐"]:
            parts.append(f"{n}为佐，兼顾兼证")
        if roles["使"]:
            parts.append(f"{roles['使'][0]}为使，调和诸药")
        text = "；".join(parts) + "。"
        if treatment:
            text += f"全方合「{treatment}」之治法。"
        return text

    def _format(self, symptoms, syn, syn_desc, treatment, roles, selected,
                conflicts, skipped, dropped_toxic, alternative_formulas) -> str:
        lines = ["【AI 组方草案 · 供执业中医师审核】"]

        head = []
        if syn.get("name"):
            head.append(f"辨证: {syn['name']}")
        if syn_desc:
            head.append(syn_desc)
        if treatment:
            head.append(f"治法: {treatment}")
        if head:
            lines.append("｜".join(head))
            lines.append("")

        lines.append("拟方组成（君臣佐使）:")
        for role in ("君", "臣", "佐", "使"):
            for name in roles[role]:
                h = self._herbs[name]
                xy = "，".join(
                    x for x in (
                        self._clean_prop(h.get("nature")),
                        self._clean_prop(h.get("flavor")),
                    ) if x
                )
                seg = f"  {role} · {name}"
                if xy:
                    seg += f"（{xy}；归经: {self._clean_prop(h.get('meridian')) or '—'}）"
                eff = self._short(h.get("efficacy"), 26)
                if eff:
                    seg += f"｜功效: {eff}"
                lines.append(seg)

        lines.append("")
        lines.append(f"方解: {self._build_fangjie(roles, treatment)}")

        zhi = "、".join(symptoms) or "（未提供症状）"
        line = f"拟方主治: {zhi}"
        if syn.get("name"):
            line += f"（证属{syn['name']}）"
        lines.append(line)
        lines.append("剂量: AI 不代定剂量；各药常规用法用量请参见知识库，由医师酌定。")

        audit = "配伍审计: 组成内未发现十八反/十九畏禁忌药对 ✅"
        if conflicts:
            audit = f"配伍审计: ⚠️ 发现 {len(conflicts)} 组禁忌药对: {'、'.join(conflicts)}"
        notes = []
        if dropped_toxic:
            notes.append("已剔除毒性药: " + "、".join(dropped_toxic))
        if skipped:
            notes.append("已避开: " + "、".join(
                f"{n}（与{'/'.join(hs)}相反）" for n, hs in skipped
            ))
        if notes:
            audit += "；" + "；".join(notes)
        lines.append(audit)

        alts = []
        if syn.get("formula"):
            alts.append(syn["formula"])
        for a in alternative_formulas or []:
            if a:
                alts.append(a)
        alts = [a for a in dict.fromkeys(alts) if a]
        if alts:
            lines.append(
                f"备选已入库方剂（医师判断草方不适用时可用）: {'、'.join(alts[:5])}"
            )

        lines.append("")
        lines.append(
            "⚠️ 本草方由算法根据知识库药材药性自动生成，仅供参考，必须经执业中医师审核后方可考虑使用；不可直接用于患者。"
        )
        return "\n".join(lines)
