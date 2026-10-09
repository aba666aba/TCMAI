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
"""TCM 知识图谱查询接口 — 实体/关系/推理查询、最短路径分析"""

import logging

import networkx as nx

logger = logging.getLogger(__name__)


class KnowledgeGraphQuery:
    """知识图谱查询接口"""

    def __init__(self, graph):
        self.graph = graph

    # 实体查询

    def find_nodes(
        self, name: str | None = None, node_type: str | None = None
    ) -> list[str]:
        results = []
        for nid in self.graph.nodes:
            data = self.graph.nodes[nid]
            if node_type and data.get("type") != node_type:
                continue
            if name:
                if name in data.get("name", "") or name in nid:
                    results.append(nid)
            else:
                results.append(nid)
        return results

    def get_node_info(self, node_id: str) -> dict | None:
        """获取节点详细信息"""
        if not self.graph.has_node(node_id):
            return None
        data = dict(self.graph.nodes[node_id])
        data["id"] = node_id
        return data

    # 关系查询

    def get_neighbors(
        self, node_id: str, edge_type: str | None = None
    ) -> list[tuple[str, str, dict]]:
        results = []
        for _, neighbor, data in self.graph.edges(node_id, data=True):
            if edge_type and data.get("type") != edge_type:
                continue
            results.append((neighbor, data.get("type", ""), data))
        return results

    def get_degree(self, node_id: str) -> int:
        """获取节点度数"""
        return self.graph.degree(node_id) if self.graph.has_node(node_id) else 0

    # 推理查询

    def symptom_to_recommendation(self, symptom_name: str, top_k: int = 10) -> dict:
        """
        症状→推荐（核心推理链路）

        输入症状名 → 找到关联药材/方剂 → 找到证型 → 返回综合推荐
        """
        snid = f"symptom:{symptom_name}"
        if not self.graph.has_node(snid):
            return {"error": f"未找到症状: {symptom_name}"}

        result = {
            "symptom": symptom_name,
            "syndromes": [],
            "herbs": [],
            "formulas": [],
        }

        # 1. 查证型
        for _, nbr, data in self.graph.edges(snid, data=True):
            if data.get("type") == "belongs_to":
                result["syndromes"].append(self._node_name(nbr))

        # 2. 查关联药材（按权重排序）
        herb_scores = []
        for _, nbr, data in self.graph.edges(snid, data=True):
            if data.get("type") == "associated_with_herb":
                herb_scores.append((self._node_name(nbr), data.get("weight", 0)))
        herb_scores.sort(key=lambda x: -x[1])
        result["herbs"] = [{"name": h, "score": s} for h, s in herb_scores[:top_k]]

        # 3. 查关联方剂
        formula_scores = []
        for _, nbr, data in self.graph.edges(snid, data=True):
            if data.get("type") == "associated_with_formula":
                formula_scores.append((self._node_name(nbr), data.get("weight", 0)))
        formula_scores.sort(key=lambda x: -x[1])
        result["formulas"] = [
            {"name": f, "score": s} for f, s in formula_scores[:top_k]
        ]

        return result

    def syndrome_analysis(self, symptom_names: list[str]) -> dict:
        """
        多症状综合分析 → 推断证型

        输入一组症状 → 找到共同证型 → 推荐药材/方剂
        """
        # 收集所有症状关联的证型
        syndrome_votes = {}
        for sym_name in symptom_names:
            snid = f"symptom:{sym_name}"
            if not self.graph.has_node(snid):
                continue
            for _, nbr, data in self.graph.edges(snid, data=True):
                if data.get("type") == "belongs_to":
                    syn_name = self._node_name(nbr)
                    syndrome_votes[syn_name] = syndrome_votes.get(syn_name, 0) + 1

        # 按匹配症状数排序
        sorted_syns = sorted(syndrome_votes.items(), key=lambda x: -x[1])

        result = {
            "input_symptoms": symptom_names,
            "matched_syndromes": [],
        }

        for syn_name, votes in sorted_syns:
            syn_info = {"name": syn_name, "matched_symptom_count": votes}
            # 获取证型描述
            syn_nid = f"syndrome:{syn_name}"
            if self.graph.has_node(syn_nid):
                data = self.graph.nodes[syn_nid]
                syn_info["description"] = data.get("description", "")
                syn_info["treatment"] = data.get("treatment", "")

            # 获取推荐药材
            herbs = []
            for _, nbr, edata in self.graph.edges(syn_nid, data=True):
                if edata.get("type") == "recommends_herb":
                    herbs.append(self._node_name(nbr))
            syn_info["recommended_herbs"] = herbs

            # 获取推荐方剂
            formulas = []
            for _, nbr, edata in self.graph.edges(syn_nid, data=True):
                if edata.get("type") == "recommends_formula":
                    formulas.append(self._node_name(nbr))
            syn_info["recommended_formulas"] = formulas

            result["matched_syndromes"].append(syn_info)

        return result

    def herb_profile(self, herb_name: str) -> dict:
        """药材完整画像"""
        hnid = f"herb:{herb_name}"
        if not self.graph.has_node(hnid):
            return {"error": f"未找到药材: {herb_name}"}

        data = self.graph.nodes[hnid]
        profile = {
            "name": herb_name,
            "category": data.get("category", ""),
            "efficacy": data.get("efficacy", ""),
            "indications": data.get("indications", ""),
            "usage": data.get("usage", ""),
            "contraindications": data.get("contraindications", ""),
            "nature": [],
            "flavor": [],
            "meridian": [],
            "in_formulas": [],
            "associated_symptoms": [],
            "co_occurring_herbs": [],
            "incompatible_with": [],
            "toxic": data.get("toxic", False),
        }

        for _, nbr, edata in self.graph.edges(hnid, data=True):
            etype = edata.get("type", "")
            if etype == "has_nature":
                profile["nature"].append(self._node_name(nbr))
            elif etype == "has_flavor":
                profile["flavor"].append(self._node_name(nbr))
            elif etype == "has_meridian":
                profile["meridian"].append(self._node_name(nbr))
            elif etype == "composes":
                profile["in_formulas"].append(self._node_name(nbr))
            elif etype == "co_occurs_with":
                profile["co_occurring_herbs"].append(
                    {
                        "name": self._node_name(nbr),
                        "strength": edata.get("weight", 0),
                    }
                )
            elif etype == "incompatible_with":
                profile["incompatible_with"].append(self._node_name(nbr))

        # 收集关联症状
        for nid in self.graph.nodes:
            if self.graph.nodes[nid].get("type") != "Symptom":
                continue
            for _, nbr2, edata2 in self.graph.edges(nid, data=True):
                if nbr2 == hnid and edata2.get("type") == "associated_with_herb":
                    profile["associated_symptoms"].append(
                        {
                            "name": self._node_name(nid),
                            "weight": edata2.get("weight", 0),
                        }
                    )

        # 排序
        profile["co_occurring_herbs"].sort(key=lambda x: -x["strength"])
        profile["co_occurring_herbs"] = profile["co_occurring_herbs"][:20]
        profile["associated_symptoms"].sort(key=lambda x: -x["weight"])

        return profile

    def shortest_path(self, node_a: str, node_b: str) -> list[str] | None:
        """最短路径（基于无权图）"""
        if not self.graph.has_node(node_a) or not self.graph.has_node(node_b):
            return None
        try:
            path = nx.shortest_path(self.graph, node_a, node_b)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def recommend_by_symptoms(self, symptoms: list[str], top_k: int = 10) -> dict:
        """
        综合推荐：多症状 → 多证型 → 综合药材+方剂
        """
        # 先做证型分析
        syn_result = self.syndrome_analysis(symptoms)

        # 收集所有药材（去重加权）
        herb_scores = {}
        formula_scores = {}
        for syn in syn_result["matched_syndromes"]:
            weight = syn["matched_symptom_count"] / len(symptoms)
            for h in syn.get("recommended_herbs", []):
                herb_scores[h] = herb_scores.get(h, 0) + weight
            for f in syn.get("recommended_formulas", []):
                formula_scores[f] = formula_scores.get(f, 0) + weight

        # 从症状直接关联补充
        for sym in symptoms:
            rec = self.symptom_to_recommendation(sym)
            for h in rec.get("herbs", []):
                herb_scores[h["name"]] = herb_scores.get(h["name"], 0) + 0.3
            for f in rec.get("formulas", []):
                formula_scores[f["name"]] = formula_scores.get(f["name"], 0) + 0.3

        return {
            "symptoms": symptoms,
            "syndromes": syn_result["matched_syndromes"],
            "recommended_herbs": sorted(
                [{"name": k, "score": v} for k, v in herb_scores.items()],
                key=lambda x: -x["score"],
            )[:top_k],
            "recommended_formulas": sorted(
                [{"name": k, "score": v} for k, v in formula_scores.items()],
                key=lambda x: -x["score"],
            )[:top_k],
        }

    # 内部工具

    def _node_name(self, node_id: str) -> str:
        """从节点ID提取名称"""
        data = self.graph.nodes.get(node_id)
        if data:
            return data.get("name", node_id.split(":", 1)[-1])
        return node_id.split(":", 1)[-1]
