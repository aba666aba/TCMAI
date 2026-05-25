"""
TCM 知识图谱构建器

从 database + config + pipeline 抽取实体和关系，
构建 NetworkX 有向图。
"""

import logging
import sqlite3
import re
from pathlib import Path
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)

# 项目路径
_PROJECT = Path(__file__).resolve().parents[2]


class TCMKnowledgeGraph:
    """TCM 知识图谱构建器"""

    # 实体类型
    SYMPTOM = "Symptom"
    SYNDROME = "Syndrome"
    HERB = "Herb"
    FORMULA = "Formula"
    PATENT_MEDICINE = "PatentMedicine"
    FOLK_REMEDY = "FolkRemedy"
    CATEGORY = "Category"
    MERIDIAN = "Meridian"
    NATURE = "Nature"
    FLAVOR = "Flavor"

    # 关系类型
    BELONGS_TO = "belongs_to"  # 症状→证型
    RECOMMENDS_HERB = "recommends_herb"  # 证型→药材
    RECOMMENDS_FORMULA = "recommends_formula"  # 证型→方剂
    ASSOC_HERB = "associated_with_herb"  # 症状→药材
    ASSOC_FORMULA = "associated_with_formula"  # 症状→方剂
    COMPOSES = "composes"  # 药材→方剂
    HAS_NATURE = "has_nature"  # 药材→性
    HAS_FLAVOR = "has_flavor"  # 药材→味
    HAS_MERIDIAN = "has_meridian"  # 药材→归经
    BELONGS_CAT = "belongs_to_category"  # 药材→分类
    INCOMPATIBLE = "incompatible_with"  # 药材↔药材
    TOXIC = "toxic"  # 药材→标记
    CO_OCCURS = "co_occurs_with"  # 药材↔药材
    TREATS = "treats"  # 方剂→症状（从主治）

    def __init__(self, db_path: Optional[str] = None):
        self.graph = nx.MultiDiGraph()
        self.db_path = db_path or str(_PROJECT / "data" / "tcm_knowledge.db")
        self._db = sqlite3.connect(self.db_path)

        # 计数
        self.stats = {
            "nodes": {},
            "edges": {},
        }

    # ═══════════════════════════════════════════
    # 构建入口
    # ═══════════════════════════════════════════

    def build(self) -> "TCMKnowledgeGraph":
        """构建完整知识图谱"""
        logger.info("=" * 50)
        logger.info("TCM 知识图谱构建开始")

        # 先建所有节点，再建边（避免推荐边引用不存在的节点）
        self._add_symptoms_and_syndromes_nodes()
        self._add_herbs()
        self._add_formulas()
        self._add_patent_medicines()
        self._add_folk_remedies()
        self._add_herb_attributes()
        # 所有节点就绪后，再建边
        self._add_syndrome_recommendations()
        # 构建一次pipeline，各方法共享
        import sys

        sys.path.insert(0, str(_PROJECT / "src"))
        sys.path.insert(0, str(_PROJECT))
        from tcm_recommender.data_pipeline import TCMDataPipeline

        pipeline = TCMDataPipeline(self.db_path)
        pipeline.load_all()

        self._add_symptom_herb_associations(pipeline)
        self._add_formula_compositions(pipeline)
        self._add_incompatibilities()
        self._add_cooccurrence(pipeline)

        logger.info("=" * 50)
        logger.info(
            "构建完成: %s 节点, %s 边",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self

    # ═══════════════════════════════════════════
    # 1. 证型 + 症状
    # ═══════════════════════════════════════════

    def _add_symptoms_and_syndromes_nodes(self):
        """添加症状和证型节点（不含推荐边）"""
        import sys

        sys.path.insert(0, str(_PROJECT / "src"))
        sys.path.insert(0, str(_PROJECT))
        from tcm_recommender.extended_config import (
            EXTENDED_SYMPTOM_LIST,
            EXTENDED_SYNDROME_DEFINITIONS,
        )

        # 症状节点
        for idx, sym_name in enumerate(EXTENDED_SYMPTOM_LIST):
            self.graph.add_node(
                f"symptom:{sym_name}",
                type=self.SYMPTOM,
                name=sym_name,
                id=idx,
            )
        self.stats["nodes"]["Symptom"] = len(EXTENDED_SYMPTOM_LIST)

        # 证型节点 + 症状→证型边
        for syn_name, syn_def in EXTENDED_SYNDROME_DEFINITIONS.items():
            self.graph.add_node(
                f"syndrome:{syn_name}",
                type=self.SYNDROME,
                name=syn_name,
                description=syn_def.get("description", ""),
                treatment=syn_def.get("treatment", ""),
            )

            # 症状→证型（节点已存在）
            for sym in syn_def.get("symptoms", []):
                if self.graph.has_node(f"symptom:{sym}"):
                    self.graph.add_edge(
                        f"symptom:{sym}",
                        f"syndrome:{syn_name}",
                        type=self.BELONGS_TO,
                        label="属于",
                    )

        self.stats["nodes"]["Syndrome"] = len(EXTENDED_SYNDROME_DEFINITIONS)

    def _add_syndrome_recommendations(self):
        """在所有节点就绪后添加证型推荐边"""
        import sys

        sys.path.insert(0, str(_PROJECT / "src"))
        sys.path.insert(0, str(_PROJECT))
        from tcm_recommender.extended_config import EXTENDED_SYNDROME_DEFINITIONS

        for syn_name, syn_def in EXTENDED_SYNDROME_DEFINITIONS.items():
            # 证型→推荐药材
            for herb_name in syn_def.get("recommend_herbs", []):
                nid = f"herb:{herb_name}"
                if self.graph.has_node(nid):
                    self.graph.add_edge(
                        f"syndrome:{syn_name}",
                        nid,
                        type=self.RECOMMENDS_HERB,
                        label="推荐药材",
                    )

            # 证型→推荐方剂
            for fml_name in syn_def.get("recommend_formulas", []):
                nid = f"formula:{fml_name}"
                if self.graph.has_node(nid):
                    self.graph.add_edge(
                        f"syndrome:{syn_name}",
                        nid,
                        type=self.RECOMMENDS_FORMULA,
                        label="推荐方剂",
                    )

    # ═══════════════════════════════════════════
    # 2. 药材
    # ═══════════════════════════════════════════

    def _add_herbs(self):
        """从数据库添加药材节点"""
        conn = self._db
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, category, efficacy, indications, "
            "nature, flavor, meridian, usage, contraindications, source "
            "FROM herbs ORDER BY id"
        )
        rows = cur.fetchall()
        for row in rows:
            hid, name, cat, eff, ind, nat, flv, mer, usage, contra, src = row
            self.graph.add_node(
                f"herb:{name}",
                type=self.HERB,
                name=name,
                db_id=hid,
                category=cat or "",
                efficacy=(eff or "")[:100],
                indications=(ind or "")[:100],
                nature=nat or "",
                flavor=flv or "",
                meridian=mer or "",
                usage=usage or "",
                contraindications=contra or "",
                source=src or "",
            )
            # 分类节点
            if cat and cat.strip():
                cat_clean = cat.strip()
                if not self.graph.has_node(f"category:{cat_clean}"):
                    self.graph.add_node(
                        f"category:{cat_clean}",
                        type=self.CATEGORY,
                        name=cat_clean,
                    )
                self.graph.add_edge(
                    f"herb:{name}",
                    f"category:{cat_clean}",
                    type=self.BELONGS_CAT,
                    label="属于分类",
                )
        self.stats["nodes"]["Herb"] = len(rows)

    # ═══════════════════════════════════════════
    # 3. 方剂
    # ═══════════════════════════════════════════

    def _add_formulas(self):
        """从数据库添加方剂节点"""
        conn = self._db
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, category, source_book, composition, "
            "dosage, functions, indications, usage, modification "
            "FROM formulas ORDER BY id"
        )
        rows = cur.fetchall()
        for row in rows:
            fid, name, cat, src, comp, dose, func, ind, usage, mod = row
            self.graph.add_node(
                f"formula:{name}",
                type=self.FORMULA,
                name=name,
                db_id=fid,
                category=cat or "",
                source_book=src or "",
                composition=comp or "",
                dosage=dose or "",
                functions=func or "",
                indications=ind or "",
                usage=usage or "",
                modification=mod or "",
            )
        self.stats["nodes"]["Formula"] = len(rows)

    # ═══════════════════════════════════════════
    # 4. 中成药
    # ═══════════════════════════════════════════

    def _add_patent_medicines(self):
        conn = self._db
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, category, composition, functions, "
            "indications, usage_dosage, contraindications, source "
            "FROM patent_medicines ORDER BY id"
        )
        rows = cur.fetchall()
        for row in rows:
            pid, name, cat, comp, func, ind, usage, contra, src = row
            self.graph.add_node(
                f"patent:{name}",
                type=self.PATENT_MEDICINE,
                name=name,
                db_id=pid,
                category=cat or "",
                composition=comp or "",
                functions=func or "",
                indications=ind or "",
                usage=usage or "",
                contraindications=contra or "",
            )
        self.stats["nodes"]["PatentMedicine"] = len(rows)

    # ═══════════════════════════════════════════
    # 5. 验方
    # ═══════════════════════════════════════════

    def _add_folk_remedies(self):
        conn = self._db
        cur = conn.cursor()
        cur.execute(
            "SELECT id, disease, remedy, ingredients, method, source "
            "FROM folk_remedies ORDER BY id"
        )
        rows = cur.fetchall()
        for row in rows:
            rid, disease, remedy, ingr, method, src = row
            self.graph.add_node(
                f"folk:{disease}:{rid}",
                type=self.FOLK_REMEDY,
                name=disease,
                db_id=rid,
                disease=disease,
                remedy=remedy or "",
                ingredients=ingr or "",
                method=method or "",
                source=src or "",
            )
        self.stats["nodes"]["FolkRemedy"] = len(rows)

    # ═══════════════════════════════════════════
    # 6. 药材属性（性味归经）
    # ═══════════════════════════════════════════

    def _add_herb_attributes(self):
        """从药材的 nature/flavor/meridian 字段提取属性节点"""
        # 基础性味列表（从文本中提取）
        nature_values = {"寒", "热", "温", "凉", "平", "微寒", "微温", "大寒", "大热"}
        flavor_values = {"酸", "苦", "甘", "辛", "咸", "淡", "涩"}
        meridian_names = {
            "心",
            "肝",
            "脾",
            "肺",
            "肾",
            "胆",
            "胃",
            "大肠",
            "小肠",
            "膀胱",
            "三焦",
            "心包",
        }

        for nid in list(self.graph.nodes):
            if self.graph.nodes[nid].get("type") != self.HERB:
                continue
            data = self.graph.nodes[nid]

            # 性
            nat_text = data.get("nature", "")
            for nv in nature_values:
                if nv in nat_text:
                    nn = f"nature:{nv}"
                    if not self.graph.has_node(nn):
                        self.graph.add_node(nn, type=self.NATURE, name=nv)
                    self.graph.add_edge(nid, nn, type=self.HAS_NATURE, label=nv)
                    break  # 取第一个匹配

            # 味
            flv_text = data.get("flavor", "")
            for fv in flavor_values:
                if fv in flv_text:
                    fn = f"flavor:{fv}"
                    if not self.graph.has_node(fn):
                        self.graph.add_node(fn, type=self.FLAVOR, name=fv)
                    self.graph.add_edge(nid, fn, type=self.HAS_FLAVOR, label=fv)

            # 归经
            mer_text = data.get("meridian", "")
            for mv in meridian_names:
                if mv in mer_text:
                    mn = f"meridian:{mv}"
                    if not self.graph.has_node(mn):
                        self.graph.add_node(mn, type=self.MERIDIAN, name=mv)
                    self.graph.add_edge(nid, mn, type=self.HAS_MERIDIAN, label=mv)

        self.stats["nodes"]["Nature"] = len(
            [
                n
                for n in self.graph.nodes
                if self.graph.nodes[n].get("type") == self.NATURE
            ]
        )
        self.stats["nodes"]["Flavor"] = len(
            [
                n
                for n in self.graph.nodes
                if self.graph.nodes[n].get("type") == self.FLAVOR
            ]
        )
        self.stats["nodes"]["Meridian"] = len(
            [
                n
                for n in self.graph.nodes
                if self.graph.nodes[n].get("type") == self.MERIDIAN
            ]
        )

    # ═══════════════════════════════════════════
    # 7. 症状-药材/方剂 关联
    # ═══════════════════════════════════════════

    def _add_symptom_herb_associations(self, pipeline):
        """从改进后的 pipeline 加载关联矩阵"""

        shm = pipeline.symptom_herb_matrix
        sfm = pipeline.symptom_formula_matrix

        # 症状→药材
        edge_count = 0
        for si in range(shm.shape[0]):
            s_name = pipeline.symptoms[si]
            snid = f"symptom:{s_name}"
            if not self.graph.has_node(snid):
                continue
            for hi in range(shm.shape[1]):
                val = shm[si, hi]
                if val > 0:
                    herb_name = pipeline.herb_names[hi]
                    hnid = f"herb:{herb_name}"
                    if self.graph.has_node(hnid):
                        self.graph.add_edge(
                            snid,
                            hnid,
                            type=self.ASSOC_HERB,
                            weight=float(val),
                            label="关联",
                        )
                        edge_count += 1
        logger.info("  症状→药材: %s 边", edge_count)

        # 症状→方剂
        fml_edge_count = 0
        for fi in range(sfm.shape[1]):
            # 找方剂名
            for db_id, mid in pipeline.formula_db_id_to_model_idx.items():
                if mid - 1 == fi:
                    fml = pipeline.formulas[db_id]
                    fml_name = fml["name"]
                    break
            else:
                continue
            fnid = f"formula:{fml_name}"
            if not self.graph.has_node(fnid):
                continue
            for si in range(sfm.shape[0]):
                if sfm[si, fi] > 0:
                    s_name = pipeline.symptoms[si]
                    snid = f"symptom:{s_name}"
                    if self.graph.has_node(snid):
                        self.graph.add_edge(
                            snid,
                            fnid,
                            type=self.ASSOC_FORMULA,
                            weight=float(sfm[si, fi]),
                            label="关联",
                        )
                        fml_edge_count += 1
        logger.info("  症状→方剂: %s 边", fml_edge_count)

    # ═══════════════════════════════════════════
    # 8. 方剂组成（药材→方剂）
    # ═══════════════════════════════════════════

    def _add_formula_compositions(self, pipeline):
        """解析方剂组成，构建药材→方剂边"""

        edge_count = 0
        for fid, formula in pipeline.formulas.items():
            fml_name = formula["name"]
            fnid = f"formula:{fml_name}"
            if not self.graph.has_node(fnid):
                continue
            comp = formula.get("composition", "")
            herb_idxs = pipeline._parse_composition_indices(comp)
            for hi in herb_idxs:
                herb_name = pipeline.herb_names[hi]
                hnid = f"herb:{herb_name}"
                if self.graph.has_node(hnid):
                    self.graph.add_edge(
                        hnid,
                        fnid,
                        type=self.COMPOSES,
                        label="组成",
                    )
                    edge_count += 1
        logger.info("  药材→方剂(组成): %s 边", edge_count)

    # ═══════════════════════════════════════════
    # 9. 配伍禁忌
    # ═══════════════════════════════════════════

    def _add_incompatibilities(self):
        """添加十八反十九畏 和 毒性"""
        import sys

        sys.path.insert(0, str(_PROJECT / "src"))
        from core.tcm_config import INCOMPATIBLE_HERBS, TOXIC_HERBS

        # 相反相畏（十八反 + 十九畏）
        for category_name in ["十八反", "十九畏"]:
            pairs = INCOMPATIBLE_HERBS.get(category_name, [])
            for pair in pairs:
                if len(pair) == 2:
                    h1, h2 = pair
                    n1, n2 = f"herb:{h1}", f"herb:{h2}"
                    if self.graph.has_node(n1) and self.graph.has_node(n2):
                        self.graph.add_edge(
                            n1, n2, type=self.INCOMPATIBLE, label=f"{category_name}"
                        )
                        self.graph.add_edge(
                            n2, n1, type=self.INCOMPATIBLE, label=f"{category_name}"
                        )

        # 孕妇禁用标记
        pregnant_forbidden = INCOMPATIBLE_HERBS.get("孕妇禁用", [])
        for hname in pregnant_forbidden:
            hnid = f"herb:{hname}"
            if self.graph.has_node(hnid):
                self.graph.nodes[hnid]["pregnancy_forbidden"] = True

        # 毒性标记
        toxic_set = set(TOXIC_HERBS)
        for nid in list(self.graph.nodes):
            if self.graph.nodes[nid].get("type") != self.HERB:
                continue
            name = self.graph.nodes[nid]["name"]
            if name in toxic_set:
                self.graph.nodes[nid]["toxic"] = True

    # ═══════════════════════════════════════════
    # 10. 药材共现
    # ═══════════════════════════════════════════

    def _add_cooccurrence(self, pipeline):
        """添加方剂内药材共现关系（Top 1000 最显著）"""

        cooc = pipeline.herb_cooccurrence
        if cooc is None:
            return

        # 只取共现强度 > 0.1 的边（避免噪音）
        threshold = 0.1
        edge_count = 0
        for i in range(cooc.shape[0]):
            for j in range(i + 1, cooc.shape[1]):
                val = cooc[i, j]
                if val > threshold:
                    h1 = pipeline.herb_names[i]
                    h2 = pipeline.herb_names[j]
                    n1, n2 = f"herb:{h1}", f"herb:{h2}"
                    if self.graph.has_node(n1) and self.graph.has_node(n2):
                        self.graph.add_edge(
                            n1,
                            n2,
                            type=self.CO_OCCURS,
                            weight=float(val),
                            label="共现",
                        )
                        edge_count += 1
        logger.info("  药材共现: %s 边 (阈值>%.1f)", edge_count, threshold)

    # ═══════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════

    def close(self):
        """关闭数据库连接"""
        if self._db:
            self._db.close()

    def node_count_by_type(self) -> dict[str, int]:
        """按类型统计节点数"""
        counts = {}
        for nid in self.graph.nodes:
            ntype = self.graph.nodes[nid].get("type", "Unknown")
            counts[ntype] = counts.get(ntype, 0) + 1
        return counts

    def edge_count_by_type(self) -> dict[str, int]:
        """按类型统计边数"""
        counts = {}
        for _, _, data in self.graph.edges(data=True):
            etype = data.get("type", "Unknown")
            counts[etype] = counts.get(etype, 0) + 1
        return counts

    def summary(self) -> str:
        """打印图统计摘要"""
        lines = []
        lines.append("=" * 55)
        lines.append("  TCM 知识图谱统计")
        lines.append("=" * 55)
        lines.append(f"  总节点: {self.graph.number_of_nodes()}")
        lines.append(f"  总边数: {self.graph.number_of_edges()}")
        lines.append("")
        lines.append("  【节点按类型】")
        for ntype, count in sorted(self.node_count_by_type().items()):
            lines.append(f"    {ntype}: {count}")
        lines.append("")
        lines.append("  【边按类型】")
        for etype, count in sorted(self.edge_count_by_type().items()):
            lines.append(f"    {etype}: {count}")
        lines.append("=" * 55)
        return "\n".join(lines)

    def save(self, path: str, fmt: str = "gexf"):
        """保存图到文件"""
        if fmt == "gexf":
            nx.write_gexf(self.graph, path)
        elif fmt == "gml":
            nx.write_gml(self.graph, path)
        elif fmt == "graphml":
            nx.write_graphml(self.graph, path)
        elif fmt == "json":
            import json
            from networkx.readwrite import json_graph

            data = json_graph.node_link_data(self.graph)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        size_kb = Path(path).stat().st_size / 1024
        logger.info("已保存: %s (%.1f KB, %s)", path, size_kb, fmt)
