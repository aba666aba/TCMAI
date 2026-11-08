# -*- coding: utf-8 -*-
"""语料构建：把 SQLite 知识库 + 知识图谱导出为结构化检索文档。

输出：data/rag_corpus.jsonl
每行一个 doc：{"id", "type", "title", "text", "meta"}
  - herb / formula / patent / remedy / syndrome : 来自 SQLite
  - kg_rel : 来自知识图谱，按节点聚合其关系（症状↔证型↔药材↔方剂）

该语料是检索层（retriever）与后续答案合成器的统一证据来源。
"""
import json
import os
import sqlite3
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(ROOT, "data", "tcm_knowledge.db")
KG_PATH = os.path.join(ROOT, "data", "tcm_knowledge_graph.json")
OUT_PATH = os.path.join(ROOT, "data", "rag_corpus.jsonl")


def _safe(row, key):
    v = row[key] if key in row.keys() else None
    return (v or "").strip()


def _join(parts):
    return "；".join(p for p in parts if p)


def build_from_db(con):
    """从 SQLite 各表产出实体文档。"""
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    docs = []
    did = 0

    # 药材
    for r in cur.execute("SELECT * FROM herbs"):
        text = _join(
            [
                f"药材『{_safe(r,'name')}』，分类：{_safe(r,'category')}",
                f"性味：{_safe(r,'nature')} {_safe(r,'flavor')}",
                f"归经：{_safe(r,'meridian')}",
                f"功效：{_safe(r,'efficacy')}",
                f"主治：{_safe(r,'indications')}",
                f"用法：{_safe(r,'usage')}",
                f"禁忌：{_safe(r,'contraindications')}",
                f"来源：{_safe(r,'source')}",
            ]
        )
        docs.append(
            {
                "id": f"herb:{_safe(r,'name')}",
                "type": "herb",
                "title": _safe(r, "name"),
                "text": text,
                "meta": {"table": "herbs", "category": _safe(r, "category")},
            }
        )
        did += 1

    # 方剂
    for r in cur.execute("SELECT * FROM formulas"):
        text = _join(
            [
                f"方剂『{_safe(r,'name')}』，分类：{_safe(r,'category')}，出处：{_safe(r,'source_book')}",
                f"组成：{_safe(r,'composition')}",
                f"用量：{_safe(r,'dosage')}",
                f"功用：{_safe(r,'functions')}",
                f"主治：{_safe(r,'indications')}",
                f"用法：{_safe(r,'usage')}",
                f"加减：{_safe(r,'modification')}",
                f"方解：{_safe(r,'explanation')}",
            ]
        )
        docs.append(
            {
                "id": f"formula:{_safe(r,'name')}",
                "type": "formula",
                "title": _safe(r, "name"),
                "text": text,
                "meta": {"table": "formulas", "category": _safe(r, "category")},
            }
        )
        did += 1

    # 中成药别名（patent_aliases：同药异名/异剂型互通，2026-11-08 结构优化）
    patent_alias_map = defaultdict(list)
    try:
        for ar in cur.execute("SELECT alias, standard FROM patent_aliases"):
            patent_alias_map[ar["standard"]].append(ar["alias"])
    except sqlite3.OperationalError:
        pass  # 旧库无此表

    # 中成药
    for r in cur.execute("SELECT * FROM patent_medicines"):
        nm = _safe(r, "name")
        aliases = patent_alias_map.get(nm, [])
        head = f"中成药『{nm}』，分类：{_safe(r,'category')}"
        if aliases:
            head += f"，别名：{'、'.join(aliases)}"
        text = _join(
            [
                head,
                f"组成：{_safe(r,'composition')}",
                f"功用：{_safe(r,'functions')}",
                f"主治：{_safe(r,'indications')}",
                f"用法用量：{_safe(r,'usage_dosage')}",
                f"禁忌：{_safe(r,'contraindications')}",
                f"来源：{_safe(r,'source')}",
            ]
        )
        docs.append(
            {
                "id": f"patent:{nm}",
                "type": "patent",
                "title": nm,
                "text": text,
                "meta": {
                    "table": "patent_medicines",
                    "category": _safe(r, "category"),
                    "aliases": aliases,
                },
            }
        )
        did += 1

    # 民间验方
    for r in cur.execute("SELECT * FROM folk_remedies"):
        text = _join(
            [
                f"民间验方（主治{_safe(r,'disease')}）：{_safe(r,'remedy')}",
                f"组成：{_safe(r,'ingredients')}",
                f"制法：{_safe(r,'method')}",
                f"来源：{_safe(r,'source')}",
            ]
        )
        docs.append(
            {
                "id": f"remedy:{_safe(r,'disease')}:{_safe(r,'remedy')}",
                "type": "remedy",
                "title": f"{_safe(r,'disease')}·{_safe(r,'remedy')}",
                "text": text,
                "meta": {"table": "folk_remedies", "disease": _safe(r, "disease")},
            }
        )
        did += 1

    # 证型
    for r in cur.execute("SELECT * FROM syndrome_definitions"):
        text = _join(
            [
                f"证型『{_safe(r,'name')}』：{_safe(r,'description')}",
                f"症状：{_safe(r,'symptoms_json')}",
                f"治法：{_safe(r,'treatment')}",
                f"推荐方剂：{_safe(r,'formula')}",
            ]
        )
        docs.append(
            {
                "id": f"syndrome:{_safe(r,'name')}",
                "type": "syndrome",
                "title": _safe(r, "name"),
                "text": text,
                "meta": {"table": "syndrome_definitions"},
            }
        )
        did += 1

    return docs


def build_from_kg(kg_path):
    """从知识图谱按节点聚合关系，产出 kg_rel 文档。"""
    if not os.path.exists(kg_path):
        return []
    with open(kg_path, encoding="utf-8") as f:
        g = json.load(f)
    nodes = {n["id"]: n for n in g.get("nodes", [])}
    # 按 source 聚合关系
    rels = defaultdict(list)
    for e in g.get("links", []):
        src = e.get("source")
        tgt = e.get("target")
        label = e.get("label") or e.get("type") or "相关"
        rels[src].append((label, tgt))

    docs = []
    for nid, n in nodes.items():
        ntype = n.get("type", "Node")
        nname = n.get("name", nid)
        if nid not in rels:
            continue
        rel_lines = []
        for label, tgt in rels[nid]:
            tname = nodes.get(tgt, {}).get("name", tgt)
            rel_lines.append(f"{label}『{tname}』")
        text = f"{ntype}『{nname}』的关系：{_join(rel_lines)}。"
        docs.append(
            {
                "id": f"kg:{nid}",
                "type": "kg_rel",
                "title": nname,
                "text": text,
                "meta": {"kg_type": ntype, "kg_id": nid},
            }
        )
    return docs


def build_corpus(db_path=DB_PATH, kg_path=KG_PATH, out_path=OUT_PATH):
    con = sqlite3.connect(db_path)
    db_docs = build_from_db(con)
    con.close()
    kg_docs = build_from_kg(kg_path)
    all_docs = db_docs + kg_docs

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for d in all_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    stats = defaultdict(int)
    for d in all_docs:
        stats[d["type"]] += 1
    return {"total": len(all_docs), "by_type": dict(stats), "out_path": out_path}


def main():
    res = build_corpus()
    print("语料构建完成：")
    print(f"  输出: {res['out_path']}")
    print(f"  总文档数: {res['total']}")
    for t, c in sorted(res["by_type"].items(), key=lambda x: -x[1]):
        print(f"    {t:<10} {c}")


if __name__ == "__main__":
    main()
