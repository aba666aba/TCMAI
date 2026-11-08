# -*- coding: utf-8 -*-
"""导入第 5 批中成药本地草稿到数据库。

流程：读取 data/exports/drafts/batch5_patent_draft.json ->
备份 DB -> 去重跳过已存在 -> INSERT -> 重建 RAG 语料库。
"""
import json
import os
import sqlite3
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(REPO, "data", "tcm_knowledge.db")
DRAFT = os.path.join(REPO, "data", "exports", "drafts", "batch5_patent_draft.json")
BACKUP = os.path.join(REPO, "data", "tcm_knowledge.db.bak_prebatch5")

FIELDS = ("name", "category", "composition", "functions", "indications",
          "usage_dosage", "contraindications", "source")


def main():
    # 1) 读草稿
    with open(DRAFT, encoding="utf-8") as f:
        payload = json.load(f)
    records = payload["records"]
    print("草稿待导入记录:", len(records))

    # 2) 备份
    import shutil
    shutil.copy2(DB, BACKUP)
    print("已备份数据库 ->", os.path.relpath(BACKUP, REPO))

    # 3) 去重 + 插入
    con = sqlite3.connect(DB)
    cur = con.cursor()
    existing = {r[0] for r in cur.execute("SELECT name FROM patent_medicines")}
    added, skipped = [], []
    for rec in records:
        name = rec["name"]
        if name in existing:
            skipped.append(name)
            continue
        cur.execute(
            "INSERT INTO patent_medicines (name, category, composition, functions, "
            "indications, usage_dosage, contraindications, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            tuple(rec.get(k) or "" for k in FIELDS),
        )
        existing.add(name)
        added.append(name)
    con.commit()

    n = cur.execute("SELECT count(*) FROM patent_medicines").fetchone()[0]
    con.close()
    print("新增 %d 条: %s" % (len(added), added))
    if skipped:
        print("跳过(已存在) %d 条: %s" % (len(skipped), skipped))
    print("patent_medicines 当前行数:", n)

    # 4) 重建语料库（修复模块路径）
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    try:
        import src.rag.corpus as corpus_mod
        corpus_mod.build_corpus()
        print("RAG 语料库已重建")
    except Exception as e:  # noqa: BLE001
        print("语料库重建失败(需手动):", e)


if __name__ == "__main__":
    main()
