# -*- coding: utf-8 -*-
"""通用草稿导入脚本：把 data/exports/drafts/ 下的批次草稿导入数据库。

用法：
    python3 scripts/import_draft.py data/exports/drafts/batch6_patent_draft.json [--no-backup]

流程：读草稿 -> 备份 DB(默认) -> 去重跳过已存在 -> INSERT -> 重建 RAG 语料库。
"""
import json
import os
import shutil
import sqlite3
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(REPO, "data", "tcm_knowledge.db")

FIELDS = ("name", "category", "composition", "functions", "indications",
          "usage_dosage", "contraindications", "source")


def main():
    if len(sys.argv) < 2:
        print("用法: python3 scripts/import_draft.py <draft.json> [--no-backup]")
        sys.exit(1)
    draft_path = sys.argv[1] if os.path.isabs(sys.argv[1]) else os.path.join(REPO, sys.argv[1])
    do_backup = "--no-backup" not in sys.argv

    with open(draft_path, encoding="utf-8") as f:
        payload = json.load(f)
    records = payload["records"]
    batch = payload.get("meta", {}).get("batch", "?")
    print("[%s] 草稿待导入记录: %d" % (batch, len(records)))

    backup_path = None
    if do_backup:
        tag = str(batch)
        backup_path = os.path.join(REPO, "data", "tcm_knowledge.db.bak_%s" % tag)
        shutil.copy2(DB, backup_path)
        print("已备份数据库 ->", os.path.relpath(backup_path, REPO))

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
