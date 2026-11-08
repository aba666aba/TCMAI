# -*- coding: utf-8 -*-
"""
第 4 批中成药草稿导入器。
读取 data/exports/drafts/batch4_patent_draft.json，将 status=='draft' 的记录写入
data/tcm_knowledge.db 的 patent_medicines 表（已存在的 name 跳过）。
执行前自动备份数据库，导入后自动重建 RAG 语料库。
⚠️ 仅在用户确认后运行本脚本。
"""
import os, json, sqlite3, shutil, datetime

DB = "data/tcm_knowledge.db"
DRAFT = "data/exports/drafts/batch4_patent_draft.json"
BAK = "data/tcm_knowledge.db.bak_prebatch4"

FIELDS = ["name", "category", "composition", "functions",
          "indications", "usage_dosage", "contraindications", "source"]

def main():
    # 1) 备份
    if os.path.exists(DB):
        shutil.copy(DB, BAK)
        print("已备份数据库 ->", BAK)
    else:
        raise SystemExit("找不到数据库: " + DB)

    # 2) 读草稿
    with open(DRAFT, encoding="utf-8") as f:
        draft = json.load(f)
    recs = [r for r in draft["records"] if r.get("status") == "draft"]
    print("草稿待导入记录:", len(recs))

    # 3) 连接并去重
    con = sqlite3.connect(DB)
    cur = con.cursor()
    have = set(r[0] for r in cur.execute("SELECT name FROM patent_medicines"))
    inserted, skipped = [], []
    for r in recs:
        if r["name"] in have:
            skipped.append(r["name"])
            continue
        vals = [r.get(k) for k in FIELDS]
        q = "INSERT INTO patent_medicines (%s) VALUES (%s)" % (
            ",".join(FIELDS), ",".join("?" for _ in FIELDS))
        cur.execute(q, vals)
        inserted.append(r["name"])
        have.add(r["name"])
    con.commit()

    n = cur.execute("SELECT count(*) FROM patent_medicines").fetchone()[0]
    print("新增 %d 条: %s" % (len(inserted), inserted))
    print("跳过(已存在) %d 条: %s" % (len(skipped), skipped))
    print("patent_medicines 当前行数:", n)
    con.close()

    # 4) 重建语料库
    try:
        import src.rag.corpus as corpus
        corpus.build_corpus()
        print("RAG 语料库已重建")
    except Exception as e:
        print("语料库重建失败(需手动):", e)

if __name__ == "__main__":
    main()
