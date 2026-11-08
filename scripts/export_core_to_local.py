# -*- coding: utf-8 -*-
"""将核心四表完整导出为本地 JSON 存档，并校验与数据库一致性。
导出目录：data/exports/{table}.json
同时生成 data/exports/MANIFEST.json 记录行数与导出时间。
"""
import sqlite3, json, os, datetime

DB = "data/tcm_knowledge.db"
OUT = "data/exports"
TABLES = ["herbs", "formulas", "patent_medicines", "folk_remedies"]

os.makedirs(OUT, exist_ok=True)
con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row
cur = con.cursor()

manifest = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "source_db": DB, "tables": {}}

for t in TABLES:
    rows = cur.execute(f'SELECT * FROM "{t}"').fetchall()
    cols = [d[0] for d in cur.description]
    data = [dict(r) for r in rows]
    path = os.path.join(OUT, f"{t}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    db_n = cur.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
    manifest["tables"][t] = {
        "rows": len(data),
        "columns": cols,
        "file": f"exports/{t}.json",
        "db_rows": db_n,
        "consistent": len(data) == db_n,
    }
    print(f"{t:16s} 导出 {len(data):5d} 行 -> exports/{t}.json  (库内 {db_n} 行, {'一致' if len(data)==db_n else '不一致!'})")

with open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
con.close()
print("\nMANIFEST.json 已生成。")
