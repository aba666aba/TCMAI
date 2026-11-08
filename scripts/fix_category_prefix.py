#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修正版：剥离 patent_medicines.category 的「N、」分节序号残留。
先取列表再更新（避免同游标迭代中被 UPDATE 重置），幂等可重跑。
"""
import sqlite3, os, re

DB = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "tcm_knowledge.db"))
PREFIX_RE = re.compile(r'^[一二三四五六七八九十]+[、]\s*')


def main():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    rows = cur.execute("SELECT id, category FROM patent_medicines").fetchall()  # 先取列表
    upd = 0
    for r in rows:
        cat = r["category"]
        if not cat:
            continue
        newcat = PREFIX_RE.sub("", cat)
        if newcat != cat:
            cur.execute("UPDATE patent_medicines SET category=? WHERE id=?", (newcat, r["id"]))
            upd += 1
    con.commit()
    remain = cur.execute(
        "SELECT count(*) FROM patent_medicines WHERE category LIKE '%、%' OR category LIKE '%、 %'"
    ).fetchone()[0]
    print(f"[category] 剥离前缀 {upd} 行；剩余带前缀: {remain} 行")
    # 抽样展示
    for r in cur.execute("SELECT name,category FROM patent_medicines WHERE category IS NOT NULL AND category!='' LIMIT 5"):
        print("   ", r["name"], "->", r["category"])
    con.close()


if __name__ == "__main__":
    main()
