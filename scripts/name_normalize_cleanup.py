#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""名称归一清洗批：
1) 含空格名称：无正本者重命名去空格；有正本者删除空格版（避免重复）。
2) 删除带注射液污染数据的重复行。
3) 剥离 category 字段的「N、」分节序号残留，统一格式。
零幻觉：仅做字段清洗，不增改任何药理学内容。
"""
import sqlite3, os, re

DB = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "tcm_knowledge.db"))

PREFIX_RE = re.compile(r'^[一二三四五六七八九十]+[、]\s*')  # 去掉 "一、 " 之类的分节号


def main():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    before = cur.execute("SELECT count(*) FROM patent_medicines").fetchone()[0]
    print(f"[before] 行数 = {before}")

    # ---- 1) 重命名无正本的空格名 ----
    rename_map = {
        "复 方丹参片": "复方丹参片",
        "艾 附暖宫丸": "艾附暖宫丸",
    }
    for old, new in rename_map.items():
        dup = cur.execute("SELECT 1 FROM patent_medicines WHERE name=?", (new,)).fetchone()
        if dup:
            print(f"[warn] 目标名 {new} 已存在，跳过重命名 {old}")
            continue
        n = cur.execute("UPDATE patent_medicines SET name=? WHERE name=?", (new, old)).rowcount
        print(f"[rename] {old} -> {new}  ({n} 行)")

    # ---- 2) 删除重复/污染行 ----
    # 通 宣理肺丸(id=99)：usage_dosage 是注射液说明，明显串数据；正本 通宣理肺丸(id=66) 干净，删空格版。
    # 维 C 银翘片(id=63)：与 维C银翘片(id=94) 内容重复，删空格版。
    delete_names = ["通 宣理肺丸", "维 C 银翘片"]
    for nm in delete_names:
        n = cur.execute("DELETE FROM patent_medicines WHERE name=?", (nm,)).rowcount
        print(f"[delete] {nm}  ({n} 行)")

    # ---- 3) 剥离 category 的「N、」分节序号 ----
    upd = 0
    for r in cur.execute("SELECT id, category FROM patent_medicines"):
        cat = r["category"]
        if not cat:
            continue
        newcat = PREFIX_RE.sub("", cat)
        if newcat != cat:
            cur.execute("UPDATE patent_medicines SET category=? WHERE id=?", (newcat, r["id"]))
            upd += 1
    print(f"[category] 剥离序号前缀 {upd} 行")

    con.commit()

    # ---- 验证 ----
    after = cur.execute("SELECT count(*) FROM patent_medicines").fetchone()[0]
    spaced = cur.execute("SELECT count(*) FROM patent_medicines WHERE name LIKE '% %'").fetchone()[0]
    pref = cur.execute("SELECT count(*) FROM patent_medicines WHERE category LIKE '%、%' OR category LIKE '%、 %'").fetchone()[0]
    # 检查规范化后是否还有重名
    dups = cur.execute(
        "SELECT REPLACE(REPLACE(name,' ',''),'　','') AS n, count(*) c "
        "FROM patent_medicines GROUP BY n HAVING c>1"
    ).fetchall()
    print(f"\n[after] 行数 = {after}  (净减 {before-after})")
    print(f"[check] 仍含空格名: {spaced} 行；仍含 'N、' 类 category: {pref} 行")
    print(f"[check] 规范化重名组: {[dict(d) for d in dups] if dups else '无'}")
    con.close()


if __name__ == "__main__":
    main()
