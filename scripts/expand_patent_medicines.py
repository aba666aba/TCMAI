# -*- coding: utf-8 -*-
"""中成药扩充执行器（幂等，可重跑）。

流程：
  1. FILL —— 填充现有条目的空字段（只填空，不覆盖）
  2. NEW  —— 插入新条目（按名去重：本名、patent_aliases 标准名/别名均已查重）
  3. 类目归一（统一 "XX类中成药" 风格）
  4. 输出统计报告

运行前应已备份：cp data/tcm_knowledge.db data/tcm_knowledge.db.bak_preexpand
"""

import io
import os
import sys
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

DB = os.path.join(ROOT, "data", "tcm_knowledge.db")

from patent_dataset_fill import FILL  # noqa: E402
from patent_dataset_new_a import NEW_A  # noqa: E402
from patent_dataset_new_b import NEW_B  # noqa: E402

NEW = NEW_A + NEW_B


def normalize_category(cat: str) -> str:
    cat = (cat or "").strip()
    if not cat:
        return cat
    if cat.endswith("类中成药"):
        return cat
    if cat.endswith("类"):
        return cat + "中成药"
    return cat + "类中成药"


def main():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    before = cur.execute("SELECT count(*) FROM patent_medicines").fetchone()[0]
    print(f"扩充前条目数: {before}")

    # ── 1. 填充空字段 ──
    filled_cells = 0
    filled_rows = 0
    missing = []
    for name, fields in FILL.items():
        row = cur.execute(
            "SELECT id FROM patent_medicines WHERE name=?", (name,)
        ).fetchone()
        if not row:
            missing.append(name)
            continue
        updates = []
        values = []
        for col, val in fields.items():
            if col in ("name",):
                continue
            cur_val = cur.execute(
                f"SELECT {col} FROM patent_medicines WHERE id=?", (row[0],)
            ).fetchone()[0]
            if cur_val is None or str(cur_val).strip() == "":
                updates.append(f"{col}=?")
                values.append(val)
        if updates:
            values.append(row[0])
            cur.execute(
                f"UPDATE patent_medicines SET {', '.join(updates)} WHERE id=?",
                tuple(values),
            )
            filled_cells += len(updates)
            filled_rows += 1
    print(f"[FILL] 填充 {filled_rows} 行 / {filled_cells} 个空字段")
    if missing:
        print(f"[FILL] 库中不存在的条目（跳过 {len(missing)}）: {'、'.join(missing)}")

    # ── 2. 插入新条目 ──
    existing_names = {
        r[0] for r in cur.execute("SELECT name FROM patent_medicines")
    }
    alias_map = {
        r[0]: r[1]
        for r in cur.execute("SELECT alias, standard FROM patent_aliases")
    }
    alias_names = set(alias_map) | set(alias_map.values())

    inserted = 0
    skipped_dup = []
    for item in NEW:
        name, cat, comp, func, ind, usage, contra, source = item
        if name in existing_names or name in alias_names:
            skipped_dup.append(name)
            continue
        cur.execute(
            "INSERT INTO patent_medicines "
            "(name, category, composition, functions, indications, "
            " usage_dosage, contraindications, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                name,
                normalize_category(cat),
                comp,
                func,
                ind,
                usage,
                contra,
                source,
            ),
        )
        existing_names.add(name)
        inserted += 1
    print(f"[NEW] 新增 {inserted} 条；按名去重跳过 {len(skipped_dup)} 条")
    if skipped_dup:
        print("  跳过: " + "、".join(skipped_dup[:20]) + ("…" if len(skipped_dup) > 20 else ""))

    # ── 3. 全表类目归一 ──
    norm = 0
    for pid, cat in cur.execute(
        "SELECT id, category FROM patent_medicines"
    ).fetchall():
        ncat = normalize_category(cat)
        if ncat != cat:
            cur.execute(
                "UPDATE patent_medicines SET category=? WHERE id=?", (ncat, pid)
            )
            norm += 1
    print(f"[NORM] 类目归一修正 {norm} 行")

    con.commit()

    # ── 4. 统计报告 ──
    after = cur.execute("SELECT count(*) FROM patent_medicines").fetchone()[0]
    from collections import Counter

    cats = Counter(r[0] for r in cur.execute("SELECT category FROM patent_medicines"))
    empties = {}
    for col in (
        "composition", "functions", "indications",
        "usage_dosage", "contraindications", "source",
    ):
        e = cur.execute(
            f"SELECT count(*) FROM patent_medicines "
            f"WHERE {col} IS NULL OR trim({col})=''"
        ).fetchone()[0]
        empties[col] = e
    con.close()

    print()
    print(f"扩充后条目数: {before} → {after}（净增 {after - before}）")
    print(f"类目数: {len(cats)}")
    print("空字段统计:")
    for col, e in empties.items():
        pct = (after - e) / after * 100
        print(f"  {col:20s} 空 {e:>3}（完整率 {pct:.0f}%）")
    print()
    print("TOP12 类目:")
    for k, v in cats.most_common(12):
        print(f"  {k:20s} {v}")
    print()
    print("✅ 扩充完成（数据库已提交；备份在 data/tcm_knowledge.db.bak_preexpand）")


if __name__ == "__main__":
    main()
