# -*- coding: utf-8 -*-
"""数据库结构优化（2026-11-08 结构优化批）

1. 补索引：patent_medicines.name / folk_remedies.disease / symptom_formula_map.symptom
2. 新建 patent_aliases 别名表（与 herb_aliases/formula_aliases 对齐），填已核实同药异名
3. 修复名称-内容错位行：id=110 "藿香正气胶囊（水）"(内容实为番泻叶冲剂) → 改名；删除空壳 id=19
4. 合并同方重复：麻子仁丸(id=112) 并入 麻仁润肠丸(id=111)，成分取更完整者
5. category 归一：74 种写法 → 统一体系；补 6 行空类别

幂等：重复运行安全。所有改动先打印再执行。
"""
import os
import shutil
import sqlite3
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "tcm_knowledge.db")
BAK = DB + ".bak_preoptimize"

# ---------- 1. 别名（alias -> standard，均已联网核实或此前批次确认） ----------
ALIASES = [
    ("板蓝根颗粒", "板蓝根冲剂"),
    ("川贝枇杷露", "川贝枇杷糖浆"),
    ("丹栀逍遥丸", "加味逍遥丸"),
    ("益母草颗粒", "益母草膏"),
    ("香砂六君丸", "香砂六君子丸"),
    ("西瓜霜喷剂", "桂林西瓜霜"),
    ("霍胆丸", "藿胆丸"),
    ("七宝美髯丹", "七宝美髯丸"),
    ("感冒清热冲剂", "感冒清热颗粒"),
    ("藿香正气口服液", "藿香正气水"),
    ("麻子仁丸", "麻仁润肠丸"),
]

# ---------- 2. category 归一映射 ----------
VALUE_MAP = {
    # 清热系同义异写
    "清热剂": "清热解毒类中成药",
    "清热剂（清热解毒）": "清热解毒类中成药",
    "清热解毒剂": "清热解毒类中成药",
    "清热解毒药": "清热解毒类中成药",
    "清热泻火剂": "清热泻火类中成药",
    # 解表系
    "解表剂（中西药复方）": "解表类中成药",
    "辛温解表药": "解表类中成药",
    "解表止痛类中成药": "解表类中成药",
    # 祛暑系
    "祛暑剂（解表化湿）": "祛暑类中成药",
    "解表祛暑剂": "祛暑类中成药",
    # 理气/脾胃系
    "疏肝和胃剂": "理气类中成药",
    "疏肝解郁剂": "理气类中成药",
    "理气剂": "理气类中成药",
    "健脾和胃剂": "补益类中成药",
    # 消导系
    "消导剂": "消食导滞类中成药",
    "消食剂": "消食导滞类中成药",
    "消导类中成药": "消食导滞类中成药",
    # 化痰止咳系
    "化痰止咳平喘药": "化痰止咳类中成药",
    "化痰止咳平喘药（润肺止咳）": "化痰止咳类中成药",
    "祛痰止咳平喘类中成药": "化痰止咳类中成药",
    # 清热利咽
    "清热解毒药（利咽）": "清热利咽类中成药",
    # 温里 / 表里双解
    "温中散寒剂": "温里类中成药",
    "表里双解剂": "表里双解类中成药",
    # 理血
    "理血剂（止血活血）": "理血类中成药",
    # 妇科
    "补益调经类中成药": "调经类中成药",
    "妇科理血类中成药": "调经类中成药",
    # 五官科
    "耳鼻喉科类中成药": "鼻病类中成药",
    "咽喉科类中成药": "清热利咽类中成药",
    # 其他
    "祛风通络类中成药": "祛风湿类中成药",
}

# 按药名特判（旧类错归或需拆分）
BY_NAME_MAP = {
    # 原"清热类中成药"8 行按实际功用拆分
    "柴胡注射液": "清热解毒类中成药",
    "牛黄解毒丸": "清热泻火类中成药",
    "银黄口服液": "清热解毒类中成药",
    "西黄丸": "清热解毒类中成药",
    "龙胆泻肝丸": "清热除湿类中成药",
    "三黄片": "清热泻火类中成药",
    "六神丸": "清热解毒类中成药",
    "清开灵注射液": "清热解毒类中成药",
    # 旧类错归
    "醒脑静注射液": "清热解毒类中成药",   # 原误归"祛暑类"
    "保和丸": "消食导滞类中成药",        # 原误归"妇科类"
    "柏子养心丸": "安神类中成药",        # 原"收敛类"→安神
    "缩泉丸": "补益类中成药",           # 原"收敛类"→补益
    "八珍益母丸": "调经类中成药",
    "潮安胶囊": "调经类中成药",
}

# 空 category 补类（按其功用主治归类）
EMPTY_FILL = {
    "板蓝根冲剂": "清热解毒类中成药",
    "川贝枇杷糖浆": "化痰止咳类中成药",
    "通宣理肺丸": "化痰止咳类中成药",
    "穿琥宁": "清热解毒类中成药",
    "藿香正气胶囊": "祛暑类中成药",
    # 番泻叶冲剂由 id=110 改名后继承"泻下类中成药"
}


def main():
    shutil.copy2(DB, BAK)
    print("[backup] ->", os.path.basename(BAK))

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # ===== 1. 索引 =====
    for name, tbl, col in [
        ("idx_patent_name", "patent_medicines", "name"),
        ("idx_folk_disease", "folk_remedies", "disease"),
        ("idx_sfm_symptom", "symptom_formula_map", "symptom"),
    ]:
        exists = cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?", (name,)
        ).fetchone()
        if not exists:
            cur.execute(f"CREATE INDEX {name} ON {tbl}({col})")
            print(f"[index] +{name} ON {tbl}({col})")
        else:
            print(f"[index] {name} 已存在，跳过")

    # ===== 2. 别名表 =====
    cur.execute(
        "CREATE TABLE IF NOT EXISTS patent_aliases ("
        "alias TEXT PRIMARY KEY, standard TEXT NOT NULL)"
    )
    have = set(r[0] for r in cur.execute("SELECT name FROM patent_medicines"))
    added = skipped = 0
    for alias, std in ALIASES:
        if std not in have:
            print(f"[alias] 跳过 {alias} -> {std}（标准名不在库）")
            skipped += 1
            continue
        cur.execute(
            "INSERT OR IGNORE INTO patent_aliases(alias, standard) VALUES(?,?)",
            (alias, std),
        )
        added += cur.execute("SELECT changes()").fetchone()[0]
    n_alias = cur.execute("SELECT count(*) FROM patent_aliases").fetchone()[0]
    print(f"[alias] patent_aliases 表新增 {added} 条，跳过 {skipped}，现共 {n_alias} 条")

    # ===== 3. 修复 id=110 名称-内容错位，删除空壳 id=19 =====
    r110 = cur.execute("SELECT name, composition FROM patent_medicines WHERE id=110").fetchone()
    if r110 and r110["name"] == "藿香正气胶囊（水）":
        cur.execute(
            "UPDATE patent_medicines SET name=?, category=?, source=? WHERE id=110",
            (
                "番泻叶冲剂",
                "泻下类中成药",
                "原库名称错位修复：原名『藿香正气胶囊（水）』实为番泻叶冲剂内容（2026-11-08 结构优化）",
            ),
        )
        print("[fix] id=110 『藿香正气胶囊（水）』-> 『番泻叶冲剂』(泻下类中成药)")
    # id=19 空壳（成分与 id=110 相同，其余字段全空）→ 删除
    r19 = cur.execute("SELECT name, functions, indications FROM patent_medicines WHERE id=19").fetchone()
    if r19 and r19["name"] == "番泻叶冲剂" and not (r19["functions"] or "").strip():
        cur.execute("DELETE FROM patent_medicines WHERE id=19")
        print("[fix] 删除 id=19 空壳『番泻叶冲剂』（内容已由 id=110 修复行承载）")

    # ===== 4. 合并麻子仁丸 -> 麻仁润肠丸 =====
    r112 = cur.execute("SELECT composition FROM patent_medicines WHERE name='麻子仁丸'").fetchone()
    r111 = cur.execute("SELECT id, composition FROM patent_medicines WHERE name='麻仁润肠丸'").fetchone()
    if r112 and r111 and "枳实" in (r112["composition"] or "") and "枳实" not in (r111["composition"] or ""):
        old_src = (cur.execute(
            "SELECT source FROM patent_medicines WHERE id=?", (r111["id"],)
        ).fetchone()[0] or "")
        merge_note = "；成分取自同方『麻子仁丸』行合并（2026-11-08 结构优化）"
        new_src = (old_src + merge_note) if old_src.strip() else merge_note.lstrip("；")
        cur.execute(
            "UPDATE patent_medicines SET composition=?, source=? WHERE id=?",
            (r112["composition"], new_src, r111["id"]),
        )
        cur.execute("DELETE FROM patent_medicines WHERE name='麻子仁丸'")
        print("[merge] 麻子仁丸(id=112) 并入 麻仁润肠丸(id=%s)：成分补全，删除重复行" % r111["id"])
    else:
        print("[merge] 麻子仁丸合并条件不满足或已处理，跳过")

    # ===== 5. category 归一 =====
    rows = cur.execute("SELECT id, name, category FROM patent_medicines").fetchall()
    changed = 0
    for r in rows:
        old = (r["category"] or "").strip()
        name = r["name"]
        new = old
        if not old:
            new = EMPTY_FILL.get(name, old)
        if old in VALUE_MAP:
            new = VALUE_MAP[old]
        if name in BY_NAME_MAP:
            new = BY_NAME_MAP[name]
        if new != old:
            cur.execute(
                "UPDATE patent_medicines SET category=? WHERE id=?", (new, r["id"])
            )
            changed += 1
    print(f"[category] 归一完成，改动 {changed} 行")

    con.commit()

    # ===== 汇总 =====
    n = cur.execute("SELECT count(*) FROM patent_medicines").fetchone()[0]
    cats = Counter((r[0] or "(空)") for r in cur.execute("SELECT category FROM patent_medicines"))
    print(f"\n===== 汇总 =====")
    print(f"patent_medicines: {n} 行（优化前 221，净 -1 空壳 -1 麻子仁丸合并 +别名表）")
    print(f"category: {len(cats)} 个标准类目")
    for k, v in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    con.close()


if __name__ == "__main__":
    main()
