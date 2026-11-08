#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 1 批中成药联网核实录入 + 污染行清理。零幻觉：仅写入有权威出处的字段。"""
import sqlite3, os, sys

DB = os.path.join(os.path.dirname(__file__), "..", "data", "tcm_knowledge.db")
DB = os.path.abspath(DB)

SRC = ("据腾讯医典、中国医药信息查询平台(国家中医药管理局认证)、国家药监局(NMPA)非处方药说明书范本、"
       "科普中国/中国中医药网、同仁堂/以岭/云南白药官网等权威来源核对录入（2026-11-08）")

NEW_MEDS = [
    {
        "name": "连花清瘟胶囊",
        "category": "清热剂（清热解毒）",
        "composition": "连翘、金银花、炙麻黄、炒苦杏仁、石膏、板蓝根、绵马贯众、鱼腥草、广藿香、大黄、红景天、薄荷脑、甘草",
        "functions": "清瘟解毒，宣肺泄热",
        "indications": "用于治疗流行性感冒属热毒袭肺证，症见发热或高热、恶寒、肌肉酸痛、鼻塞流涕、咳嗽、头痛、咽干咽痛、舌偏红、苔黄或黄腻等；在新型冠状病毒肺炎的常规治疗中，可用于轻型、普通型引起的发热、咳嗽、乏力",
        "usage_dosage": "口服。一次4粒，一日3次。新型冠状病毒肺炎轻型、普通型疗程7-10天。每粒装0.35g",
        "contraindications": "对本品及所含成份过敏者忌服；运动员慎用；风寒感冒者不适用",
        "source": SRC,
    },
    {
        "name": "藿香正气水",
        "category": "祛暑剂（解表化湿）",
        "composition": "苍术、陈皮、厚朴(姜制)、白芷、茯苓、大腹皮、生半夏、甘草浸膏、广藿香油、紫苏叶油",
        "functions": "解表化湿，理气和中",
        "indications": "用于外感风寒、内伤湿滞或夏伤暑湿所致的感冒，症见头痛昏重、胸膈痞闷、脘腹胀痛、呕吐泄泻；胃肠型感冒见上述证候者",
        "usage_dosage": "口服。一次5-10ml，一日2次，用时摇匀。每支装10ml。含乙醇(酒精)40%-50%",
        "contraindications": "对本品及所含成份过敏者禁用；酒精过敏者禁用；不建议儿童、孕妇及哺乳期妇女使用；本品含生半夏，不宜过量或长期服用",
        "source": SRC,
    },
    {
        "name": "逍遥丸",
        "category": "疏肝解郁剂",
        "composition": "柴胡、当归、白芍、白术(炒)、茯苓、薄荷、生姜、甘草(蜜炙)",
        "functions": "疏肝健脾，养血调经",
        "indications": "用于肝气不舒所致月经不调、胸胁胀痛、头晕目眩、食欲减退",
        "usage_dosage": "口服。水丸一次6-9g，一日1-2次；小蜜丸一次9g，大蜜丸一次1丸，一日2次",
        "contraindications": "对本品过敏者禁用，过敏体质者慎用；月经过多者、感冒患者慎用；孕妇服用时请向医师咨询",
        "source": SRC,
    },
    {
        "name": "云南白药",
        "category": "理血剂（止血活血）",
        "composition": "国家保密方（本品含草乌(制)，其余成份为保密处方；公开资料示其以三七、重楼等为主）",
        "functions": "化瘀止血，活血止痛，解毒消肿",
        "indications": "用于跌打损伤、瘀血肿痛，吐血、咳血、便血、痔血、崩漏下血、手术出血，疮疡肿毒及软组织挫伤、闭合性骨折，支气管扩张及肺结核咳血、溃疡病出血，以及皮肤感染性疾病",
        "usage_dosage": "口服。一次0.25-0.5g，一日4次（二至五岁按1/4剂量，六至十二岁按1/2剂量）。刀枪跌打诸伤出血者温开水送服，瘀血肿痛与未流血者用酒送服；较重跌打损伤可先服保险子1粒。每瓶装4g",
        "contraindications": "孕妇禁用；对本品过敏者禁用；皮肤及黏膜破溃、化脓者禁外用；运动员慎用；服药一日内忌食蚕豆、鱼类及酸冷食物",
        "source": SRC,
    },
    {
        "name": "健胃消食片",
        "category": "消食剂",
        "composition": "太子参、陈皮、山药、炒麦芽、山楂",
        "functions": "健胃消食",
        "indications": "用于脾胃虚弱所致的食积，症见不思饮食、嗳腐酸臭、脘腹胀满；消化不良见上述证候者",
        "usage_dosage": "口服或咀嚼。成人一次4-6片，一日3次；儿童二至四岁一次2片，五至八岁一次3片，九至十四岁一次4片，一日3次。每片重0.5g或0.8g",
        "contraindications": "对本品过敏者禁用，过敏体质者慎用；有高血压、心脏病、肝病、糖尿病、肾病等慢性病严重者应在医师指导下服用",
        "source": SRC,
    },
    {
        "name": "牛黄解毒片",
        "category": "清热泻火剂",
        "composition": "人工牛黄、雄黄、石膏、大黄、黄芩、桔梗、冰片、甘草",
        "functions": "清热解毒",
        "indications": "用于火热内盛，咽喉肿痛，牙龈肿痛，口舌生疮，目赤肿痛",
        "usage_dosage": "口服。一次3片，一日2-3次",
        "contraindications": "孕妇、哺乳期妇女禁用；婴幼儿禁用；对本品及所含成份过敏者禁用；本品含雄黄，不可超剂量或长期服用；平素脾胃虚弱、大便溏薄者及严重肝损害、急慢性疾病肾病患者慎用",
        "source": SRC,
    },
]

def main():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    before = cur.execute("SELECT count(*) FROM patent_medicines").fetchone()[0]
    print(f"[before] patent_medicines 行数 = {before}")

    # 1) 清理污染行：name 被写成整段说明文字（非药名）
    suspicious = cur.execute(
        "SELECT id, name FROM patent_medicines WHERE length(name) > 15"
    ).fetchall()
    print(f"\n[scan] 长度>15的名称（疑似污染/异常）共 {len(suspicious)} 条:")
    for r in suspicious:
        print(f"   id={r['id']}  name={r['name'][:30]}...")

    # 明确为段落说明文字的两条（理气类/理血类整段描述）
    del_pat = (
        "DELETE FROM patent_medicines "
        "WHERE name LIKE '理气类中成药%' OR name LIKE '理血类中成药%'"
    )
    del_cnt = cur.execute(del_pat).rowcount
    print(f"\n[clean] 删除污染行数 = {del_cnt}")

    # 2) 插入 6 个新药（跳过已存在的 name）
    inserted = 0
    skipped = 0
    for m in NEW_MEDS:
        exists = cur.execute(
            "SELECT 1 FROM patent_medicines WHERE name=?", (m["name"],)
        ).fetchone()
        if exists:
            skipped += 1
            print(f"[skip] 已存在: {m['name']}")
            continue
        cur.execute(
            """INSERT INTO patent_medicines
               (name, category, composition, functions, indications, usage_dosage, contraindications, source)
               VALUES (?,?,?,?,?,?,?,?)""",
            (m["name"], m["category"], m["composition"], m["functions"],
             m["indications"], m["usage_dosage"], m["contraindications"], m["source"]),
        )
        inserted += 1
        print(f"[insert] {m['name']}")

    con.commit()
    after = cur.execute("SELECT count(*) FROM patent_medicines").fetchone()[0]
    print(f"\n[after] patent_medicines 行数 = {after}  (新增 {inserted}, 跳过 {skipped}, 删除 {del_cnt})")
    con.close()
    return inserted

if __name__ == "__main__":
    n = main()
    sys.exit(0 if n >= 0 else 1)
