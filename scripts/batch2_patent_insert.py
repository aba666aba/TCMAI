#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 2 批中成药联网核实录入：感冒/发热/咽痛/咳嗽类（10 个）。
零幻觉：仅写入有权威出处的字段；分小批量、先核对再入库。
"""
import sqlite3, os, sys

DB = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "tcm_knowledge.db"))
SRC = ("据腾讯医典、中国医药信息查询平台(国家中医药管理局认证)、国家药监局(NMPA)非处方药说明书范本、"
       "科普中国/中国中医药网、同仁堂/以岭/京都念慈菴等官网等权威来源核对录入（2026-11-08）")

NEW_MEDS = [
    {
        "name": "小柴胡颗粒",
        "category": "表里双解剂",
        "composition": "柴胡、姜半夏、黄芩、党参、甘草、生姜、大枣",
        "functions": "解表散热，疏肝和胃",
        "indications": "用于外感病，邪犯少阳证，症见寒热往来、胸胁苦满、食欲不振、心烦喜呕、口苦咽干",
        "usage_dosage": "开水冲服。一次1-2袋，一日3次",
        "contraindications": "对本品及所含成份过敏者禁用；风寒表证者不宜使用",
        "source": SRC,
    },
    {
        "name": "清开灵颗粒",
        "category": "清热剂",
        "composition": "胆酸、珍珠母、猪去氧胆酸、栀子、水牛角、板蓝根、黄芩苷、金银花",
        "functions": "清热解毒，镇静安神",
        "indications": "用于外感风热所致发热，烦躁不安，咽喉肿痛；及上呼吸道感染、病毒性感冒、急性咽炎见上述证候者",
        "usage_dosage": "口服。一次1-2袋，一日2-3次。每袋装3g(含黄芩苷20mg)",
        "contraindications": "孕妇禁用；对本品及所含成份过敏者禁用；风寒感冒者不适用",
        "source": SRC,
    },
    {
        "name": "感冒灵颗粒",
        "category": "解表剂（中西药复方）",
        "composition": "三叉苦、金盏银盘、野菊花、岗梅、咖啡因、对乙酰氨基酚、马来酸氯苯那敏(扑尔敏)、薄荷油",
        "functions": "解热镇痛",
        "indications": "用于感冒引起的头痛、发热、鼻塞、流涕、咽痛",
        "usage_dosage": "开水冲服，一次10g(1袋)，一日3次。每袋含对乙酰氨基酚0.2g",
        "contraindications": "严重肝肾功能不全者禁用；对本品及所含成份过敏者禁用；膀胱颈梗阻、甲亢、青光眼、高血压和前列腺肥大者慎用",
        "source": SRC,
    },
    {
        "name": "抗病毒口服液",
        "category": "清热解毒剂",
        "composition": "板蓝根、石膏、芦根、生地黄、郁金、知母、石菖蒲、广藿香、连翘",
        "functions": "清热祛湿，凉血解毒",
        "indications": "用于风热感冒、流感",
        "usage_dosage": "口服，一次10ml，一日2-3次（早饭前和午、晚饭后各服一次）。每支装10ml",
        "contraindications": "孕妇、哺乳期妇女禁用；对本品及所含成份过敏者禁用；脾胃虚寒泄泻者慎服",
        "source": SRC,
    },
    {
        "name": "蓝芩口服液",
        "category": "清热解毒药",
        "composition": "板蓝根、黄芩、栀子、黄柏、胖大海",
        "functions": "清热解毒，利咽消肿",
        "indications": "用于急性咽炎、肺胃实热证所致的咽痛、咽干、咽部灼热",
        "usage_dosage": "口服，一次10ml，一日3次。每支装10ml",
        "contraindications": "对本品过敏者禁用；脾虚大便溏者、属风寒感冒咽痛者、孕妇慎用",
        "source": SRC,
    },
    {
        "name": "蒲地蓝消炎口服液",
        "category": "清热解毒剂",
        "composition": "蒲公英、苦地丁、板蓝根、黄芩",
        "functions": "清热解毒，消肿利咽（抗炎消肿）",
        "indications": "用于疖肿、咽炎、扁桃体炎等",
        "usage_dosage": "口服。一次10ml，一日3次，小儿酌减。如有沉淀，摇匀后服用。每支装10ml",
        "contraindications": "肝功能不全者禁用；对本品及所含成份过敏者禁用；孕妇、脾胃虚寒者慎用",
        "source": SRC,
    },
    {
        "name": "急支糖浆",
        "category": "化痰止咳平喘药",
        "composition": "鱼腥草、金荞麦、四季青、麻黄、紫菀、前胡、枳壳、甘草",
        "functions": "清热化痰，宣肺止咳",
        "indications": "用于外感风热所致的咳嗽，症见发热、恶寒、胸膈满闷、咳嗽咽痛；急性支气管炎、慢性支气管炎急性发作见上述证候者",
        "usage_dosage": "口服。一次20-30ml，一日3-4次；儿童按年龄递减",
        "contraindications": "对本品过敏者禁用；高血压、心脏病患者慎用；运动员慎用",
        "source": SRC,
    },
    {
        "name": "蜜炼川贝枇杷膏",
        "category": "化痰止咳平喘药（润肺止咳）",
        "composition": "川贝母、枇杷叶、桔梗、陈皮、法半夏、北沙参、五味子、款冬花、杏仁水、薄荷脑（辅料蔗糖、蜂蜜）",
        "functions": "清热润肺，止咳平喘，理气化痰",
        "indications": "适用于肺燥之咳嗽，痰多，胸闷，咽喉痛痒，声音沙哑",
        "usage_dosage": "口服，一次22g(约一汤匙)，一日3次。每瓶装138g",
        "contraindications": "糖尿病患者忌用；对本品及所含成份过敏者禁用",
        "source": SRC,
    },
    {
        "name": "金嗓子喉片",
        "category": "清热解毒药（利咽）",
        "composition": "薄荷脑、金银花、西青果、桉油、石斛、罗汉果、橘红、八角茴香油",
        "functions": "疏风清热，解毒利咽，芳香辟秽",
        "indications": "适用于改善急性咽炎所致的咽喉肿痛，干燥灼热，声音嘶哑",
        "usage_dosage": "含服，一次1片，一日6次。每片重2g",
        "contraindications": "对本品过敏者禁用；孕妇、糖尿病患者、儿童慎用；脾虚大便溏者、风寒感冒咽痛者慎用",
        "source": SRC,
    },
    {
        "name": "正柴胡饮颗粒",
        "category": "辛温解表药",
        "composition": "柴胡、陈皮、防风、赤芍、甘草、生姜",
        "functions": "发散风寒，解热止痛",
        "indications": "用于外感风寒初起：发热恶寒，无汗，头痛，鼻塞，喷嚏，咽痒咳嗽，四肢酸痛；流感初起、轻度上呼吸道感染见上述症候者",
        "usage_dosage": "开水冲服，一次5g(1袋)，一日3次。每袋装5g",
        "contraindications": "孕妇禁用；对本品及所含成份过敏者禁用；风热感冒者不适用",
        "source": SRC,
    },
]


def main():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    before = cur.execute("SELECT count(*) FROM patent_medicines").fetchone()[0]
    print(f"[before] patent_medicines 行数 = {before}")

    inserted = skipped = 0
    for m in NEW_MEDS:
        if cur.execute("SELECT 1 FROM patent_medicines WHERE name=?", (m["name"],)).fetchone():
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
    print(f"\n[after] patent_medicines 行数 = {after}  (新增 {inserted}, 跳过 {skipped})")
    con.close()
    return inserted


if __name__ == "__main__":
    n = main()
    sys.exit(0 if n >= 0 else 1)
