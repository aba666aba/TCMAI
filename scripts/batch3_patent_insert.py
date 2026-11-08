# -*- coding: utf-8 -*-
"""第 3 批：消化/脾胃/理气类中成药补录（零幻觉，均经 WebSearch 联网核对权威来源）。
备份：data/tcm_knowledge.db.bak_prebatch3
来源标注：腾讯医典 / 中国医药信息查询平台(国家中医药管理局认证) / 科普中国·中国中医药网 / 药品说明书范本
"""
import sqlite3

DB = "data/tcm_knowledge.db"
SOURCE = ("据腾讯医典、中国医药信息查询平台(国家中医药管理局认证)、"
          "科普中国·中国中医药网、药品说明书范本等权威来源核对录入（2026-11-08）")

records = [
    {
        "name": "六味能消胶囊",
        "category": "消导剂",
        "composition": "大黄、诃子、干姜、藏木香、碱花、寒水石(平制)",
        "functions": "宽中理气，润肠通便，调节血脂",
        "indications": "用于胃脘胀痛、厌食、纳差及大便秘结；高脂血症及肥胖症",
        "usage_dosage": "口服。便秘、胃脘胀痛一次2粒，一日3次；高脂血症一次1粒，一日3次。老人及儿童遵医嘱。",
        "contraindications": "妊娠及哺乳期妇女忌用。",
    },
    {
        "name": "摩罗丹",
        "category": "疏肝和胃剂",
        "composition": ("百合、茯苓、玄参、乌药、泽泻、麦冬、当归、白术(麸炒)、茵陈、白芍、"
                        "石斛、九节菖蒲、川芎、三七、地榆、延胡索(醋炙)、蒲黄、鸡内金(炒香)"),
        "functions": "和胃降逆，健脾消胀，通络定痛",
        "indications": "用于慢性萎缩性胃炎及胃疼、胀满、痞闷、纳呆、嗳气、烧心等症",
        "usage_dosage": ("口服。大蜜丸一次1-2丸，一日3次，饭前用米汤或温开水送下；"
                         "浓缩丸一次16丸，一日3次。"),
        "contraindications": "孕妇慎用；对本品及所含成份过敏者禁用。",
    },
    {
        "name": "胃苏颗粒",
        "category": "疏肝和胃剂",
        "composition": "紫苏梗、香附、陈皮、香橼、佛手、枳壳、槟榔、炒鸡内金",
        "functions": "理气消胀，和胃止痛",
        "indications": ("用于气滞型胃脘痛，症见胃脘胀痛、窜及两胁、得嗳气或矢气则舒、"
                        "情绪郁怒则加重、胸闷食少、排便不畅及慢性胃炎见上述证候者"),
        "usage_dosage": "用适量开水冲服。一次1袋，一日3次。15天为一个疗程。",
        "contraindications": "孕妇忌服。",
    },
    {
        "name": "三九胃泰颗粒",
        "category": "疏肝和胃剂",
        "composition": "三叉苦、九里香、两面针、木香、黄芩、茯苓、地黄、白芍",
        "functions": "清热燥湿，行气活血，柔肝止痛",
        "indications": ("用于湿热内蕴、气滞血瘀所致的胃痛，症见脘腹隐痛、饱胀反酸、"
                        "恶心呕吐、嘈杂纳减；浅表性胃炎、萎缩性胃炎、糜烂性胃炎见上述证候者"),
        "usage_dosage": "用开水冲服，一次1袋，一日2次。",
        "contraindications": "对本品及所含成份过敏者禁用。",
    },
    {
        "name": "补脾益肠丸",
        "category": "健脾和胃剂",
        "composition": ("黄芪、党参(米炒)、砂仁、白芍、白术(土炒)、肉桂、延胡索(制)、"
                        "干姜(炮)、防风、木香、补骨脂(盐制)、赤石脂(煅)等15味"),
        "functions": "补中益气，健脾和胃，涩肠止泻",
        "indications": "用于脾虚泄泻症，症见腹泻腹痛、腹胀、肠鸣",
        "usage_dosage": "口服。一次6g，一日3次；儿童酌减；重症加量或遵医嘱。",
        "contraindications": "孕妇禁用；泄泻时腹部热胀痛者忌服。",
    },
    {
        "name": "保济丸",
        "category": "解表祛暑剂",
        "composition": ("钩藤、菊花、蒺藜、厚朴、木香、苍术、天花粉、广藿香、葛根、"
                        "化橘红、白芷、薏苡仁、稻芽、薄荷、茯苓、广东神曲"),
        "functions": "解表，祛湿，和中",
        "indications": ("用于暑湿感冒，症见发热头痛、腹痛腹泻、恶心呕吐、肠胃不适；"
                        "亦可用于晕车晕船"),
        "usage_dosage": "口服。一次1.85-3.7g，一日3次。",
        "contraindications": "孕妇忌用；外感燥热者不宜服用；对本品过敏者禁用。",
    },
    {
        "name": "大山楂丸",
        "category": "消导剂",
        "composition": "山楂、六神曲(麸炒)、炒麦芽",
        "functions": "开胃消食",
        "indications": "用于食积内停所致的食欲不振、消化不良、脘腹胀闷",
        "usage_dosage": "口服，一次1-2丸，一日1-3次；小儿酌减。",
        "contraindications": "脾胃虚弱、无积滞而食欲不振者不适用；对本品过敏者禁用。",
    },
    {
        "name": "温胃舒颗粒",
        "category": "温中散寒剂",
        "composition": ("党参、附子(制)、黄芪(炙)、肉桂、山药、肉苁蓉(制)、白术(炒)、"
                        "山楂(炒)、乌梅、砂仁、陈皮、补骨脂"),
        "functions": "温胃止痛",
        "indications": "用于慢性胃炎，胃脘凉痛，饮食生冷，受寒痛甚",
        "usage_dosage": "开水冲服，一次1-2袋，一日2次。",
        "contraindications": "胃大出血时忌用；孕妇忌用。",
    },
    {
        "name": "荜铃胃痛颗粒",
        "category": "理气剂",
        "composition": ("荜澄茄、川楝子、延胡索(醋制)、大黄(酒制)、黄连、吴茱萸、"
                        "香附(醋制)、香橼、佛手、海螵蛸、瓦楞子(煅)"),
        "functions": "行气活血，和胃止痛",
        "indications": "用于气滞血瘀所致的胃脘痛；慢性胃炎见有上述证候者",
        "usage_dosage": "开水冲服。一次5g，一日3次。",
        "contraindications": "孕妇禁用。",
    },
    {
        "name": "枳术宽中胶囊",
        "category": "疏肝和胃剂",
        "composition": "白术(炒)、枳实、柴胡、山楂",
        "functions": "健脾和胃，理气消痞",
        "indications": ("用于胃痞(脾虚气滞)，症见呕吐、反胃、纳呆、反酸等，"
                        "以及功能性消化不良见以上症状者"),
        "usage_dosage": "口服，一次3粒，一日3次，疗程为2周。",
        "contraindications": "对本药任何成分过敏者禁用。",
    },
]

con = sqlite3.connect(DB)
con.execute("PRAGMA foreign_keys=OFF")
cur = con.cursor()
before = cur.execute("SELECT count(*) FROM patent_medicines").fetchone()[0]

added, skipped = [], []
for r in records:
    exists = cur.execute(
        "SELECT 1 FROM patent_medicines WHERE name=?", (r["name"],)
    ).fetchone()
    if exists:
        skipped.append(r["name"])
        continue
    cur.execute(
        """INSERT INTO patent_medicines
           (name, category, composition, functions, indications, usage_dosage, contraindications, source)
           VALUES (?,?,?,?,?,?,?,?)""",
        (r["name"], r["category"], r["composition"], r["functions"],
         r["indications"], r["usage_dosage"], r["contraindications"], SOURCE),
    )
    added.append(r["name"])

con.commit()
after = cur.execute("SELECT count(*) FROM patent_medicines").fetchone()[0]
con.close()

print(f"before={before}  after={after}  added={len(added)}  skipped={len(skipped)}")
print("新增:", added)
if skipped:
    print("跳过(已存在):", skipped)
