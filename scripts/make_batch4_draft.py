# -*- coding: utf-8 -*-
"""
第 4 批（补益/安神/妇科类）中成药联网补充 —— 生成本地草稿 JSON。
数据全部来自 WebSearch 联网核对（腾讯医典 / 中国医药信息查询平台(国家中医药管理局认证)
/ 中国中医药网(科普中国) / 同仁堂·白云山·中国中药官网 / 百度百科(中医药名词术语项目)），
零幻觉，只填有出处的字段。
此脚本只【生成本地草稿】，不写入数据库。导入由 import_draft_batch4.py 在用户确认后执行。
"""
import os, json, datetime

OUT_DIR = os.path.join("data", "exports", "drafts")
OUT_FILE = os.path.join(OUT_DIR, "batch4_patent_draft.json")

SRC = ("腾讯医典 / 中国医药信息查询平台(国家中医药管理局认证) / 中国中医药网(科普中国) / "
       "同仁堂·白云山·中国中药官网 / 百度百科(中医药名词术语成果转化与规范推广项目)"
       "（2026-11-08 WebSearch 联网核对录入）")

# 每条：name, category, composition, functions, indications, usage_dosage, contraindications
records = [
    {
        "name": "归脾丸",
        "category": "补益类中成药",
        "composition": "党参、炒白术、炙黄芪、炙甘草、茯苓、制远志、炒酸枣仁、龙眼肉、当归、木香、大枣(去核)",
        "functions": "益气健脾，养血安神",
        "indications": "心脾两虚，气短心悸，失眠多梦，头昏头晕，肢倦乏力，食欲不振",
        "usage_dosage": "用温开水或生姜汤送服。水蜜丸一次6克，一日3次；小蜜丸一次9克，一日3次；大蜜丸一次1丸，一日3次",
        "contraindications": "对本品过敏者禁用；感冒发热病人不宜服用；有口渴、尿黄、便秘等内热表现者不宜服用；儿童、孕妇、哺乳期妇女应在医师指导下服用",
    },
    {
        "name": "天王补心丹",
        "category": "安神类中成药",
        "composition": "丹参、当归、石菖蒲、党参、茯苓、五味子、麦冬、天冬、地黄、玄参、远志(制)、酸枣仁(炒)、柏子仁、桔梗、甘草、朱砂",
        "functions": "滋阴养血，补心安神",
        "indications": "心阴不足，心悸健忘，失眠多梦，大便干燥",
        "usage_dosage": "口服。大蜜丸一次1丸(9g)，一日2次；水蜜丸一次6克，一日2次",
        "contraindications": "处方中含朱砂，不宜过量久服，肝肾功能不全者慎用；脾胃虚弱、纳食欠佳、大便不实者不宜长期服用",
    },
    {
        "name": "杞菊地黄丸",
        "category": "补益类中成药",
        "composition": "枸杞子、菊花、熟地黄、酒萸肉、牡丹皮、山药、茯苓、泽泻",
        "functions": "滋肾养肝",
        "indications": "肝肾阴亏，眩晕耳鸣，羞明畏光，迎风流泪，视物昏花",
        "usage_dosage": "口服。水蜜丸一次6克，一日2次；浓缩丸一次8丸，一日3次；小蜜丸一次9克，一日2次；大蜜丸一次1丸，一日2次",
        "contraindications": "对本品过敏者禁用，过敏体质者慎用；感冒发热病人不宜服用；有高血压、心脏病、肝病、糖尿病、肾病等慢性病严重者应在医师指导下服用",
    },
    {
        "name": "妇科千金片",
        "category": "妇科类中成药",
        "composition": "千斤拔、金樱根、穿心莲、功劳木、单面针、当归、鸡血藤、党参",
        "functions": "清热除湿，益气化瘀",
        "indications": "湿热瘀阻所致的带下病、腹痛，症见带下量多、色黄质稠，小腹疼痛，腰骶酸痛，神疲乏力；慢性盆腔炎见有上述证候者",
        "usage_dosage": "口服。一次6片，一日3次",
        "contraindications": "孕妇禁用；对本品及所含成分过敏者禁用；气滞血瘀、寒凝血瘀者忌用；糖尿病患者慎用",
    },
    {
        "name": "养血安神片",
        "category": "安神类中成药",
        "composition": "仙鹤草、墨旱莲、鸡血藤、熟地黄、地黄、合欢皮、首乌藤",
        "functions": "滋阴养血，宁心安神",
        "indications": "阴虚血少，头眩心悸，失眠健忘",
        "usage_dosage": "口服。一次5片，一日3次",
        "contraindications": "对本品过敏者禁用，过敏体质者慎用；脾胃虚弱者宜在饭后服用，以减轻药物对肠胃的刺激",
    },
    {
        "name": "枣仁安神胶囊",
        "category": "安神类中成药",
        "composition": "炒酸枣仁、丹参、醋五味子",
        "functions": "养血安神",
        "indications": "心血不足所致的失眠、健忘、心烦、头晕；神经衰弱症见上述证候者",
        "usage_dosage": "口服。一次5粒，一日1次，临睡前服用",
        "contraindications": "孕妇慎用；由于消化不良所导致的睡眠差者忌用；对本品过敏者禁用，过敏体质者慎用",
    },
    {
        "name": "生脉饮口服液",
        "category": "补益类中成药",
        "composition": "人参(或红参/党参)、麦冬、五味子",
        "functions": "益气，养阴生津",
        "indications": "气阴两亏，心悸气短，自汗",
        "usage_dosage": "口服。一次10毫升(1支)，一日3次",
        "contraindications": "感冒发热病人不宜服用；对本品过敏者禁用，过敏体质者慎用；糖尿病患者及有高血压、心脏病、肝病、肾病等慢性病严重者应在医师指导下服用",
    },
    {
        "name": "坤泰胶囊",
        "category": "妇科类中成药",
        "composition": "熟地黄、黄连、白芍、黄芩、阿胶、茯苓",
        "functions": "滋阴清热，安神除烦",
        "indications": "绝经期前后诸证。阴虚火旺者，症见潮热面红、自汗盗汗，心烦不宁，失眠多梦，头晕耳鸣，腰膝酸软，手足心热；妇女卵巢功能衰退更年期综合症见上述表现者",
        "usage_dosage": "口服。一次4粒，一日3次，2-4周为一疗程，或遵医嘱",
        "contraindications": "阳虚体质者忌用；对本品过敏者禁用，过敏体质者慎用；高血压、心脏病、肾病及脾胃虚弱者请在医师指导下服用",
    },
    {
        "name": "定坤丹",
        "category": "妇科类中成药",
        "composition": "红参、鹿茸、西红花、三七、白芍、熟地黄、当归、白术、枸杞子、黄芩、香附、茺蔚子、川芎、鹿角霜、阿胶、延胡索、鸡血藤膏、红花、益母草、五灵脂、茯苓、柴胡、乌药、砂仁、杜仲、干姜、细辛、川牛膝、肉桂、炙甘草",
        "functions": "滋补气血，调经舒郁",
        "indications": "气血两虚、气滞血瘀所致的月经不调、行经腹痛",
        "usage_dosage": "口服。一次3.5-7克，一日2次；或遵医嘱",
        "contraindications": "孕妇禁用；对本品及所含成分过敏者禁用；伤风感冒时停服；脾胃虚弱者慎用",
    },
]

# 组装草稿
draft = {
    "meta": {
        "batch": "batch4",
        "topic": "补益/安神/妇科类中成药补充",
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "draft",  # 待用户确认后导入数据库
        "source": SRC,
        "record_count": len(records),
        "note": "本文件为联网核对后的本地草稿，尚未写入 data/tcm_knowledge.db；"
                "确认无误后由 import_draft_batch4.py 导入。",
    },
    "records": [],
}
for r in records:
    item = dict(r)
    item["source"] = SRC
    item["status"] = "draft"
    draft["records"].append(item)

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(draft, f, ensure_ascii=False, indent=2)

print("已生成本地草稿:", OUT_FILE)
print("记录数:", len(records))
for r in records:
    print("  -", r["name"], "|", r["category"])
