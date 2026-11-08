# -*- coding: utf-8 -*-
"""生成第 7 批中成药本地草稿（清热/解毒/祛湿类，10 个）。

数据来源：腾讯医典 / 中国医药信息查询平台(国家中医药管理局认证) /
中国中医药网 / 科普中国 / 百度百科(国家中医药管理局名词术语项目)，
均经 WebSearch 联网核对（2026-11-08）。零幻觉：只录有出处字段。
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "data", "exports", "drafts", "batch7_patent_draft.json")

SOURCE = ("腾讯医典 / 中国医药信息查询平台(国家中医药管理局认证) / 中国中医药网 / "
          "科普中国 / 百度百科(国家中医药管理局名词术语项目)（2026-11-08 联网核对）")

RECORDS = [
    {
        "name": "防风通圣丸",
        "category": "表里双解类中成药",
        "composition": "防风、荆芥穗、薄荷、麻黄、大黄、芒硝、栀子、滑石、桔梗、石膏、川芎、当归、白芍、黄芩、连翘、甘草、白术(炒)",
        "functions": "解表通里，清热解毒",
        "indications": "用于外寒内热，表里俱实，恶寒壮热，头痛咽干，小便短赤，大便秘结，瘰疬初起，风疹湿疮。",
        "usage_dosage": "口服。一次6g，一日2次。",
        "contraindications": "孕妇慎用；高血压、心脏病患者慎用；忌烟、酒及辛辣、油腻、鱼虾海鲜类食物；不宜同服滋补性中药；不宜长期服用；运动员慎用。",
    },
    {
        "name": "夏桑菊颗粒",
        "category": "清热解毒类中成药",
        "composition": "夏枯草、野菊花、桑叶",
        "functions": "清肝明目，疏风散热，除湿痹，解疮毒",
        "indications": "用于风热感冒，目赤头痛，头晕耳鸣，咽喉肿痛，疔疮肿毒等症，并可作清凉饮料。",
        "usage_dosage": "口服，一次1-2袋(10-20g)，一日3次。",
        "contraindications": "风寒感冒者不适用；脾胃虚寒(腹痛泄泻)者慎用；小儿、年老体弱者、孕妇应在医师指导下服用；忌烟酒及辛辣、生冷、油腻食物；不宜同服滋补性中药。",
    },
    {
        "name": "二妙丸",
        "category": "清热除湿类中成药",
        "composition": "苍术(炒)、黄柏(炒)",
        "functions": "燥湿清热",
        "indications": "用于湿热下注，白带，阴囊湿痒。",
        "usage_dosage": "口服，一次6-9g，一日2次。",
        "contraindications": "对本品过敏者禁用；忌烟酒、辛辣、油腻及腥发食物；儿童、孕妇、哺乳期妇女、年老体弱者应在医师指导下服用。",
    },
    {
        "name": "四妙丸",
        "category": "清热除湿类中成药",
        "composition": "苍术、牛膝、盐黄柏、薏苡仁",
        "functions": "清热利湿",
        "indications": "用于湿热下注所致的痹病，症见足膝红肿，筋骨疼痛。",
        "usage_dosage": "口服，一次6g，一日2次。",
        "contraindications": "孕妇慎用；脾胃虚弱者慎用；虚寒体质(畏寒肢冷、大便溏薄)、阴液亏虚者禁用；不适用于寒湿痹阻证。",
    },
    {
        "name": "五苓散",
        "category": "利水渗湿类中成药",
        "composition": "茯苓、泽泻、猪苓、肉桂、炒白术",
        "functions": "温阳化气，利湿行水",
        "indications": "用于阳不化气、水湿内停所致的水肿，症见小便不利、水肿腹胀、呕逆泄泻、渴不思饮。",
        "usage_dosage": "口服。一次6-9g，一日2次。",
        "contraindications": "对本品成分过敏者禁用；阴虚津液不足导致的口渴、小便不利者不适宜服用；湿热者忌用；服药期间忌食生冷、辛辣、油腻食物。",
    },
    {
        "name": "萆薢分清丸",
        "category": "扶正祛湿类中成药",
        "composition": "粉萆薢、石菖蒲、甘草、乌药、盐益智仁",
        "functions": "分清化浊，温肾利湿",
        "indications": "用于肾不化气，清浊不分所致的白浊、小便频数。",
        "usage_dosage": "口服。一次6-9g，一日2次。",
        "contraindications": "孕妇禁用；对本品过敏者禁用；忌食油腻、茶、醋及辛辣刺激性物。",
    },
    {
        "name": "排石颗粒",
        "category": "清热通淋类中成药",
        "composition": "连钱草、车前子(盐水炒)、关木通、徐长卿、石韦、瞿麦、忍冬藤、滑石、苘麻子、甘草",
        "functions": "清热利水，通淋排石",
        "indications": "用于下焦湿热所致的石淋，症见腰腹疼痛、排尿不畅或伴有血尿；泌尿系结石见上述证候者。",
        "usage_dosage": "开水冲服。一次1袋，一日3次；或遵医嘱。",
        "contraindications": "孕妇忌服；体虚者、脾虚便溏者慎用；服药期间多饮水并适当活动，忌油腻食物。",
    },
    {
        "name": "复方金钱草颗粒",
        "category": "清热通淋类中成药",
        "composition": "广金钱草、车前草、石韦、玉米须",
        "functions": "清热祛湿，利尿排石，消炎止痛",
        "indications": "用于湿热下注所致的热淋、石淋，症见尿频、尿急、尿痛、腰痛；泌尿系结石、尿路感染见上述证候者。",
        "usage_dosage": "开水冲服。一次1-2袋，一日3次。",
        "contraindications": "对本品及所含成份过敏者禁用；双肾结石或结石直径≥0.6cm或结石嵌顿时间长者不宜使用；妊娠期妇女慎用；严重腹泻者、严重肝肾功能异常者慎用；请勿超剂量、长期、连续用药；糖尿病患者慎用含糖型(可选无糖型)。",
    },
    {
        "name": "结石通片",
        "category": "清热通淋类中成药",
        "composition": "广金钱草、玉米须、石韦、鸡骨草、茯苓、车前草、海金沙草、白茅根",
        "functions": "清热利湿，通淋排石，镇痛止血",
        "indications": "用于泌尿系统感染，膀胱炎，肾炎水肿，尿路结石，血尿，淋漓浑浊，尿道灼痛等。",
        "usage_dosage": "口服，一次5片，一日3次。",
        "contraindications": "孕妇忌用；双肾结石或结石直径≥1.5cm或结石嵌顿时间长的病例忌用；肝郁气滞、脾肾亏虚、膀胱气化不行所致淋证不宜使用；忌食辛、燥、酸、辣食物。",
    },
    {
        "name": "清胃黄连丸",
        "category": "清热泻火类中成药",
        "composition": "黄连、石膏、桔梗、甘草、知母、玄参、地黄、牡丹皮、天花粉、连翘、栀子、黄柏、黄芩、赤芍",
        "functions": "清胃泻火，解毒消肿",
        "indications": "用于肺胃火盛所致的口舌生疮，齿龈、咽喉肿痛。",
        "usage_dosage": "口服。一次1袋(9g)，一日2次。",
        "contraindications": "对本品成分过敏者禁用；孕妇慎用；忌烟、酒及辛辣、油腻食物；不宜同服滋补性中药；服药后大便次数增多且不成形者应酌情减量。",
    },
]


def main():
    records = []
    for r in RECORDS:
        rec = dict(r)
        rec["source"] = SOURCE
        rec["status"] = "verified_draft"
        records.append(rec)

    payload = {
        "meta": {
            "batch": "batch7",
            "category_theme": "清热/解毒/祛湿",
            "record_count": len(records),
            "generated_at": "2026-11-08 12:08:00",
            "verified_by": "WebSearch 联网核对权威来源（零幻觉，未核实字段留空不填）",
            "status": "draft_pending_import",
        },
        "records": records,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("draft written ->", OUT)
    print("records:", len(records))


if __name__ == "__main__":
    main()
