# -*- coding: utf-8 -*-
"""生成第 6 批中成药本地草稿（感冒/解表/清热利咽类，10 个）。

数据来源：腾讯医典 / 中国医药信息查询平台(国家中医药管理局认证) /
中国中医药网 / 科普中国 / 百度百科(国家中医药管理局名词术语项目)，
均经 WebSearch 联网核对（2026-11-08）。零幻觉：只录有出处字段。

本批调整说明：
- 跳过"板蓝根颗粒"（与已有"板蓝根冲剂"为同一药的新旧命名，成分均为板蓝根）；
- 跳过"川贝枇杷露"（与已有"川贝枇杷糖浆"同方异剂型，构成近似重复）；
- 补位"玄麦甘桔颗粒""复方草珊瑚含片"（同为缺口池清热利咽类）。
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "data", "exports", "drafts", "batch6_patent_draft.json")

SOURCE = ("腾讯医典 / 中国医药信息查询平台(国家中医药管理局认证) / 中国中医药网 / "
          "科普中国 / 百度百科(国家中医药管理局名词术语项目)（2026-11-08 联网核对）")

RECORDS = [
    {
        "name": "桑菊感冒片",
        "category": "解表类中成药",
        "composition": "桑叶、菊花、连翘、苦杏仁、桔梗、芦根、薄荷素油、甘草",
        "functions": "疏风清热，宣肺止咳",
        "indications": "用于风热感冒初起，头痛，咳嗽，口干，咽痛。",
        "usage_dosage": "口服，一次4-8片，一日2-3次。",
        "contraindications": "风寒感冒者不适用；忌烟酒及辛辣、生冷、油腻食物；不宜在服药期间同时服用滋补性中药；儿童、孕妇、哺乳期妇女、年老体弱及脾虚便溏者应在医师指导下服用。",
    },
    {
        "name": "银翘解毒片",
        "category": "解表类中成药",
        "composition": "金银花、连翘、薄荷、荆芥、淡豆豉、牛蒡子(炒)、桔梗、淡竹叶、甘草",
        "functions": "疏风解表，清热解毒",
        "indications": "用于风热感冒，症见发热头痛、咳嗽口干、咽喉疼痛。",
        "usage_dosage": "口服，一次4片，一日2-3次。",
        "contraindications": "风寒感冒者不适用；孕妇忌用；忌烟、酒及辛辣、生冷、油腻食物；不宜在服药期间同时服用滋补性中药。",
    },
    {
        "name": "羚羊感冒片",
        "category": "解表类中成药",
        "composition": "羚羊角、牛蒡子、淡豆豉、金银花、荆芥、连翘、淡竹叶、桔梗、薄荷素油、甘草",
        "functions": "清热解表",
        "indications": "用于流行性感冒，症见发热恶风、头痛头晕、咳嗽、胸闷、咽喉肿痛。",
        "usage_dosage": "口服。一次4-6片，一日2次。",
        "contraindications": "风寒感冒者不适用；忌烟酒及辛辣、生冷、油腻食物；不宜同服滋补性中药；儿童、孕妇、哺乳期妇女、年老体弱及脾虚便溏者应在医师指导下服用。",
    },
    {
        "name": "复方鲜竹沥液",
        "category": "化痰止咳类中成药",
        "composition": "鲜竹沥、鱼腥草、生半夏、生姜、枇杷叶、桔梗、薄荷素油",
        "functions": "清热化痰，止咳",
        "indications": "用于痰热咳嗽，痰黄黏稠。",
        "usage_dosage": "口服。一次20ml，一日2-3次。",
        "contraindications": "对本品及所含成份过敏者禁用；风寒咳嗽者不适用；本品含生半夏，应严格按用法用量服用，不宜过量或长期服用；儿童、孕妇、哺乳期妇女慎用。",
    },
    {
        "name": "蛇胆川贝液",
        "category": "化痰止咳类中成药",
        "composition": "蛇胆汁、平贝母、蔗糖、蜂蜜、杏仁水",
        "functions": "祛风止咳，除痰散结",
        "indications": "用于风热咳嗽，痰多，气喘，胸闷，咳痰不爽或久咳不止。",
        "usage_dosage": "口服，一次10ml，一日2次。",
        "contraindications": "对本品及所含成份过敏者禁用；孕妇、体质虚弱者慎用；含少量乙醇，对酒精过敏者慎用，服药期间不建议与头孢菌素类、甲硝唑、替硝唑等药联用；不宜与川乌、制川乌、草乌、制草乌、附子及其制剂同用。",
    },
    {
        "name": "橘红痰咳液",
        "category": "化痰止咳类中成药",
        "composition": "化橘红、百部(蜜炙)、茯苓、半夏(制)、白前、甘草、苦杏仁、五味子",
        "functions": "理气祛痰，润肺止咳",
        "indications": "用于痰浊阻肺所致的咳嗽、气喘、痰多；感冒、支气管炎、咽喉炎见上述证候者。",
        "usage_dosage": "口服，一次10-20ml，一日3次。",
        "contraindications": "风热者忌用；忌烟、酒及辛辣、生冷、油腻食物；不宜在服药期间同时服用滋补性中药。",
    },
    {
        "name": "牛黄上清片",
        "category": "清热泻火类中成药",
        "composition": "人工牛黄、薄荷、菊花、荆芥穗、白芷、川芎、栀子、黄连、黄柏、黄芩、大黄、连翘、赤芍、当归、地黄、桔梗、甘草、石膏、冰片",
        "functions": "清热泻火，散风止痛",
        "indications": "用于热毒内盛、风火上攻所致的头痛眩晕，目赤耳鸣，咽喉肿痛，口舌生疮，牙龈肿痛，大便燥结。",
        "usage_dosage": "口服，一次4片，一日2次。",
        "contraindications": "对本品过敏者禁用；孕妇慎用；忌食辛辣食物；不宜在服药期间同时服用温补性中成药；小儿、年老体弱者、大便溏软者应在医师指导下服用。",
    },
    {
        "name": "黄连上清片",
        "category": "清热泻火类中成药",
        "composition": "黄连、栀子、连翘、炒蔓荆子、防风、荆芥穗、白芷、黄芩、菊花、薄荷、酒大黄、黄柏、桔梗、川芎、石膏、旋覆花、甘草",
        "functions": "散风清热，泻火止痛",
        "indications": "用于风热上攻、肺胃热盛所致的头晕目眩，暴发火眼，牙齿疼痛，口舌生疮，咽喉肿痛，耳痛耳鸣、大便秘结，小便短赤。",
        "usage_dosage": "口服。一次6片，一日2次。",
        "contraindications": "孕妇忌服；脾胃虚寒者禁用；对本品及所含成份过敏者禁用；忌食辛辣；不宜同服温补性中成药；不宜长期服用。",
    },
    {
        "name": "玄麦甘桔颗粒",
        "category": "清热滋阴类中成药",
        "composition": "玄参、麦冬、甘草、桔梗",
        "functions": "清热滋阴，祛痰利咽",
        "indications": "用于阴虚火旺，虚火上浮，口鼻干燥，咽喉肿痛。",
        "usage_dosage": "开水冲服。一次1袋(10g)，一日3-4次。",
        "contraindications": "对本品过敏者禁用；忌烟酒、辛辣、鱼腥食物；儿童、孕妇、哺乳期妇女、年老体弱、脾虚便溏者应在医师指导下服用；糖尿病患者慎用(含蔗糖型)。",
    },
    {
        "name": "复方草珊瑚含片",
        "category": "清热利咽类中成药",
        "composition": "肿节风浸膏、薄荷脑、薄荷素油",
        "functions": "疏风清热，消肿止痛，清利咽喉",
        "indications": "用于外感风热所致的喉痹，症见咽喉肿痛、声哑失音；急性咽喉炎见上述证候者。",
        "usage_dosage": "含服。一次2片(规格0.44g)或一次1片(规格1.0g)，每隔2小时1次，一日6次。",
        "contraindications": "对本品过敏者禁用；忌烟酒、辛辣、鱼腥食物；不宜同服滋补性中药；声哑失音较重者应及时就医。",
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
            "batch": "batch6",
            "category_theme": "感冒/解表/清热利咽",
            "record_count": len(records),
            "generated_at": "2026-11-08 11:58:00",
            "verified_by": "WebSearch 联网核对权威来源（零幻觉，未核实字段留空不填）",
            "skipped_in_pool": {
                "板蓝根颗粒": "与已有『板蓝根冲剂』为同一药新旧命名(成分均板蓝根)，避免重复",
                "川贝枇杷露": "与已有『川贝枇杷糖浆』同方异剂型，避免近似重复",
            },
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
