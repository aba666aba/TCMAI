# -*- coding: utf-8 -*-
"""生成第 9 批中成药本地草稿（补益/安神/扶正类，10 个）。

数据来源：腾讯医典 / 中国医药信息查询平台(国家中医药管理局认证) /
中国中医药网 / 科普中国 / 百度百科(国家中医药管理局名词术语项目) /
同仁堂官网 / 东阿阿胶官网，均经 WebSearch 联网核对（2026-11-08）。
零幻觉：只录有出处字段。

本批调整说明：
- 跳过"香砂六君丸"（与已有"香砂六君子丸"同药异名，后者为规范全名）；
- 补位"人参养荣丸"。
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "data", "exports", "drafts", "batch9_patent_draft.json")

SOURCE = ("腾讯医典 / 中国医药信息查询平台(国家中医药管理局认证) / 中国中医药网 / "
          "科普中国 / 百度百科(国家中医药管理局名词术语项目) / 同仁堂官网 / 东阿阿胶官网"
          "（2026-11-08 联网核对）")

RECORDS = [
    {
        "name": "桂附地黄丸",
        "category": "补益类中成药",
        "composition": "肉桂、附子(制)、熟地黄、山茱萸(制)、牡丹皮、山药、茯苓、泽泻",
        "functions": "温补肾阳",
        "indications": "用于肾阳不足，腰膝痿冷，小便不利或反多，痰饮喘咳。",
        "usage_dosage": "口服。大蜜丸一次1丸，小蜜丸一次9g，水蜜丸一次6g，一日2次。",
        "contraindications": "孕妇忌服；感冒发热病人不宜服用；阴虚内热者不适用；不宜长期服用；不宜与赤石脂及其制剂同用。",
    },
    {
        "name": "麦味地黄丸",
        "category": "补益类中成药",
        "composition": "熟地黄、山茱萸(制)、山药、茯苓、牡丹皮、泽泻、麦冬、五味子",
        "functions": "滋肾养肺",
        "indications": "用于肺肾阴亏，潮热盗汗，咽干，眩晕耳鸣，腰膝酸软。",
        "usage_dosage": "口服。大蜜丸一次1丸，一日2次；水蜜丸一次6g，一日2次。",
        "contraindications": "外感风热、喉痛、里热盛者忌服；痰浊阻肺咳嗽痰多而白稀、便溏者忌服；感冒发热病人不宜服用。",
    },
    {
        "name": "明目地黄丸",
        "category": "眼科类中成药",
        "composition": "熟地黄、酒萸肉、牡丹皮、山药、茯苓、泽泻、枸杞子、菊花、当归、白芍、蒺藜、煅石决明",
        "functions": "滋肾，养肝，明目",
        "indications": "用于肝肾阴虚，目涩畏光，视物模糊，迎风流泪。",
        "usage_dosage": "口服。大蜜丸一次1丸，一日2次；浓缩水丸一次8-10丸，一日3次；水蜜丸一次6g，一日2次。",
        "contraindications": "暴发火眼者忌用(眼白充血发红、怕光、流泪、眼屎多)；感冒时不宜服用；忌烟、酒及辛辣刺激性食物。",
    },
    {
        "name": "人参健脾丸",
        "category": "补益类中成药",
        "composition": "人参、白术(麸炒)、茯苓、山药、陈皮、木香、砂仁、炙黄芪、当归、酸枣仁(炒)、远志(制)",
        "functions": "健脾益气，和胃止泻",
        "indications": "用于脾胃虚弱所致的饮食不化、脘闷嘈杂、恶心呕吐、腹痛便溏、不思饮食、体弱倦怠。",
        "usage_dosage": "口服。水蜜丸一次8g，大蜜丸一次2丸，一日2次。",
        "contraindications": "感冒发热病人不宜服用；忌不易消化食物。",
    },
    {
        "name": "十全大补丸",
        "category": "补益类中成药",
        "composition": "党参、炒白术、茯苓、炙甘草、当归、川芎、酒白芍、熟地黄、炙黄芪、肉桂",
        "functions": "温补气血",
        "indications": "用于气血两虚，面色苍白，气短心悸，头晕自汗，体倦乏力，四肢不温，月经量多。",
        "usage_dosage": "口服。水蜜丸一次6g，小蜜丸一次9g，大蜜丸一次1丸，一日2次。",
        "contraindications": "孕妇忌用；身体壮实不虚者忌服；外感风寒、风热，实热内盛者不宜服用；不宜与藜芦、赤石脂或其制剂同服；不宜和感冒类药同时服用。",
    },
    {
        "name": "八珍颗粒",
        "category": "补益类中成药",
        "composition": "白芍、白术、川芎、当归、党参、茯苓、甘草、熟地黄",
        "functions": "补气益血",
        "indications": "用于气血两亏，面色萎黄，食欲不振，四肢乏力，月经过多。",
        "usage_dosage": "开水冲服，一次1袋，一日2次。宜饭前服用或进食同时服。",
        "contraindications": "孕妇慎用；咳嗽痰多、脘腹胀痛、纳食不消、腹胀便溏者忌服(药性粘腻碍消化)；不宜和感冒类药同时服用；不宜与藜芦或其制剂同服。",
    },
    {
        "name": "复方阿胶浆",
        "category": "补益类中成药",
        "composition": "阿胶、红参、熟地黄、党参、山楂",
        "functions": "补气养血",
        "indications": "用于气血两虚，头晕目眩，心悸失眠，食欲不振及贫血。",
        "usage_dosage": "口服，一次20ml，一日3次。宜饭前服用。",
        "contraindications": "对本品过敏者禁用；感冒病人不宜服用；脾胃虚弱、呕吐泄泻、腹胀便溏、咳嗽痰多者慎用；不宜同服藜芦、五灵脂、皂荚或其制剂，不宜喝茶和吃萝卜。",
    },
    {
        "name": "芪参益气滴丸",
        "category": "理血类中成药",
        "composition": "黄芪、丹参、三七、降香",
        "functions": "益气通脉，活血止痛",
        "indications": "用于气虚血瘀型胸痹，症见胸闷胸痛、气短乏力、心悸、面色少华、自汗，舌质暗或紫暗或有瘀斑；冠心病心绞痛见上述证候者。",
        "usage_dosage": "餐后半小时服用，一次1袋，一日3次。4周为一疗程或遵医嘱。",
        "contraindications": "对本品或所含成分过敏者禁用；孕妇慎用。",
    },
    {
        "name": "稳心颗粒",
        "category": "理血类中成药",
        "composition": "党参、黄精、三七、琥珀、甘松",
        "functions": "益气养阴，活血化瘀",
        "indications": "用于气阴两虚，心脉瘀阻所致的心悸不宁、气短乏力、胸闷胸痛；室性早搏、房性早搏见上述证候者。",
        "usage_dosage": "开水冲服，一次1袋，一日3次或遵医嘱。用前请将药液充分搅匀，勿将杯底药粉丢弃。",
        "contraindications": "对本品及所含成份过敏者禁用；缓慢性心律失常禁用；孕妇慎用；含党参，不宜与藜芦同用；忌烟酒、浓茶。",
    },
    {
        "name": "人参养荣丸",
        "category": "补益类中成药",
        "composition": "人参、白术(土炒)、茯苓、炙甘草、当归、熟地黄、白芍(麸炒)、炙黄芪、陈皮、远志(制)、肉桂、五味子(酒蒸)",
        "functions": "温补气血",
        "indications": "用于心脾不足，气血两亏，形瘦神疲，食少便溏，病后虚弱。",
        "usage_dosage": "口服。大蜜丸一次1丸，一日1-2次；水蜜丸一次6g，一日1-2次。",
        "contraindications": "感冒发热病人不宜服用；忌不易消化食物。",
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
            "batch": "batch9",
            "category_theme": "补益/安神/扶正",
            "record_count": len(records),
            "generated_at": "2026-11-08 12:16:00",
            "verified_by": "WebSearch 联网核对权威来源（零幻觉，未核实字段留空不填）",
            "skipped_in_pool": {
                "香砂六君丸": "与已有『香砂六君子丸』同药异名(后者为规范全名)，避免重复",
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
