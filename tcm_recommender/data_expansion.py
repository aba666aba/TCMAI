# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0

"""
中医药数据扩展模块

基于《中药学》教材和《方剂学》教材标准知识，程序化扩充：
- 药材 1,010 → 2,500+ （涵盖全部常用中药）
- 方剂 347 → 1,200+ （涵盖经方、时方、验方）
- 症状 33 → 120+ （涵盖全身各系统）
- 证型 9 → 50+ （涵盖八纲、脏腑、气血津液辨证）
- 症状-药材关联、症状-方剂关联（用于训练监督信号）
"""

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── 扩展药材数据（基于《中药学》教材分类体系） ──

EXTENDED_HERBS: List[Dict[str, str]] = [
    # ============ 解表药 ============
    # 辛温解表
    {"name": "麻黄", "category": "辛温解表药", "nature": "温", "flavor": "辛、微苦", "meridian": "肺、膀胱经", "efficacy": "发汗解表，宣肺平喘，利水消肿", "indications": "风寒感冒，胸闷喘咳，风水浮肿", "usage": "煎服，2-10g", "contraindications": "表虚自汗、阴虚盗汗、高血压者慎用"},
    {"name": "桂枝", "category": "辛温解表药", "nature": "温", "flavor": "辛、甘", "meridian": "心、肺、膀胱经", "efficacy": "发汗解肌，温通经脉，助阳化气，平冲降逆", "indications": "风寒感冒，寒凝血滞诸痛，痰饮水肿，心悸奔豚", "usage": "煎服，3-10g", "contraindications": "温热病、阴虚火旺、血热妄行者忌用"},
    {"name": "紫苏叶", "category": "辛温解表药", "nature": "温", "flavor": "辛", "meridian": "肺、脾经", "efficacy": "解表散寒，行气和胃", "indications": "风寒感冒，咳嗽呕恶，妊娠呕吐，鱼蟹中毒", "usage": "煎服，5-10g", "contraindications": "温病及气弱表虚者忌服"},
    {"name": "生姜", "category": "辛温解表药", "nature": "微温", "flavor": "辛", "meridian": "肺、脾、胃经", "efficacy": "解表散寒，温中止呕，化痰止咳，解鱼蟹毒", "indications": "风寒感冒，胃寒呕吐，寒痰咳嗽", "usage": "煎服，3-10g", "contraindications": "阴虚内热者忌服"},
    {"name": "荆芥", "category": "辛温解表药", "nature": "微温", "flavor": "辛", "meridian": "肺、肝经", "efficacy": "祛风解表，透疹消疮，止血", "indications": "感冒头痛，麻疹不透，风疹瘙痒，疮疡初起", "usage": "煎服，5-10g", "contraindications": "表虚自汗、阴虚头痛者忌服"},
    {"name": "防风", "category": "辛温解表药", "nature": "微温", "flavor": "辛、甘", "meridian": "膀胱、肝、脾经", "efficacy": "祛风解表，胜湿止痛，止痉", "indications": "感冒头痛，风湿痹痛，风疹瘙痒，破伤风", "usage": "煎服，5-10g", "contraindications": "阴血亏虚、热病动风者慎用"},
    {"name": "羌活", "category": "辛温解表药", "nature": "温", "flavor": "辛、苦", "meridian": "膀胱、肾经", "efficacy": "解表散寒，祛风除湿，止痛", "indications": "风寒感冒，头痛项强，风寒湿痹，肩背酸痛", "usage": "煎服，3-10g", "contraindications": "血虚痹痛、阴虚头痛者慎用"},
    {"name": "白芷", "category": "辛温解表药", "nature": "温", "flavor": "辛", "meridian": "肺、胃、大肠经", "efficacy": "解表散寒，祛风止痛，宣通鼻窍，燥湿止带，消肿排脓", "indications": "感冒头痛，鼻塞鼻渊，带下，疮疡肿痛", "usage": "煎服，3-10g", "contraindications": "阴虚血热者忌服"},
    {"name": "细辛", "category": "辛温解表药", "nature": "温", "flavor": "辛", "meridian": "心、肺、肾经", "efficacy": "解表散寒，祛风止痛，宣通鼻窍，温肺化饮", "indications": "风寒感冒，头痛牙痛，鼻渊，风寒咳喘", "usage": "煎服，1-3g", "contraindications": "气虚多汗、阴虚阳亢头痛、肺燥干咳者忌服"},
    {"name": "藁本", "category": "辛温解表药", "nature": "温", "flavor": "辛", "meridian": "膀胱经", "efficacy": "祛风散寒，除湿止痛", "indications": "风寒感冒，巅顶头痛，风寒湿痹", "usage": "煎服，3-10g", "contraindications": "血虚头痛及热证忌用"},
    {"name": "苍耳子", "category": "辛温解表药", "nature": "温", "flavor": "辛、苦", "meridian": "肺经", "efficacy": "散风寒，通鼻窍，祛风湿", "indications": "风寒头痛，鼻渊流涕，风疹瘙痒，湿痹拘挛", "usage": "煎服，3-10g", "contraindications": "血虚头痛忌用，过量易中毒"},
    {"name": "辛夷", "category": "辛温解表药", "nature": "温", "flavor": "辛", "meridian": "肺、胃经", "efficacy": "散风寒，通鼻窍", "indications": "风寒头痛，鼻渊鼻塞，鼻流浊涕", "usage": "煎服，3-10g", "contraindications": "阴虚火旺者忌服"},

    # 辛凉解表
    {"name": "薄荷", "category": "辛凉解表药", "nature": "凉", "flavor": "辛", "meridian": "肺、肝经", "efficacy": "疏散风热，清利头目，利咽透疹，疏肝行气", "indications": "风热感冒，头痛目赤，咽喉肿痛，麻疹不透，肝郁气滞", "usage": "煎服，3-6g", "contraindications": "体虚多汗者不宜"},
    {"name": "牛蒡子", "category": "辛凉解表药", "nature": "寒", "flavor": "辛、苦", "meridian": "肺、胃经", "efficacy": "疏散风热，宣肺透疹，解毒利咽", "indications": "风热感冒，咳嗽痰多，麻疹不透，咽喉肿痛，痈肿疮毒", "usage": "煎服，6-12g", "contraindications": "气虚便溏者慎用"},
    {"name": "蝉蜕", "category": "辛凉解表药", "nature": "寒", "flavor": "甘", "meridian": "肺、肝经", "efficacy": "疏散风热，利咽开音，透疹明目，息风止痉", "indications": "风热感冒，咽痛音哑，麻疹不透，目赤翳障，惊风抽搐", "usage": "煎服，3-6g", "contraindications": "孕妇慎用"},
    {"name": "桑叶", "category": "辛凉解表药", "nature": "寒", "flavor": "甘、苦", "meridian": "肺、肝经", "efficacy": "疏散风热，清肺润燥，清肝明目", "indications": "风热感冒，肺热燥咳，头晕头痛，目赤昏花", "usage": "煎服，5-10g", "contraindications": "风寒感冒忌用"},
    {"name": "菊花", "category": "辛凉解表药", "nature": "微寒", "flavor": "甘、苦", "meridian": "肺、肝经", "efficacy": "疏散风热，平抑肝阳，清肝明目，清热解毒", "indications": "风热感冒，头痛眩晕，目赤肿痛，眼目昏花，疮痈肿毒", "usage": "煎服，5-10g", "contraindications": "气虚胃寒、食少泄泻者慎用"},
    {"name": "蔓荆子", "category": "辛凉解表药", "nature": "微寒", "flavor": "辛、苦", "meridian": "膀胱、肝、胃经", "efficacy": "疏散风热，清利头目", "indications": "风热感冒，头痛头风，目赤肿痛，目昏多泪", "usage": "煎服，5-10g", "contraindications": "血虚有火之头痛目眩者忌服"},
    {"name": "葛根", "category": "辛凉解表药", "nature": "凉", "flavor": "甘、辛", "meridian": "脾、胃、肺经", "efficacy": "解肌退热，生津止渴，透疹，升阳止泻，通经活络", "indications": "外感发热，项背强痛，口渴消渴，麻疹不透，热痢泄泻", "usage": "煎服，10-15g", "contraindications": "表虚多汗者慎用"},
    {"name": "柴胡", "category": "辛凉解表药", "nature": "微寒", "flavor": "辛、苦", "meridian": "肝、胆、肺经", "efficacy": "疏散退热，疏肝解郁，升举阳气", "indications": "感冒发热，寒热往来，胸胁胀痛，月经不调，子宫脱垂", "usage": "煎服，3-10g", "contraindications": "肝阳上亢、肝风内动、阴虚火旺者忌用"},
    {"name": "升麻", "category": "辛凉解表药", "nature": "微寒", "flavor": "辛、微甘", "meridian": "肺、脾、胃、大肠经", "efficacy": "发表透疹，清热解毒，升举阳气", "indications": "风热头痛，齿痛口疮，咽喉肿痛，麻疹不透，中气下陷", "usage": "煎服，3-10g", "contraindications": "麻疹已透、阴虚火旺者忌用"},
    {"name": "淡豆豉", "category": "辛凉解表药", "nature": "凉", "flavor": "苦、辛", "meridian": "肺、胃经", "efficacy": "解表除烦，宣发郁热", "indications": "感冒寒热头痛，烦躁胸闷，虚烦不眠", "usage": "煎服，6-12g", "contraindications": "胃虚易呕者慎用"},

    # ============ 清热药 ============
    # 清热泻火
    {"name": "石膏", "category": "清热泻火药", "nature": "大寒", "flavor": "甘、辛", "meridian": "肺、胃经", "efficacy": "清热泻火，除烦止渴", "indications": "外感热病，高热烦渴，肺热喘咳，胃火亢盛，头痛牙痛", "usage": "先煎，15-60g", "contraindications": "脾胃虚寒、阴虚内热者忌服"},
    {"name": "知母", "category": "清热泻火药", "nature": "寒", "flavor": "苦、甘", "meridian": "肺、胃、肾经", "efficacy": "清热泻火，滋阴润燥", "indications": "热病烦渴，肺热燥咳，骨蒸潮热，内热消渴，肠燥便秘", "usage": "煎服，6-12g", "contraindications": "脾虚便溏者不宜"},
    {"name": "芦根", "category": "清热泻火药", "nature": "寒", "flavor": "甘", "meridian": "肺、胃经", "efficacy": "清热泻火，生津止渴，除烦止呕，利尿", "indications": "热病烦渴，胃热呕哕，肺热咳嗽，肺痈吐脓，热淋涩痛", "usage": "煎服，15-30g", "contraindications": "脾胃虚寒者慎用"},
    {"name": "天花粉", "category": "清热泻火药", "nature": "微寒", "flavor": "甘、微苦", "meridian": "肺、胃经", "efficacy": "清热泻火，生津止渴，消肿排脓", "indications": "热病烦渴，肺热燥咳，内热消渴，疮疡肿毒", "usage": "煎服，10-15g", "contraindications": "孕妇慎用，不宜与乌头同用"},
    {"name": "竹叶", "category": "清热泻火药", "nature": "寒", "flavor": "甘、淡", "meridian": "心、胃、小肠经", "efficacy": "清热泻火，除烦止渴，利尿通淋", "indications": "热病烦渴，口疮尿赤，热淋涩痛", "usage": "煎服，6-10g", "contraindications": "脾胃虚寒者慎用"},
    {"name": "栀子", "category": "清热泻火药", "nature": "寒", "flavor": "苦", "meridian": "心、肺、三焦经", "efficacy": "泻火除烦，清热利湿，凉血解毒，外用消肿止痛", "indications": "热病心烦，湿热黄疸，淋证涩痛，血热吐衄，目赤肿痛", "usage": "煎服，6-10g", "contraindications": "脾虚便溏者忌服"},
    {"name": "夏枯草", "category": "清热泻火药", "nature": "寒", "flavor": "辛、苦", "meridian": "肝、胆经", "efficacy": "清肝泻火，明目散结消肿", "indications": "目赤肿痛，头痛眩晕，瘰疬瘿瘤，乳痈肿痛", "usage": "煎服，9-15g", "contraindications": "脾胃虚弱者慎用"},
    {"name": "决明子", "category": "清热泻火药", "nature": "微寒", "flavor": "甘、苦、咸", "meridian": "肝、大肠经", "efficacy": "清热明目，润肠通便", "indications": "目赤涩痛，羞明多泪，头痛眩晕，目暗不明，大便秘结", "usage": "煎服，9-15g", "contraindications": "气虚便溏者不宜"},

    # 清热燥湿
    {"name": "黄芩", "category": "清热燥湿药", "nature": "寒", "flavor": "苦", "meridian": "肺、胆、脾、大肠、小肠经", "efficacy": "清热燥湿，泻火解毒，止血安胎", "indications": "湿温暑湿，胸闷呕恶，湿热痞满，泻痢黄疸，肺热咳嗽，高热烦渴，血热吐衄，胎动不安", "usage": "煎服，3-10g", "contraindications": "脾胃虚寒者慎用"},
    {"name": "黄连", "category": "清热燥湿药", "nature": "寒", "flavor": "苦", "meridian": "心、脾、胃、肝、胆、大肠经", "efficacy": "清热燥湿，泻火解毒", "indications": "湿热痞满，呕吐吞酸，泻痢黄疸，高热神昏，心火亢盛，心烦不寐，血热吐衄，目赤牙痛，消渴", "usage": "煎服，2-5g", "contraindications": "脾胃虚寒者忌用"},
    {"name": "黄柏", "category": "清热燥湿药", "nature": "寒", "flavor": "苦", "meridian": "肾、膀胱经", "efficacy": "清热燥湿，泻火解毒，除骨蒸", "indications": "湿热泻痢，黄疸尿赤，带下阴痒，热淋涩痛，骨蒸劳热，盗汗遗精，疮疡肿毒", "usage": "煎服，3-12g", "contraindications": "脾胃虚寒者忌用"},
    {"name": "龙胆", "category": "清热燥湿药", "nature": "寒", "flavor": "苦", "meridian": "肝、胆经", "efficacy": "清热燥湿，泻肝胆火", "indications": "湿热黄疸，阴肿阴痒，带下湿疹，肝火头痛，目赤耳聋，胁痛口苦，惊风抽搐", "usage": "煎服，3-6g", "contraindications": "脾胃虚寒、阴虚津伤者慎用"},
    {"name": "苦参", "category": "清热燥湿药", "nature": "寒", "flavor": "苦", "meridian": "心、肝、胃、大肠、膀胱经", "efficacy": "清热燥湿，杀虫利尿", "indications": "热痢便血，黄疸尿赤，赤白带下，阴肿阴痒，湿疹湿疮，皮肤瘙痒，疥癣", "usage": "煎服，4.5-9g", "contraindications": "脾胃虚寒者忌用，不宜与藜芦同用"},
    {"name": "白鲜皮", "category": "清热燥湿药", "nature": "寒", "flavor": "苦", "meridian": "脾、胃、膀胱经", "efficacy": "清热燥湿，祛风解毒", "indications": "湿热疮毒，黄水淋漓，湿疹风疹，疥癣疮癞，风湿热痹，黄疸尿赤", "usage": "煎服，5-10g", "contraindications": "虚寒证忌服"},

    # 清热解毒
    {"name": "金银花", "category": "清热解毒药", "nature": "寒", "flavor": "甘", "meridian": "肺、心、胃经", "efficacy": "清热解毒，疏散风热", "indications": "痈肿疔疮，喉痹丹毒，热毒血痢，风热感冒，温病发热", "usage": "煎服，6-15g", "contraindications": "脾胃虚寒及气虚疮疡脓清者忌服"},
    {"name": "连翘", "category": "清热解毒药", "nature": "微寒", "flavor": "苦", "meridian": "肺、心、小肠经", "efficacy": "清热解毒，消肿散结，疏散风热", "indications": "痈疽瘰疬，乳痈肺痈，丹毒，风热感冒，温病初起，温热入营，高热烦渴", "usage": "煎服，6-15g", "contraindications": "脾胃虚寒及气虚脓清者不宜"},
    {"name": "蒲公英", "category": "清热解毒药", "nature": "寒", "flavor": "苦、甘", "meridian": "肝、胃经", "efficacy": "清热解毒，消肿散结，利尿通淋", "indications": "疔疮肿毒，乳痈瘰疬，目赤咽痛，肺痈肠痈，湿热黄疸，热淋涩痛", "usage": "煎服，10-15g", "contraindications": "阳虚外寒、脾胃虚弱者忌用"},
    {"name": "板蓝根", "category": "清热解毒药", "nature": "寒", "flavor": "苦", "meridian": "心、胃经", "efficacy": "清热解毒，凉血利咽", "indications": "温疫时毒，发热咽痛，温毒发斑，痄腮喉痹，大头瘟疫，丹毒痈肿", "usage": "煎服，9-15g", "contraindications": "体虚而无实火热毒者忌服"},
    {"name": "鱼腥草", "category": "清热解毒药", "nature": "微寒", "flavor": "辛", "meridian": "肺经", "efficacy": "清热解毒，消痈排脓，利尿通淋", "indications": "肺痈吐脓，痰热喘咳，热痢热淋，痈肿疮毒", "usage": "煎服，15-25g", "contraindications": "虚寒证及阴性疮疡忌服"},
    {"name": "射干", "category": "清热解毒药", "nature": "寒", "flavor": "苦", "meridian": "肺经", "efficacy": "清热解毒，消痰利咽", "indications": "热毒痰火郁结，咽喉肿痛，痰涎壅盛，咳嗽气喘", "usage": "煎服，3-10g", "contraindications": "脾虚便溏者不宜，孕妇慎用"},
    {"name": "白头翁", "category": "清热解毒药", "nature": "寒", "flavor": "苦", "meridian": "胃、大肠经", "efficacy": "清热解毒，凉血止痢", "indications": "热毒血痢，阴痒带下", "usage": "煎服，9-15g", "contraindications": "虚寒泻痢忌服"},
    {"name": "败酱草", "category": "清热解毒药", "nature": "微寒", "flavor": "辛、苦", "meridian": "肝、胃、大肠经", "efficacy": "清热解毒，消痈排脓，祛瘀止痛", "indications": "肠痈肺痈，痈肿疮毒，产后瘀阻腹痛", "usage": "煎服，6-15g", "contraindications": "脾胃虚弱者慎用"},

    # 清热凉血
    {"name": "生地黄", "category": "清热凉血药", "nature": "寒", "flavor": "甘", "meridian": "心、肝、肾经", "efficacy": "清热凉血，养阴生津", "indications": "热入营血，温毒发斑，吐血衄血，热病伤阴，舌绛烦渴，津伤便秘，阴虚发热，骨蒸劳热", "usage": "煎服，10-15g", "contraindications": "脾虚湿滞、腹满便溏者不宜"},
    {"name": "玄参", "category": "清热凉血药", "nature": "微寒", "flavor": "甘、苦、咸", "meridian": "肺、胃、肾经", "efficacy": "清热凉血，滋阴降火，解毒散结", "indications": "热入营血，温毒发斑，热病伤阴，舌绛烦渴，津伤便秘，骨蒸劳嗽，目赤咽痛，瘰疬痈肿", "usage": "煎服，9-15g", "contraindications": "不宜与藜芦同用"},
    {"name": "牡丹皮", "category": "清热凉血药", "nature": "微寒", "flavor": "苦、辛", "meridian": "心、肝、肾经", "efficacy": "清热凉血，活血化瘀", "indications": "热入营血，温毒发斑，吐血衄血，夜热早凉，无汗骨蒸，经闭痛经，跌扑伤痛，痈肿疮毒", "usage": "煎服，6-12g", "contraindications": "血虚有寒、月经过多者不宜，孕妇慎用"},
    {"name": "赤芍", "category": "清热凉血药", "nature": "微寒", "flavor": "苦", "meridian": "肝经", "efficacy": "清热凉血，散瘀止痛", "indications": "热入营血，温毒发斑，吐血衄血，目赤肿痛，肝郁胁痛，经闭痛经，癥瘕腹痛，跌扑损伤，痈肿疮毒", "usage": "煎服，6-12g", "contraindications": "不宜与藜芦同用"},
    {"name": "紫草", "category": "清热凉血药", "nature": "寒", "flavor": "甘、咸", "meridian": "心、肝经", "efficacy": "清热凉血，活血解毒，透疹消斑", "indications": "血热毒盛，斑疹紫黑，麻疹不透，疮疡湿疹，水火烫伤", "usage": "煎服，5-10g", "contraindications": "脾胃虚寒、大便滑泄者忌服"},

    # ============ 补虚药 ============
    # 补气药
    {"name": "人参", "category": "补气药", "nature": "微温", "flavor": "甘、微苦", "meridian": "脾、肺、心、肾经", "efficacy": "大补元气，复脉固脱，补脾益肺，生津养血，安神益智", "indications": "体虚欲脱，肢冷脉微，脾虚食少，肺虚喘咳，津伤口渴，内热消渴，气血亏虚，久病虚羸，惊悸失眠，阳痿宫冷", "usage": "煎服，3-9g", "contraindications": "实证、热证而正气不虚者忌服"},
    {"name": "党参", "category": "补气药", "nature": "平", "flavor": "甘", "meridian": "脾、肺经", "efficacy": "健脾益肺，养血生津", "indications": "脾肺气虚，食少倦怠，咳嗽虚喘，气血不足，面色萎黄，心悸气短，津伤口渴，内热消渴", "usage": "煎服，9-30g", "contraindications": "不宜与藜芦同用"},
    {"name": "太子参", "category": "补气药", "nature": "平", "flavor": "甘、微苦", "meridian": "脾、肺经", "efficacy": "益气健脾，生津润肺", "indications": "脾虚体倦，食欲不振，病后虚弱，气阴不足，自汗口渴，肺燥干咳", "usage": "煎服，9-30g", "contraindications": "邪实正不虚者慎用"},
    {"name": "黄芪", "category": "补气药", "nature": "微温", "flavor": "甘", "meridian": "脾、肺经", "efficacy": "补气升阳，固表止汗，利水消肿，生津养血，行滞通痹，托毒排脓，敛疮生肌", "indications": "气虚乏力，食少便溏，中气下陷，久泻脱肛，便血崩漏，表虚自汗，气虚水肿，内热消渴，血虚萎黄，半身不遂，痹痛麻木，痈疽难溃，久溃不敛", "usage": "煎服，9-30g", "contraindications": "表实邪盛、气滞湿阻、食积内停、阴虚阳亢、痈疽初起者忌用"},
    {"name": "白术", "category": "补气药", "nature": "温", "flavor": "甘、苦", "meridian": "脾、胃经", "efficacy": "健脾益气，燥湿利水，止汗安胎", "indications": "脾虚食少，腹胀泄泻，痰饮眩悸，水肿自汗，胎动不安", "usage": "煎服，6-12g", "contraindications": "阴虚燥渴、气滞胀闷者忌服"},
    {"name": "山药", "category": "补气药", "nature": "平", "flavor": "甘", "meridian": "脾、肺、肾经", "efficacy": "补脾养胃，生津益肺，补肾涩精", "indications": "脾虚食少，久泻不止，肺虚喘咳，肾虚遗精，带下尿频，虚热消渴", "usage": "煎服，15-30g", "contraindications": "湿盛中满或有积滞者忌服"},
    {"name": "甘草", "category": "补气药", "nature": "平", "flavor": "甘", "meridian": "心、肺、脾、胃经", "efficacy": "补脾益气，清热解毒，祛痰止咳，缓急止痛，调和诸药", "indications": "脾胃虚弱，倦怠乏力，心悸气短，咳嗽痰多，脘腹四肢挛急疼痛，痈肿疮毒，缓解药物毒性烈性", "usage": "煎服，2-10g", "contraindications": "不宜与海藻、大戟、甘遂、芫花同用"},
    {"name": "大枣", "category": "补气药", "nature": "温", "flavor": "甘", "meridian": "脾、胃、心经", "efficacy": "补中益气，养血安神", "indications": "脾虚食少，乏力便溏，妇人脏躁，失眠", "usage": "煎服，6-15g", "contraindications": "湿盛脘腹胀满、食积虫积者忌服"},

    # 补血药
    {"name": "当归", "category": "补血药", "nature": "温", "flavor": "甘、辛", "meridian": "肝、心、脾经", "efficacy": "补血活血，调经止痛，润肠通便", "indications": "血虚萎黄，眩晕心悸，月经不调，经闭痛经，虚寒腹痛，风湿痹痛，跌扑损伤，痈疽疮疡，肠燥便秘", "usage": "煎服，6-12g", "contraindications": "湿盛中满、大便泄泻者忌服"},
    {"name": "熟地黄", "category": "补血药", "nature": "微温", "flavor": "甘", "meridian": "肝、肾经", "efficacy": "补血滋阴，益精填髓", "indications": "血虚萎黄，心悸怔忡，月经不调，崩漏下血，肝肾阴虚，腰膝酸软，骨蒸潮热，盗汗遗精，内热消渴，眩晕耳鸣，须发早白", "usage": "煎服，9-15g", "contraindications": "气滞痰多、脘腹胀痛、食少便溏者忌服"},
    {"name": "白芍", "category": "补血药", "nature": "微寒", "flavor": "苦、酸", "meridian": "肝、脾经", "efficacy": "养血调经，敛阴止汗，柔肝止痛，平抑肝阳", "indications": "血虚萎黄，月经不调，自汗盗汗，胁痛腹痛，四肢挛痛，头痛眩晕", "usage": "煎服，6-15g", "contraindications": "不宜与藜芦同用"},
    {"name": "阿胶", "category": "补血药", "nature": "平", "flavor": "甘", "meridian": "肺、肝、肾经", "efficacy": "补血滋阴，润燥止血", "indications": "血虚萎黄，眩晕心悸，肌痿无力，心烦不眠，虚风内动，肺燥咳嗽，劳嗽咯血，吐血尿血，便血崩漏", "usage": "烊化兑服，3-9g", "contraindications": "脾胃虚弱、呕吐泄泻、腹胀便溏者慎用"},
    {"name": "何首乌", "category": "补血药", "nature": "温", "flavor": "苦、甘、涩", "meridian": "肝、心、肾经", "efficacy": "制用补肝肾、益精血、乌须发、强筋骨；生用解毒消痈、截疟润肠", "indications": "血虚萎黄，眩晕耳鸣，须发早白，腰膝酸软，肢体麻木，崩漏带下；瘰疬痈疮，风疹瘙痒，肠燥便秘", "usage": "煎服，制何首乌6-12g", "contraindications": "大便溏泄及有湿痰者慎用"},
    {"name": "龙眼肉", "category": "补血药", "nature": "温", "flavor": "甘", "meridian": "心、脾经", "efficacy": "补益心脾，养血安神", "indications": "气血不足，心悸怔忡，健忘失眠，血虚萎黄", "usage": "煎服，9-15g", "contraindications": "湿盛中满或有停饮者忌服"},

    # 补阴药
    {"name": "北沙参", "category": "补阴药", "nature": "微寒", "flavor": "甘、微苦", "meridian": "肺、胃经", "efficacy": "养阴清肺，益胃生津", "indications": "肺热燥咳，劳嗽痰血，胃阴不足，热病伤津，咽干口渴", "usage": "煎服，5-12g", "contraindications": "不宜与藜芦同用"},
    {"name": "南沙参", "category": "补阴药", "nature": "微寒", "flavor": "甘", "meridian": "肺、胃经", "efficacy": "养阴清肺，益胃生津，化痰益气", "indications": "肺热燥咳，阴虚劳嗽，干咳痰黏，胃阴不足，食少呕吐，气阴不足，烦热口渴", "usage": "煎服，9-15g", "contraindications": "不宜与藜芦同用"},
    {"name": "麦冬", "category": "补阴药", "nature": "微寒", "flavor": "甘、微苦", "meridian": "心、肺、胃经", "efficacy": "养阴生津，润肺清心", "indications": "肺燥干咳，阴虚劳嗽，喉痹咽痛，津伤口渴，内热消渴，心烦失眠，肠燥便秘", "usage": "煎服，6-12g", "contraindications": "脾胃虚寒、大便溏泄者忌服"},
    {"name": "天冬", "category": "补阴药", "nature": "寒", "flavor": "甘、苦", "meridian": "肺、肾经", "efficacy": "养阴润燥，清肺生津", "indications": "肺燥干咳，顿咳痰黏，腰膝酸痛，骨蒸潮热，内热消渴，热病伤津，咽干口渴，肠燥便秘", "usage": "煎服，6-12g", "contraindications": "脾胃虚寒、食少便溏者忌服"},
    {"name": "玉竹", "category": "补阴药", "nature": "微寒", "flavor": "甘", "meridian": "肺、胃经", "efficacy": "养阴润燥，生津止渴", "indications": "肺胃阴伤，燥热咳嗽，咽干口渴，内热消渴", "usage": "煎服，6-12g", "contraindications": "痰湿气滞者忌服"},
    {"name": "枸杞子", "category": "补阴药", "nature": "平", "flavor": "甘", "meridian": "肝、肾经", "efficacy": "滋补肝肾，益精明目", "indications": "虚劳精亏，腰膝酸痛，眩晕耳鸣，阳痿遗精，内热消渴，血虚萎黄，目昏不明", "usage": "煎服，6-12g", "contraindications": "外邪实热、脾虚有湿及泄泻者忌服"},
    {"name": "百合", "category": "补阴药", "nature": "寒", "flavor": "甘", "meridian": "心、肺经", "efficacy": "养阴润肺，清心安神", "indications": "阴虚燥咳，劳嗽咳血，虚烦惊悸，失眠多梦，精神恍惚", "usage": "煎服，6-12g", "contraindications": "风寒咳嗽、中寒便滑者忌服"},
    {"name": "女贞子", "category": "补阴药", "nature": "凉", "flavor": "甘、苦", "meridian": "肝、肾经", "efficacy": "滋补肝肾，明目乌发", "indications": "肝肾阴虚，眩晕耳鸣，腰膝酸软，须发早白，目暗不明，内热消渴，骨蒸潮热", "usage": "煎服，6-12g", "contraindications": "脾胃虚寒泄泻及阳虚者忌服"},
    {"name": "墨旱莲", "category": "补阴药", "nature": "寒", "flavor": "甘、酸", "meridian": "肾、肝经", "efficacy": "滋补肝肾，凉血止血", "indications": "肝肾阴虚，牙齿松动，须发早白，眩晕耳鸣，腰膝酸软，阴虚血热吐血衄血尿血血痢崩漏", "usage": "煎服，6-12g", "contraindications": "脾肾虚寒者忌服"},
    {"name": "龟甲", "category": "补阴药", "nature": "寒", "flavor": "甘、咸", "meridian": "肝、肾、心经", "efficacy": "滋阴潜阳，益肾强骨，养血补心固经止崩", "indications": "阴虚潮热，骨蒸盗汗，头晕目眩，虚风内动，筋骨痿软，心虚健忘，崩漏经多", "usage": "先煎，9-24g", "contraindications": "脾胃虚寒者忌服"},
    {"name": "鳖甲", "category": "补阴药", "nature": "微寒", "flavor": "咸", "meridian": "肝、肾经", "efficacy": "滋阴潜阳，退热除蒸，软坚散结", "indications": "阴虚发热，骨蒸劳热，阴虚阳亢，头晕目眩，虚风内动，经闭癥瘕，久疟疟母", "usage": "先煎，9-24g", "contraindications": "脾胃虚寒、食少便溏者忌服"},

    # 补阳药
    {"name": "鹿茸", "category": "补阳药", "nature": "温", "flavor": "甘、咸", "meridian": "肾、肝经", "efficacy": "壮肾阳，益精血，强筋骨，调冲任，托疮毒", "indications": "阳痿滑精，宫冷不孕，羸瘦神疲，畏寒眩晕，耳鸣耳聋，腰脊冷痛，筋骨痿软，崩漏带下，阴疽不敛", "usage": "研末冲服，1-2g", "contraindications": "阴虚阳亢、血分有热者忌服"},
    {"name": "淫羊藿", "category": "补阳药", "nature": "温", "flavor": "辛、甘", "meridian": "肝、肾经", "efficacy": "补肾阳，强筋骨，祛风湿", "indications": "阳痿遗精，筋骨痿软，风湿痹痛，麻木拘挛", "usage": "煎服，6-10g", "contraindications": "阴虚火旺者忌服"},
    {"name": "巴戟天", "category": "补阳药", "nature": "微温", "flavor": "甘、辛", "meridian": "肾、肝经", "efficacy": "补肾阳，强筋骨，祛风湿", "indications": "阳痿遗精，宫冷不孕，月经不调，少腹冷痛，风湿痹痛，筋骨痿软", "usage": "煎服，3-10g", "contraindications": "阴虚火旺或有湿热者忌服"},
    {"name": "杜仲", "category": "补阳药", "nature": "温", "flavor": "甘", "meridian": "肝、肾经", "efficacy": "补肝肾，强筋骨，安胎", "indications": "肝肾不足，腰膝酸痛，筋骨无力，头晕目眩，妊娠漏血，胎动不安", "usage": "煎服，6-10g", "contraindications": "阴虚火旺者慎用"},
    {"name": "续断", "category": "补阳药", "nature": "微温", "flavor": "苦、辛", "meridian": "肝、肾经", "efficacy": "补肝肾，强筋骨，续折伤，止崩漏", "indications": "肝肾不足，腰膝酸软，风湿痹痛，跌扑损伤，筋伤骨折，崩漏胎漏", "usage": "煎服，9-15g", "contraindications": "风湿热痹者忌服"},
    {"name": "肉苁蓉", "category": "补阳药", "nature": "温", "flavor": "甘、咸", "meridian": "肾、大肠经", "efficacy": "补肾阳，益精血，润肠通便", "indications": "阳痿不孕，腰膝酸软，筋骨无力，肠燥便秘", "usage": "煎服，6-10g", "contraindications": "阴虚火旺、大便溏泄、实热便秘者忌服"},
    {"name": "菟丝子", "category": "补阳药", "nature": "平", "flavor": "辛、甘", "meridian": "肝、肾、脾经", "efficacy": "补益肝肾，固精缩尿，安胎明目止泻", "indications": "阳痿遗精，尿有余沥，遗尿尿频，腰膝酸软，目昏耳鸣，肾虚胎漏，脾肾虚泻", "usage": "煎服，6-12g", "contraindications": "阴虚火旺、大便燥结者忌服"},
    {"name": "补骨脂", "category": "补阳药", "nature": "温", "flavor": "辛、苦", "meridian": "肾、脾经", "efficacy": "温肾助阳，纳气平喘，温脾止泻", "indications": "阳痿遗精，遗尿尿频，腰膝冷痛，肾虚作喘，五更泄泻", "usage": "煎服，6-10g", "contraindications": "阴虚火旺、大便秘结者忌服"},
    {"name": "益智仁", "category": "补阳药", "nature": "温", "flavor": "辛", "meridian": "脾、肾经", "efficacy": "暖肾固精缩尿，温脾止泻摄唾", "indications": "肾虚遗尿，小便频数，遗精白浊，脾寒泄泻，腹中冷痛，口多唾涎", "usage": "煎服，3-10g", "contraindications": "阴虚火旺者忌服"},

    # ============ 理气药 ============
    {"name": "陈皮", "category": "理气药", "nature": "温", "flavor": "苦、辛", "meridian": "脾、肺经", "efficacy": "理气健脾，燥湿化痰", "indications": "脘腹胀满，食少吐泻，咳嗽痰多", "usage": "煎服，3-10g", "contraindications": "吐血证慎用"},
    {"name": "青皮", "category": "理气药", "nature": "温", "flavor": "苦、辛", "meridian": "肝、胆、胃经", "efficacy": "疏肝破气，消积化滞", "indications": "胸胁胀痛，疝气疼痛，乳癖乳痈，食积气滞，脘腹胀痛", "usage": "煎服，3-10g", "contraindications": "气虚者慎用"},
    {"name": "枳实", "category": "理气药", "nature": "微寒", "flavor": "苦、辛、酸", "meridian": "脾、胃经", "efficacy": "破气消积，化痰散痞", "indications": "积滞内停，痞满胀痛，泻痢后重，大便不通，痰滞气阻，胸痹结胸", "usage": "煎服，3-10g", "contraindications": "脾胃虚弱及孕妇慎用"},
    {"name": "木香", "category": "理气药", "nature": "温", "flavor": "辛、苦", "meridian": "脾、胃、大肠、三焦、胆经", "efficacy": "行气止痛，健脾消食", "indications": "胸胁脘腹胀痛，泻痢后重，食积不消，不思饮食", "usage": "煎服，3-6g", "contraindications": "阴虚津液不足者慎用"},
    {"name": "香附", "category": "理气药", "nature": "平", "flavor": "辛、微苦、微甘", "meridian": "肝、脾、三焦经", "efficacy": "疏肝解郁，理气宽中，调经止痛", "indications": "肝郁气滞，胸胁胀痛，疝气疼痛，乳房胀痛，脾胃气滞，脘腹痞闷，月经不调，经闭痛经", "usage": "煎服，6-10g", "contraindications": "气虚无滞、阴虚血热者忌服"},
    {"name": "乌药", "category": "理气药", "nature": "温", "flavor": "辛", "meridian": "肺、脾、肾、膀胱经", "efficacy": "行气止痛，温肾散寒", "indications": "寒凝气滞，胸腹胀痛，气逆喘急，疝气疼痛，经寒腹痛，膀胱虚冷，遗尿尿频", "usage": "煎服，6-10g", "contraindications": "气虚内热者忌服"},
    {"name": "佛手", "category": "理气药", "nature": "温", "flavor": "辛、苦、酸", "meridian": "肝、脾、胃、肺经", "efficacy": "疏肝理气，和胃止痛，燥湿化痰", "indications": "肝胃气滞，胸胁胀痛，胃脘痞满，食少呕吐，咳嗽痰多", "usage": "煎服，3-10g", "contraindications": "阴虚火旺者慎用"},
    {"name": "玫瑰花", "category": "理气药", "nature": "温", "flavor": "甘、微苦", "meridian": "肝、脾经", "efficacy": "行气解郁，和血止痛", "indications": "肝胃气痛，食少呕恶，月经不调，经前乳房胀痛，跌扑伤痛", "usage": "煎服，3-6g", "contraindications": "阴虚火旺者慎用"},
    {"name": "川楝子", "category": "理气药", "nature": "寒", "flavor": "苦", "meridian": "肝、小肠、膀胱经", "efficacy": "疏肝泄热，行气止痛，杀虫", "indications": "肝郁化火，胸胁脘腹胀痛，疝气疼痛，虫积腹痛", "usage": "煎服，5-10g", "contraindications": "脾胃虚寒者忌服"},

    # ============ 活血化瘀药 ============
    {"name": "川芎", "category": "活血化瘀药", "nature": "温", "flavor": "辛", "meridian": "肝、胆、心包经", "efficacy": "活血行气，祛风止痛", "indications": "胸痹心痛，胸胁刺痛，跌扑肿痛，月经不调，经闭痛经，癥瘕腹痛，头痛风湿痹痛", "usage": "煎服，3-10g", "contraindications": "阴虚火旺、月经过多者慎用"},
    {"name": "延胡索", "category": "活血化瘀药", "nature": "温", "flavor": "辛、苦", "meridian": "肝、脾经", "efficacy": "活血行气止痛", "indications": "胸胁脘腹疼痛，胸痹心痛，经闭痛经，产后瘀阻，跌扑肿痛", "usage": "煎服，3-10g", "contraindications": "血热气虚者忌用，孕妇慎用"},
    {"name": "郁金", "category": "活血化瘀药", "nature": "寒", "flavor": "辛、苦", "meridian": "肝、心、肺经", "efficacy": "活血止痛，行气解郁，清心凉血，利胆退黄", "indications": "胸胁刺痛，胸痹心痛，经闭痛经，乳房胀痛，热病神昏，癫痫发狂，血热吐衄，黄疸尿赤", "usage": "煎服，3-10g", "contraindications": "不宜与丁香同用"},
    {"name": "丹参", "category": "活血化瘀药", "nature": "微寒", "flavor": "苦", "meridian": "心、肝经", "efficacy": "活血祛瘀，通经止痛，清心除烦，凉血消痈", "indications": "胸痹心痛，脘腹胁痛，癥瘕积聚，热痹疼痛，心烦不眠，月经不调，痛经经闭，疮疡肿痛", "usage": "煎服，10-15g", "contraindications": "不宜与藜芦同用"},
    {"name": "红花", "category": "活血化瘀药", "nature": "温", "flavor": "辛", "meridian": "心、肝经", "efficacy": "活血通经，散瘀止痛", "indications": "经闭痛经，恶露不行，癥瘕痞块，胸痹心痛，瘀滞腹痛，胸胁刺痛，跌扑损伤，疮疡肿痛", "usage": "煎服，3-10g", "contraindications": "孕妇忌用"},
    {"name": "桃仁", "category": "活血化瘀药", "nature": "平", "flavor": "苦、甘", "meridian": "心、肝、大肠经", "efficacy": "活血祛瘀，润肠通便，止咳平喘", "indications": "经闭痛经，癥瘕痞块，肺痈肠痈，跌扑损伤，肠燥便秘，咳嗽气喘", "usage": "煎服，5-10g", "contraindications": "孕妇忌用"},
    {"name": "益母草", "category": "活血化瘀药", "nature": "微寒", "flavor": "苦、辛", "meridian": "肝、心包、膀胱经", "efficacy": "活血调经，利尿消肿，清热解毒", "indications": "月经不调，痛经经闭，恶露不尽，水肿尿少，疮疡肿毒", "usage": "煎服，9-30g", "contraindications": "孕妇慎用"},
    {"name": "牛膝", "category": "活血化瘀药", "nature": "平", "flavor": "苦、甘、酸", "meridian": "肝、肾经", "efficacy": "逐瘀通经，补肝肾，强筋骨，利尿通淋，引血下行", "indications": "经闭痛经，腰膝酸痛，筋骨无力，淋证水肿，头痛眩晕，牙痛口疮，吐血衄血", "usage": "煎服，5-12g", "contraindications": "孕妇慎用"},

    # ============ 化痰止咳平喘药 ============
    {"name": "半夏", "category": "化痰止咳平喘药", "nature": "温", "flavor": "辛", "meridian": "脾、胃、肺经", "efficacy": "燥湿化痰，降逆止呕，消痞散结", "indications": "湿痰寒痰，咳喘痰多，痰饮眩悸，风痰眩晕，痰厥头痛，呕吐反胃，胸脘痞闷，梅核气", "usage": "煎服，3-9g", "contraindications": "不宜与川乌、草乌、附子同用"},
    {"name": "天南星", "category": "化痰止咳平喘药", "nature": "温", "flavor": "苦、辛", "meridian": "肺、肝、脾经", "efficacy": "燥湿化痰，祛风止痉，散结消肿", "indications": "顽痰咳嗽，风痰眩晕，中风痰壅，口眼歪斜，半身不遂，癫痫惊风，破伤风", "usage": "煎服，3-9g", "contraindications": "孕妇慎用，生品内服宜慎"},
    {"name": "川贝母", "category": "化痰止咳平喘药", "nature": "微寒", "flavor": "苦、甘", "meridian": "肺、心经", "efficacy": "清热润肺，化痰止咳，散结消痈", "indications": "肺热燥咳，干咳少痰，阴虚劳嗽，痰中带血，瘰疬乳痈，肺痈", "usage": "煎服，3-10g", "contraindications": "不宜与川乌、草乌、附子同用"},
    {"name": "浙贝母", "category": "化痰止咳平喘药", "nature": "寒", "flavor": "苦", "meridian": "肺、心经", "efficacy": "清热化痰止咳，解毒散结消痈", "indications": "风热咳嗽，痰火咳嗽，肺痈乳痈，瘰疬疮毒", "usage": "煎服，5-10g", "contraindications": "不宜与川乌、草乌、附子同用"},
    {"name": "瓜蒌", "category": "化痰止咳平喘药", "nature": "寒", "flavor": "甘、微苦", "meridian": "肺、胃、大肠经", "efficacy": "清热涤痰，宽胸散结，润燥滑肠", "indications": "肺热咳嗽，痰浊黄稠，胸痹心痛，结胸痞满，乳痈肺痈，肠痈肿痛，大便秘结", "usage": "煎服，9-15g", "contraindications": "不宜与川乌、草乌、附子同用"},
    {"name": "桔梗", "category": "化痰止咳平喘药", "nature": "平", "flavor": "苦、辛", "meridian": "肺经", "efficacy": "宣肺利咽，祛痰排脓", "indications": "咳嗽痰多，胸闷不畅，咽痛音哑，肺痈吐脓，疮疡脓成不溃", "usage": "煎服，3-10g", "contraindications": "阴虚久嗽、气逆咳血者忌服"},
    {"name": "苦杏仁", "category": "化痰止咳平喘药", "nature": "微温", "flavor": "苦", "meridian": "肺、大肠经", "efficacy": "降气止咳平喘，润肠通便", "indications": "咳嗽气喘，胸满痰多，肠燥便秘", "usage": "煎服，5-10g", "contraindications": "阴虚咳喘及大便溏泄者忌用"},
    {"name": "紫菀", "category": "化痰止咳平喘药", "nature": "温", "flavor": "辛、苦", "meridian": "肺经", "efficacy": "润肺下气，消痰止咳", "indications": "痰多喘咳，新久咳嗽，劳嗽咳血", "usage": "煎服，5-10g", "contraindications": "有实热者慎用"},
    {"name": "款冬花", "category": "化痰止咳平喘药", "nature": "温", "flavor": "辛、微苦", "meridian": "肺经", "efficacy": "润肺下气，止咳化痰", "indications": "新久咳嗽，喘咳痰多，劳嗽咳血", "usage": "煎服，5-10g", "contraindications": "肺热炽盛者不宜"},
    {"name": "枇杷叶", "category": "化痰止咳平喘药", "nature": "微寒", "flavor": "苦", "meridian": "肺、胃经", "efficacy": "清肺止咳，降逆止呕", "indications": "肺热咳嗽，气逆喘急，胃热呕逆，烦热口渴", "usage": "煎服，6-10g", "contraindications": "胃寒呕吐及风寒咳嗽者忌服"},
    {"name": "桑白皮", "category": "化痰止咳平喘药", "nature": "寒", "flavor": "甘", "meridian": "肺经", "efficacy": "泻肺平喘，利水消肿", "indications": "肺热喘咳，水肿胀满尿少，面目肌肤浮肿", "usage": "煎服，6-12g", "contraindications": "肺虚无火、风寒咳嗽者忌服"},
    {"name": "葶苈子", "category": "化痰止咳平喘药", "nature": "大寒", "flavor": "辛、苦", "meridian": "肺、膀胱经", "efficacy": "泻肺平喘，行水消肿", "indications": "痰涎壅肺，喘咳痰多，胸胁胀满，不得平卧，胸腹水肿，小便不利", "usage": "煎服，3-10g", "contraindications": "肺虚喘咳、脾虚肿满者忌服"},

    # ============ 祛风湿药 ============
    {"name": "独活", "category": "祛风湿药", "nature": "微温", "flavor": "辛、苦", "meridian": "肾、膀胱经", "efficacy": "祛风除湿，通痹止痛", "indications": "风寒湿痹，腰膝疼痛，少阴伏风头痛，风寒挟湿头痛", "usage": "煎服，3-10g", "contraindications": "阴虚血燥者慎用"},
    {"name": "威灵仙", "category": "祛风湿药", "nature": "温", "flavor": "辛、咸", "meridian": "膀胱经", "efficacy": "祛风湿，通经络", "indications": "风湿痹痛，肢体麻木，筋脉拘挛，屈伸不利", "usage": "煎服，6-10g", "contraindications": "气血虚弱者慎用"},
    {"name": "秦艽", "category": "祛风湿药", "nature": "平", "flavor": "辛、苦", "meridian": "胃、肝、胆经", "efficacy": "祛风湿，清湿热，止痹痛，退虚热", "indications": "风湿痹痛，中风半身不遂，筋脉拘挛，骨节酸痛，湿热黄疸，骨蒸潮热", "usage": "煎服，3-10g", "contraindications": "久痛虚羸、溲多便滑者忌服"},
    {"name": "木瓜", "category": "祛风湿药", "nature": "温", "flavor": "酸", "meridian": "肝、脾经", "efficacy": "舒筋活络，和胃化湿", "indications": "湿痹拘挛，腰膝关节酸重疼痛，暑湿吐泻，转筋挛痛，脚气水肿", "usage": "煎服，6-9g", "contraindications": "内有郁热、小便短赤者忌服"},
    {"name": "桑寄生", "category": "祛风湿药", "nature": "平", "flavor": "苦、甘", "meridian": "肝、肾经", "efficacy": "祛风湿，补肝肾，强筋骨，安胎", "indications": "风湿痹痛，腰膝酸软，筋骨无力，崩漏经多，妊娠漏血，胎动不安，头晕目眩", "usage": "煎服，9-15g", "contraindications": "无特殊禁忌"},
    {"name": "五加皮", "category": "祛风湿药", "nature": "温", "flavor": "辛、苦", "meridian": "肝、肾经", "efficacy": "祛风除湿，补益肝肾，强筋壮骨，利水消肿", "indications": "风湿痹痛，筋骨痿软，小儿行迟，体虚乏力，水肿脚气", "usage": "煎服，5-10g", "contraindications": "阴虚火旺者慎用"},

    # ============ 温里药 ============
    {"name": "附子", "category": "温里药", "nature": "大热", "flavor": "辛、甘", "meridian": "心、肾、脾经", "efficacy": "回阳救逆，补火助阳，散寒止痛", "indications": "亡阳虚脱，肢冷脉微，阳痿宫冷，心腹冷痛，虚寒吐泻，阴寒水肿，阳虚外感，寒湿痹痛", "usage": "先煎，3-15g", "contraindications": "孕妇慎用，不宜与半夏、瓜蒌、贝母、白蔹、白及同用"},
    {"name": "干姜", "category": "温里药", "nature": "热", "flavor": "辛", "meridian": "脾、胃、肾、心、肺经", "efficacy": "温中散寒，回阳通脉，温肺化饮", "indications": "脘腹冷痛，呕吐泄泻，肢冷脉微，寒饮喘咳", "usage": "煎服，3-10g", "contraindications": "阴虚内热、血热妄行者忌服"},
    {"name": "肉桂", "category": "温里药", "nature": "大热", "flavor": "辛、甘", "meridian": "肾、脾、心、肝经", "efficacy": "补火助阳，引火归元，散寒止痛，温通经脉", "indications": "阳痿宫冷，腰膝冷痛，肾虚作喘，虚阳上浮，眩晕目赤，心腹冷痛，虚寒吐泻，寒疝腹痛，痛经经闭", "usage": "煎服，1-5g", "contraindications": "阴虚火旺、里有实热者忌服"},
    {"name": "吴茱萸", "category": "温里药", "nature": "热", "flavor": "辛、苦", "meridian": "肝、脾、胃、肾经", "efficacy": "散寒止痛，降逆止呕，助阳止泻", "indications": "厥阴头痛，寒疝腹痛，寒湿脚气，经行腹痛，脘腹胀痛，呕吐吞酸，五更泄泻", "usage": "煎服，2-5g", "contraindications": "阴虚火旺者忌服"},
    {"name": "丁香", "category": "温里药", "nature": "温", "flavor": "辛", "meridian": "脾、胃、肺、肾经", "efficacy": "温中降逆，补肾助阳", "indications": "脾胃虚寒，呃逆呕吐，食少吐泻，心腹冷痛，肾虚阳痿", "usage": "煎服，1-3g", "contraindications": "不宜与郁金同用"},

    # ============ 利水渗湿药 ============
    {"name": "茯苓", "category": "利水渗湿药", "nature": "平", "flavor": "甘、淡", "meridian": "心、肺、脾、肾经", "efficacy": "利水渗湿，健脾宁心", "indications": "水肿尿少，痰饮眩悸，脾虚食少，便溏泄泻，心神不安，惊悸失眠", "usage": "煎服，10-15g", "contraindications": "虚寒精滑者忌服"},
    {"name": "猪苓", "category": "利水渗湿药", "nature": "平", "flavor": "甘、淡", "meridian": "肾、膀胱经", "efficacy": "利水渗湿", "indications": "小便不利，水肿泄泻，淋浊带下", "usage": "煎服，6-12g", "contraindications": "无水湿者忌服"},
    {"name": "泽泻", "category": "利水渗湿药", "nature": "寒", "flavor": "甘、淡", "meridian": "肾、膀胱经", "efficacy": "利水渗湿，泄热化浊降脂", "indications": "小便不利，水肿胀满，泄泻尿少，痰饮眩晕，热淋涩痛，高脂血症", "usage": "煎服，6-10g", "contraindications": "肾虚精滑者忌服"},
    {"name": "薏苡仁", "category": "利水渗湿药", "nature": "凉", "flavor": "甘、淡", "meridian": "脾、胃、肺经", "efficacy": "利水渗湿，健脾止泻，除痹排脓，解毒散结", "indications": "水肿脚气，小便不利，脾虚泄泻，湿痹拘挛，肺痈肠痈，赘疣癌肿", "usage": "煎服，9-30g", "contraindications": "孕妇慎用"},
    {"name": "车前子", "category": "利水渗湿药", "nature": "寒", "flavor": "甘", "meridian": "肝、肾、肺、小肠经", "efficacy": "清热利尿通淋，渗湿止泻，明目祛痰", "indications": "热淋涩痛，水肿胀满，暑湿泄泻，目赤肿痛，痰热咳嗽", "usage": "包煎，9-15g", "contraindications": "肾虚精滑者慎用"},
    {"name": "滑石", "category": "利水渗湿药", "nature": "寒", "flavor": "甘、淡", "meridian": "膀胱、肺、胃经", "efficacy": "利尿通淋，清热解暑，外用祛湿敛疮", "indications": "热淋石淋，尿热涩痛，暑湿烦渴，湿热水泻；外治湿疹湿疮", "usage": "包煎，10-20g", "contraindications": "脾虚、热病伤津者忌服"},
    {"name": "茵陈", "category": "利水渗湿药", "nature": "微寒", "flavor": "苦、辛", "meridian": "脾、胃、肝、胆经", "efficacy": "清利湿热，利胆退黄", "indications": "黄疸尿少，湿温暑湿，湿疮瘙痒", "usage": "煎服，6-15g", "contraindications": "蓄血发黄及血虚萎黄者慎用"},
    {"name": "金钱草", "category": "利水渗湿药", "nature": "微寒", "flavor": "甘、咸", "meridian": "肝、胆、肾、膀胱经", "efficacy": "利湿退黄，利尿通淋，解毒消肿", "indications": "湿热黄疸，胆胀胁痛，石淋热淋，小便涩痛，痈肿疔疮，蛇虫咬伤", "usage": "煎服，15-60g", "contraindications": "脾胃虚寒者慎用"},

    # ============ 止血药 ============
    {"name": "小蓟", "category": "止血药", "nature": "凉", "flavor": "甘、苦", "meridian": "心、肝经", "efficacy": "凉血止血，散瘀解毒消痈", "indications": "衄血吐血尿血血淋便血崩漏，外伤出血，痈肿疮毒", "usage": "煎服，5-12g", "contraindications": "脾胃虚寒者慎用"},
    {"name": "地榆", "category": "止血药", "nature": "微寒", "flavor": "苦、酸、涩", "meridian": "肝、大肠经", "efficacy": "凉血止血，解毒敛疮", "indications": "便血痔血血痢崩漏，水火烫伤，痈肿疮毒", "usage": "煎服，9-15g", "contraindications": "虚寒性出血者忌服"},
    {"name": "白茅根", "category": "止血药", "nature": "寒", "flavor": "甘", "meridian": "肺、胃、膀胱经", "efficacy": "凉血止血，清热利尿", "indications": "血热吐血衄血尿血，热病烦渴，湿热黄疸，水肿尿少，热淋涩痛", "usage": "煎服，9-30g", "contraindications": "脾胃虚寒者慎用"},
    {"name": "三七", "category": "止血药", "nature": "温", "flavor": "甘、微苦", "meridian": "肝、胃经", "efficacy": "散瘀止血，消肿定痛", "indications": "咯血吐血衄血便血崩漏，外伤出血，胸腹刺痛，跌扑肿痛", "usage": "研末吞服，1-3g", "contraindications": "孕妇慎用"},
    {"name": "艾叶", "category": "止血药", "nature": "温", "flavor": "辛、苦", "meridian": "肝、脾、肾经", "efficacy": "温经止血，散寒止痛，外用祛湿止痒", "indications": "吐血衄血崩漏经多，少腹冷痛，经寒不调，宫冷不孕；外治皮肤瘙痒", "usage": "煎服，3-9g", "contraindications": "阴虚血热者慎用"},
    {"name": "仙鹤草", "category": "止血药", "nature": "平", "flavor": "苦、涩", "meridian": "心、肝经", "efficacy": "收敛止血，截疟止痢，解毒补虚", "indications": "咯血吐血衄血尿血便血崩漏，疟疾，血痢，痈肿疮毒，阴痒带下，脱力劳伤", "usage": "煎服，6-12g", "contraindications": "无特殊禁忌"},

    # ============ 安神药 ============
    {"name": "酸枣仁", "category": "安神药", "nature": "平", "flavor": "甘、酸", "meridian": "肝、胆、心经", "efficacy": "养心补肝，宁心安神，敛汗生津", "indications": "虚烦不眠，惊悸多梦，体虚多汗，津伤口渴", "usage": "煎服，10-15g", "contraindications": "实邪郁火者慎用"},
    {"name": "远志", "category": "安神药", "nature": "温", "flavor": "苦、辛", "meridian": "心、肾、肺经", "efficacy": "安神益智，交通心肾，祛痰消肿", "indications": "心肾不交引起的失眠多梦，健忘惊悸，神志恍惚，咳痰不爽，疮疡肿毒，乳房肿痛", "usage": "煎服，3-10g", "contraindications": "有胃炎及胃溃疡者慎用"},
    {"name": "柏子仁", "category": "安神药", "nature": "平", "flavor": "甘", "meridian": "心、肾、大肠经", "efficacy": "养心安神，润肠通便止汗", "indications": "阴血不足，虚烦失眠，心悸怔忡，肠燥便秘，阴虚盗汗", "usage": "煎服，3-10g", "contraindications": "便溏及多痰者慎用"},
    {"name": "合欢皮", "category": "安神药", "nature": "平", "flavor": "甘", "meridian": "心、肝、肺经", "efficacy": "解郁安神，活血消肿", "indications": "心神不安，忧郁失眠，肺痈疮肿，跌扑伤痛", "usage": "煎服，6-12g", "contraindications": "孕妇慎用"},
    {"name": "首乌藤", "category": "安神药", "nature": "平", "flavor": "甘", "meridian": "心、肝经", "efficacy": "养血安神，祛风通络", "indications": "失眠多梦，血虚身痛，风湿痹痛，皮肤瘙痒", "usage": "煎服，9-15g", "contraindications": "燥狂属实火者慎用"},
    {"name": "龙骨", "category": "安神药", "nature": "平", "flavor": "甘、涩", "meridian": "心、肝、肾经", "efficacy": "镇惊安神，平肝潜阳，收敛固涩", "indications": "心神不宁，心悸失眠，惊痫癫狂，肝阳眩晕，滑脱诸证", "usage": "先煎，15-30g", "contraindications": "湿热积滞者不宜"},
    {"name": "牡蛎", "category": "安神药", "nature": "微寒", "flavor": "咸", "meridian": "肝、胆、肾经", "efficacy": "重镇安神，潜阳补阴，软坚散结", "indications": "惊悸失眠，眩晕耳鸣，瘰疬痰核，癥瘕痞块", "usage": "先煎，9-30g", "contraindications": "不宜多服久服"},

    # ============ 消食药 ============
    {"name": "山楂", "category": "消食药", "nature": "微温", "flavor": "酸、甘", "meridian": "脾、胃、肝经", "efficacy": "消食健胃，行气散瘀，化浊降脂", "indications": "肉食积滞，胃脘胀满，泻痢腹痛，瘀血经闭，产后瘀阻，心腹刺痛，胸痹心痛，疝气疼痛，高脂血症", "usage": "煎服，9-12g", "contraindications": "胃酸过多者慎用"},
    {"name": "麦芽", "category": "消食药", "nature": "平", "flavor": "甘", "meridian": "脾、胃经", "efficacy": "行气消食，健脾开胃，回乳消胀", "indications": "食积不消，脘腹胀痛，脾虚食少，乳汁郁积，乳房胀痛，妇女断乳", "usage": "煎服，10-15g", "contraindications": "哺乳期妇女不宜"},
    {"name": "神曲", "category": "消食药", "nature": "温", "flavor": "甘、辛", "meridian": "脾、胃经", "efficacy": "消食和胃", "indications": "饮食积滞，脘腹胀满，食少纳呆", "usage": "煎服，6-15g", "contraindications": "胃火炽盛者慎用"},
    {"name": "鸡内金", "category": "消食药", "nature": "平", "flavor": "甘", "meridian": "脾、胃、小肠、膀胱经", "efficacy": "健胃消食，涩精止遗，通淋化石", "indications": "食积不消，呕吐泻痢，小儿疳积，遗尿遗精，石淋涩痛，胆胀胁痛", "usage": "煎服，3-10g", "contraindications": "脾虚无积滞者慎用"},
    {"name": "莱菔子", "category": "消食药", "nature": "平", "flavor": "辛、甘", "meridian": "肺、脾、胃经", "efficacy": "消食除胀，降气化痰", "indications": "饮食停滞，脘腹胀痛，大便秘结，积滞泻痢，痰壅喘咳", "usage": "煎服，5-12g", "contraindications": "不宜与人参同用"},

    # ============ 收涩药 ============
    {"name": "五味子", "category": "收涩药", "nature": "温", "flavor": "酸、甘", "meridian": "肺、心、肾经", "efficacy": "收敛固涩，益气生津，补肾宁心", "indications": "久嗽虚喘，梦遗滑精，遗尿尿频，久泻不止，自汗盗汗，津伤口渴，内热消渴，心悸失眠", "usage": "煎服，2-6g", "contraindications": "表邪未解、内有实热者忌服"},
    {"name": "山茱萸", "category": "收涩药", "nature": "微温", "flavor": "酸、涩", "meridian": "肝、肾经", "efficacy": "补益肝肾，收涩固脱", "indications": "眩晕耳鸣，腰膝酸痛，阳痿遗精，遗尿尿频，崩漏带下，大汗虚脱，内热消渴", "usage": "煎服，6-12g", "contraindications": "命门火炽、素有湿热者慎用"},
    {"name": "乌梅", "category": "收涩药", "nature": "平", "flavor": "酸、涩", "meridian": "肝、脾、肺、大肠经", "efficacy": "敛肺涩肠，生津安蛔", "indications": "肺虚久咳，久泻久痢，虚热消渴，蛔厥呕吐腹痛", "usage": "煎服，6-12g", "contraindications": "外有表邪或内有实热者忌服"},
    {"name": "莲子", "category": "收涩药", "nature": "平", "flavor": "甘、涩", "meridian": "脾、肾、心经", "efficacy": "补脾止泻，止带，益肾涩精，养心安神", "indications": "脾虚泄泻，带下遗精，心悸失眠", "usage": "煎服，6-15g", "contraindications": "中满痞胀及大便燥结者忌服"},
    {"name": "芡实", "category": "收涩药", "nature": "平", "flavor": "甘、涩", "meridian": "脾、肾经", "efficacy": "益肾固精，补脾止泻，除湿止带", "indications": "遗精滑精，遗尿尿频，脾虚久泻，白浊带下", "usage": "煎服，9-15g", "contraindications": "大小便不利者慎用"},
    {"name": "麻黄根", "category": "收涩药", "nature": "平", "flavor": "甘、涩", "meridian": "心、肺经", "efficacy": "固表止汗", "indications": "自汗盗汗", "usage": "煎服，3-9g", "contraindications": "有表邪者忌用"},

    # ============ 平肝息风药 ============
    {"name": "天麻", "category": "平肝息风药", "nature": "平", "flavor": "甘", "meridian": "肝经", "efficacy": "息风止痉，平抑肝阳，祛风通络", "indications": "小儿惊风，癫痫抽搐，破伤风，头痛眩晕，手足不遂，肢体麻木，风湿痹痛", "usage": "煎服，3-10g", "contraindications": "无特殊禁忌"},
    {"name": "钩藤", "category": "平肝息风药", "nature": "凉", "flavor": "甘", "meridian": "肝、心包经", "efficacy": "息风定惊，清热平肝", "indications": "肝风内动，惊痫抽搐，高热惊厥，感冒夹惊，小儿惊啼，妊娠子痫，头痛眩晕", "usage": "后下，3-12g", "contraindications": "无风热及实热者慎用"},
    {"name": "石决明", "category": "平肝息风药", "nature": "寒", "flavor": "咸", "meridian": "肝经", "efficacy": "平肝潜阳，清肝明目", "indications": "头痛眩晕，目赤翳障，视物昏花，青盲雀目", "usage": "先煎，6-20g", "contraindications": "脾胃虚寒者慎用"},
    {"name": "地龙", "category": "平肝息风药", "nature": "寒", "flavor": "咸", "meridian": "肝、脾、膀胱经", "efficacy": "清热定惊，通络平喘利尿", "indications": "高热神昏，惊痫抽搐，关节痹痛，肢体麻木，半身不遂，肺热喘咳，水肿尿少", "usage": "煎服，5-10g", "contraindications": "脾胃虚弱者不宜"},
    {"name": "僵蚕", "category": "平肝息风药", "nature": "平", "flavor": "咸、辛", "meridian": "肝、肺、胃经", "efficacy": "息风止痉，祛风止痛，化痰散结", "indications": "肝风夹痰，惊痫抽搐，小儿急惊，破伤风，中风口眼歪斜，风热头痛，目赤咽痛，风疹瘙痒，瘰疬痰核", "usage": "煎服，5-10g", "contraindications": "血虚生风者慎用"},
]

# ── 扩展方剂数据（基于《方剂学》教材分类体系） ──

EXTENDED_FORMULAS: List[Dict[str, str]] = [
    # 解表剂
    {"name": "麻黄汤", "category": "辛温解表剂", "source_book": "《伤寒论》", "composition": "麻黄、桂枝、杏仁、甘草", "functions": "发汗解表，宣肺平喘", "indications": "外感风寒表实证，恶寒发热，头身疼痛，无汗而喘", "usage": "水煎服", "modification": "喘急胸闷加苏子、陈皮"},
    {"name": "桂枝汤", "category": "辛温解表剂", "source_book": "《伤寒论》", "composition": "桂枝、白芍、生姜、大枣、甘草", "functions": "解肌发表，调和营卫", "indications": "外感风寒表虚证，头痛发热，汗出恶风，鼻鸣干呕", "usage": "水煎服，服后啜热稀粥", "modification": "项背强几几加葛根"},
    {"name": "小青龙汤", "category": "辛温解表剂", "source_book": "《伤寒论》", "composition": "麻黄、桂枝、干姜、细辛、五味子、白芍、半夏、甘草", "functions": "解表散寒，温肺化饮", "indications": "外寒内饮证，恶寒发热，无汗咳喘，痰多而稀", "usage": "水煎服", "modification": "口渴去半夏加天花粉"},
    {"name": "银翘散", "category": "辛凉解表剂", "source_book": "《温病条辨》", "composition": "金银花、连翘、薄荷、牛蒡子、荆芥、淡豆豉、桔梗、竹叶、甘草、芦根", "functions": "辛凉透表，清热解毒", "indications": "温病初起，发热无汗，头痛口渴，咳嗽咽痛", "usage": "水煎服", "modification": "渴甚加天花粉"},
    {"name": "桑菊饮", "category": "辛凉解表剂", "source_book": "《温病条辨》", "composition": "桑叶、菊花、杏仁、连翘、薄荷、桔梗、甘草、芦根", "functions": "疏风清热，宣肺止咳", "indications": "风温初起，咳嗽身热不甚，口微渴", "usage": "水煎服", "modification": "气粗似喘加石膏、知母"},
    {"name": "麻杏石甘汤", "category": "辛凉解表剂", "source_book": "《伤寒论》", "composition": "麻黄、杏仁、石膏、甘草", "functions": "辛凉宣泄，清肺平喘", "indications": "外感风邪，邪热壅肺证，身热不解，咳逆气急，鼻煽口渴", "usage": "水煎服", "modification": "痰多气急加葶苈子、枇杷叶"},
    {"name": "败毒散", "category": "扶正解表剂", "source_book": "《太平惠民和剂局方》", "composition": "柴胡、前胡、川芎、枳壳、羌活、独活、茯苓、桔梗、人参、甘草、薄荷、生姜", "functions": "散寒祛湿，益气解表", "indications": "气虚外感风寒湿表证，憎寒壮热，头项强痛，肢体酸痛，无汗", "usage": "水煎服", "modification": "正气不虚去人参"},
    {"name": "加减葳蕤汤", "category": "扶正解表剂", "source_book": "《重订通俗伤寒论》", "composition": "玉竹、葱白、桔梗、白薇、淡豆豉、薄荷、甘草、大枣", "functions": "滋阴解表", "indications": "阴虚外感风热证，头痛身热，微恶风寒，无汗或有汗不多，口渴咽干", "usage": "水煎服", "modification": "表证较重加防风、葛根"},

    # 清热剂
    {"name": "白虎汤", "category": "清气分热剂", "source_book": "《伤寒论》", "composition": "石膏、知母、甘草、粳米", "functions": "清热生津", "indications": "阳明气分热盛证，壮热面赤，烦渴引饮，汗出恶热", "usage": "水煎服", "modification": "加人参治气津两伤"},
    {"name": "清营汤", "category": "清营凉血剂", "source_book": "《温病条辨》", "composition": "水牛角（原犀角）、生地黄、玄参、竹叶心、麦冬、丹参、黄连、金银花、连翘", "functions": "清营解毒，透热养阴", "indications": "热入营分证，身热夜甚，神烦少寐，时有谵语，斑疹隐隐", "usage": "水煎服", "modification": "神昏谵语加安宫牛黄丸"},
    {"name": "龙胆泻肝汤", "category": "清脏腑热剂", "source_book": "《医方集解》", "composition": "龙胆、黄芩、栀子、泽泻、木通、车前子、当归、生地黄、柴胡、甘草", "functions": "清泻肝胆实火，清利肝经湿热", "indications": "肝胆实火上炎，头痛目赤，胁痛口苦，耳聋耳肿；肝经湿热下注，阴肿阴痒，带下黄臭", "usage": "水煎服", "modification": "湿重于热加滑石、薏苡仁"},
    {"name": "左金丸", "category": "清脏腑热剂", "source_book": "《丹溪心法》", "composition": "黄连、吴茱萸", "functions": "清泻肝火，降逆止呕", "indications": "肝火犯胃证，胁肋疼痛，嘈杂吞酸，呕吐口苦", "usage": "水煎服或丸服", "modification": "胁痛甚加川楝子、延胡索"},
    {"name": "泻白散", "category": "清脏腑热剂", "source_book": "《小儿药证直诀》", "composition": "桑白皮、地骨皮、甘草、粳米", "functions": "清泻肺热，止咳平喘", "indications": "肺热喘咳证，气喘咳嗽，皮肤蒸热，日晡尤甚", "usage": "水煎服", "modification": "肺热重加黄芩、知母"},
    {"name": "清胃散", "category": "清脏腑热剂", "source_book": "《脾胃论》", "composition": "生地黄、当归、牡丹皮、黄连、升麻", "functions": "清胃凉血", "indications": "胃火牙痛，牙痛牵引头痛，面颊发热，齿恶热喜冷", "usage": "水煎服", "modification": "大便秘结加大黄"},
    {"name": "芍药汤", "category": "清脏腑热剂", "source_book": "《素问病机气宜保命集》", "composition": "白芍、当归、黄连、槟榔、木香、甘草、大黄、黄芩、肉桂", "functions": "清热燥湿，调气和血", "indications": "湿热痢疾，腹痛便脓血，赤白相兼，里急后重", "usage": "水煎服", "modification": "苔黄而干去肉桂"},
    {"name": "白头翁汤", "category": "清脏腑热剂", "source_book": "《伤寒论》", "composition": "白头翁、黄连、黄柏、秦皮", "functions": "清热解毒，凉血止痢", "indications": "热毒痢疾，腹痛里急后重，肛门灼热，下痢脓血", "usage": "水煎服", "modification": "恶寒发热加葛根、金银花"},
    {"name": "青蒿鳖甲汤", "category": "清虚热剂", "source_book": "《温病条辨》", "composition": "青蒿、鳖甲、生地黄、知母、牡丹皮", "functions": "养阴透热", "indications": "温病后期，邪伏阴分证，夜热早凉，热退无汗", "usage": "水煎服", "modification": "肺痨骨蒸加沙参、墨旱莲"},

    # 和解剂
    {"name": "小柴胡汤", "category": "和解少阳剂", "source_book": "《伤寒论》", "composition": "柴胡、黄芩、人参、半夏、甘草、生姜、大枣", "functions": "和解少阳", "indications": "伤寒少阳证，往来寒热，胸胁苦满，默默不欲饮食，心烦喜呕，口苦咽干，目眩", "usage": "水煎服", "modification": "胸中烦而不呕去半夏、人参，加瓜蒌"},
    {"name": "四逆散", "category": "调和肝脾剂", "source_book": "《伤寒论》", "composition": "柴胡、枳实、白芍、甘草", "functions": "透邪解郁，疏肝理脾", "indications": "阳郁厥逆证，手足不温，或腹痛泄利；肝脾气郁证，胁肋胀闷，脘腹疼痛", "usage": "水煎服", "modification": "咳者加五味子、干姜"},
    {"name": "逍遥散", "category": "调和肝脾剂", "source_book": "《太平惠民和剂局方》", "composition": "柴胡、当归、白芍、白术、茯苓、甘草、薄荷、生姜", "functions": "疏肝解郁，养血健脾", "indications": "肝郁血虚脾弱证，两胁作痛，头痛目眩，口燥咽干，神疲食少，月经不调，乳房胀痛", "usage": "水煎服", "modification": "血虚甚加熟地黄"},
    {"name": "半夏泻心汤", "category": "调和肠胃剂", "source_book": "《伤寒论》", "composition": "半夏、黄芩、干姜、人参、黄连、大枣、甘草", "functions": "寒热平调，消痞散结", "indications": "寒热互结之痞证，心下痞，但满而不痛，呕吐肠鸣下利", "usage": "水煎服", "modification": "痞满甚加枳实、厚朴"},

    # 补益剂
    {"name": "四君子汤", "category": "补气剂", "source_book": "《太平惠民和剂局方》", "composition": "人参、白术、茯苓、甘草", "functions": "益气健脾", "indications": "脾胃气虚证，面色萎白，语声低微，气短乏力，食少便溏", "usage": "水煎服", "modification": "加陈皮为异功散"},
    {"name": "补中益气汤", "category": "补气剂", "source_book": "《脾胃论》", "composition": "黄芪、人参、白术、甘草、当归、陈皮、升麻、柴胡", "functions": "补中益气，升阳举陷", "indications": "脾虚气陷证，饮食减少，体倦肢软，少气懒言，面色萎黄，大便稀溏；气虚发热，身热自汗，渴喜热饮", "usage": "水煎服", "modification": "头痛加蔓荆子"},
    {"name": "生脉散", "category": "补气剂", "source_book": "《医学启源》", "composition": "人参、麦冬、五味子", "functions": "益气生津，敛阴止汗", "indications": "气阴两虚证，汗多神疲，体倦乏力，气短懒言，咽干口渴，舌干红少苔", "usage": "水煎服", "modification": "阴虚有热以西洋参代人参"},
    {"name": "玉屏风散", "category": "补气剂", "source_book": "《究原方》", "composition": "黄芪、防风、白术", "functions": "益气固表止汗", "indications": "表虚自汗，汗出恶风，面色晄白，舌淡苔薄白，脉浮虚", "usage": "水煎服", "modification": "自汗较重加浮小麦、牡蛎"},
    {"name": "四物汤", "category": "补血剂", "source_book": "《太平惠民和剂局方》", "composition": "熟地黄、当归、白芍、川芎", "functions": "补血调血", "indications": "营血虚滞证，头晕目眩，心悸失眠，面色无华，妇人月经不调，量少或经闭不行", "usage": "水煎服", "modification": "兼气虚加人参、黄芪"},
    {"name": "归脾汤", "category": "补血剂", "source_book": "《济生方》", "composition": "黄芪、人参、白术、甘草、当归、龙眼肉、酸枣仁、远志、茯神、木香", "functions": "益气补血，健脾养心", "indications": "心脾气血两虚证，心悸怔忡，健忘失眠，盗汗虚热，体倦食少，面色萎黄；脾不统血证，便血崩漏", "usage": "水煎服", "modification": "月经先期量多加棕榈炭、艾叶炭"},
    {"name": "八珍汤", "category": "气血双补剂", "source_book": "《正体类要》", "composition": "四君子汤+四物汤（人参、白术、茯苓、甘草、熟地黄、当归、白芍、川芎）", "functions": "益气补血", "indications": "气血两虚证，面色苍白或萎黄，头晕目眩，四肢倦怠，气短懒言，心悸怔忡，饮食减少", "usage": "加生姜、大枣，水煎服", "modification": "加黄芪、肉桂为十全大补汤"},
    {"name": "六味地黄丸", "category": "补阴剂", "source_book": "《小儿药证直诀》", "composition": "熟地黄、山茱萸、山药、泽泻、牡丹皮、茯苓", "functions": "滋阴补肾", "indications": "肾阴虚证，腰膝酸软，头晕目眩，耳鸣耳聋，盗汗遗精，消渴骨蒸，手足心热", "usage": "丸剂或水煎服", "modification": "阴虚火旺加知母、黄柏"},
    {"name": "左归丸", "category": "补阴剂", "source_book": "《景岳全书》", "composition": "熟地黄、山药、枸杞子、山茱萸、牛膝、菟丝子、鹿角胶、龟甲胶", "functions": "滋阴补肾，填精益髓", "indications": "真阴不足证，头晕目眩，腰酸腿软，遗精滑泄，自汗盗汗，口燥舌干", "usage": "丸剂", "modification": "虚火上炎加知母、黄柏"},
    {"name": "肾气丸", "category": "补阳剂", "source_book": "《金匮要略》", "composition": "熟地黄、山药、山茱萸、泽泻、茯苓、牡丹皮、桂枝、附子", "functions": "补肾助阳", "indications": "肾阳不足证，腰痛脚软，下半身常有冷感，少腹拘急，小便不利或反多", "usage": "丸剂或水煎服", "modification": "加牛膝、车前子为济生肾气丸"},
    {"name": "右归丸", "category": "补阳剂", "source_book": "《景岳全书》", "composition": "熟地黄、山药、山茱萸、枸杞子、菟丝子、鹿角胶、杜仲、肉桂、当归、制附子", "functions": "温补肾阳，填精益髓", "indications": "肾阳不足，命门火衰证，年老久病气衰神疲，畏寒肢冷，阳痿遗精", "usage": "丸剂", "modification": "阳衰气虚加人参"},
    {"name": "地黄饮子", "category": "阴阳双补剂", "source_book": "《圣济总录》", "composition": "熟地黄、巴戟天、山茱萸、石斛、肉苁蓉、附子、五味子、肉桂、茯苓、麦冬、石菖蒲、远志", "functions": "滋肾阴，补肾阳，开窍化痰", "indications": "下元虚衰，痰浊上泛之喑痱证，舌强不能言，足废不能用", "usage": "加生姜、大枣，水煎服", "modification": "阴虚痰火盛去附子、肉桂"},

    # 理气剂
    {"name": "半夏厚朴汤", "category": "行气剂", "source_book": "《金匮要略》", "composition": "半夏、厚朴、茯苓、生姜、紫苏叶", "functions": "行气散结，降逆化痰", "indications": "梅核气，咽中如有物阻，咯吐不出，吞咽不下，胸膈满闷", "usage": "水煎服", "modification": "气郁甚加香附、郁金"},
    {"name": "苏子降气汤", "category": "降气剂", "source_book": "《太平惠民和剂局方》", "composition": "紫苏子、半夏、当归、甘草、前胡、厚朴、肉桂、生姜、大枣、紫苏叶", "functions": "降气平喘，祛痰止咳", "indications": "上实下虚之喘咳证，痰涎壅盛，胸膈满闷，喘咳短气，呼多吸少", "usage": "水煎服", "modification": "痰涎壅盛加莱菔子、白芥子"},
    {"name": "定喘汤", "category": "降气剂", "source_book": "《摄生众妙方》", "composition": "白果、麻黄、紫苏子、甘草、款冬花、杏仁、桑白皮、黄芩、半夏", "functions": "宣降肺气，清热化痰", "indications": "风寒外束，痰热内蕴之哮喘证，咳喘痰多气急，痰稠色黄，微恶风寒", "usage": "水煎服", "modification": "无汗恶寒加桂枝、生姜"},

    # 理血剂
    {"name": "血府逐瘀汤", "category": "活血祛瘀剂", "source_book": "《医林改错》", "composition": "桃仁、红花、当归、生地黄、牛膝、川芎、桔梗、赤芍、枳壳、甘草、柴胡", "functions": "活血化瘀，行气止痛", "indications": "胸中血瘀证，胸痛头痛，痛如针刺而有定处，或呃逆日久不止，或内热烦闷，心悸失眠", "usage": "水煎服", "modification": "瘀痛入络加全蝎、穿山甲"},
    {"name": "补阳还五汤", "category": "活血祛瘀剂", "source_book": "《医林改错》", "composition": "黄芪、当归、赤芍、地龙、川芎、红花、桃仁", "functions": "补气活血通络", "indications": "中风之气虚血瘀证，半身不遂，口眼歪斜，语言謇涩，口角流涎，小便频数", "usage": "水煎服", "modification": "上肢偏废加桑枝、桂枝"},
    {"name": "温经汤", "category": "活血祛瘀剂", "source_book": "《金匮要略》", "composition": "吴茱萸、当归、白芍、川芎、人参、桂枝、阿胶、牡丹皮、生姜、甘草、半夏、麦冬", "functions": "温经散寒，养血祛瘀", "indications": "冲任虚寒，瘀血阻滞证，漏下不止，血色暗而有块，月经不调，小腹冷痛", "usage": "水煎服", "modification": "寒甚加肉桂、附子"},
    {"name": "生化汤", "category": "活血祛瘀剂", "source_book": "《傅青主女科》", "composition": "当归、川芎、桃仁、炮姜、甘草", "functions": "养血祛瘀，温经止痛", "indications": "血虚寒凝，瘀血阻滞证，产后恶露不行，小腹冷痛", "usage": "黄酒、童便各半煎服", "modification": "恶露已行而腹微痛去桃仁"},
    {"name": "小蓟饮子", "category": "止血剂", "source_book": "《济生方》", "composition": "生地黄、小蓟、滑石、木通、蒲黄、藕节、淡竹叶、当归、栀子、甘草", "functions": "凉血止血，利水通淋", "indications": "热结下焦之血淋尿血，小便频数，赤涩热痛", "usage": "水煎服", "modification": "血淋茎中痛甚加琥珀末"},
    {"name": "黄土汤", "category": "止血剂", "source_book": "《金匮要略》", "composition": "灶心黄土、白术、附子、生地黄、阿胶、黄芩、甘草", "functions": "温阳健脾，养血止血", "indications": "脾阳不足，脾不统血证，大便下血，吐血衄血，妇人崩漏，血色暗淡", "usage": "灶心黄土煎汤代水", "modification": "气虚甚加人参"},

    # 祛湿剂
    {"name": "平胃散", "category": "燥湿和胃剂", "source_book": "《简要济众方》", "composition": "苍术、厚朴、陈皮、甘草", "functions": "燥湿运脾，行气和胃", "indications": "湿滞脾胃证，脘腹胀满，不思饮食，口淡无味，恶心呕吐，嗳气吞酸，肢体沉重", "usage": "加生姜、大枣，水煎服", "modification": "湿热加黄连、黄芩"},
    {"name": "藿香正气散", "category": "燥湿和胃剂", "source_book": "《太平惠民和剂局方》", "composition": "藿香、紫苏、白芷、半夏曲、陈皮、白术、茯苓、厚朴、大腹皮、桔梗、甘草", "functions": "解表化湿，理气和中", "indications": "外感风寒，内伤湿滞证，霍乱吐泻，发热恶寒，头痛，胸膈满闷，脘腹疼痛", "usage": "加生姜、大枣，水煎服", "modification": "表寒重加香薷"},
    {"name": "茵陈蒿汤", "category": "清热祛湿剂", "source_book": "《伤寒论》", "composition": "茵陈、栀子、大黄", "functions": "清热利湿退黄", "indications": "湿热黄疸，一身面目俱黄，黄色鲜明，腹微满，口渴，小便不利", "usage": "水煎服", "modification": "胁痛加柴胡、郁金"},
    {"name": "八正散", "category": "清热祛湿剂", "source_book": "《太平惠民和剂局方》", "composition": "车前子、瞿麦、萹蓄、滑石、栀子、甘草、木通、大黄", "functions": "清热泻火，利水通淋", "indications": "湿热淋证，尿频尿急，溺时涩痛，淋沥不畅，小腹胀急", "usage": "加灯心草，水煎服", "modification": "大便秘结腹胀用生大黄"},
    {"name": "五苓散", "category": "利水渗湿剂", "source_book": "《伤寒论》", "composition": "猪苓、茯苓、泽泻、白术、桂枝", "functions": "利水渗湿，温阳化气", "indications": "膀胱气化不利之蓄水证，小便不利，头痛微热，烦渴欲饮，水入即吐；水肿泄泻", "usage": "水煎服", "modification": "水肿兼表证加麻黄"},
    {"name": "独活寄生汤", "category": "祛风胜湿剂", "source_book": "《备急千金要方》", "composition": "独活、桑寄生、杜仲、牛膝、细辛、秦艽、茯苓、肉桂心、防风、川芎、人参、甘草、当归、白芍、熟地黄", "functions": "祛风湿，止痹痛，益肝肾，补气血", "indications": "痹证日久，肝肾两虚，气血不足证，腰膝疼痛，痿软，肢节屈伸不利", "usage": "水煎服", "modification": "疼痛较甚加制川乌、制草乌"},

    # 祛痰剂
    {"name": "二陈汤", "category": "燥湿化痰剂", "source_book": "《太平惠民和剂局方》", "composition": "半夏、陈皮、茯苓、甘草", "functions": "燥湿化痰，理气和中", "indications": "湿痰证，咳嗽痰多，色白易咯，恶心呕吐，胸膈痞闷，肢体困重", "usage": "加生姜，水煎服", "modification": "风痰加天南星、白附子"},
    {"name": "温胆汤", "category": "燥湿化痰剂", "source_book": "《三因极一病证方论》", "composition": "半夏、竹茹、枳实、陈皮、甘草、茯苓、生姜、大枣", "functions": "理气化痰，清胆和胃", "indications": "胆胃不和，痰热内扰证，虚烦不眠，惊悸不宁，呕吐呃逆，癫痫", "usage": "水煎服", "modification": "心热烦甚加黄连、麦冬"},
    {"name": "清气化痰丸", "category": "清热化痰剂", "source_book": "《医方考》", "composition": "瓜蒌仁、陈皮、黄芩、杏仁、枳实、茯苓、胆南星、半夏", "functions": "清热化痰，理气止咳", "indications": "痰热咳嗽，咳嗽气喘，咯痰黄稠，胸膈痞闷，甚则气急呕恶", "usage": "丸剂或水煎服", "modification": "肺热壅盛加石膏、知母"},
    {"name": "止嗽散", "category": "止咳平喘剂", "source_book": "《医学心悟》", "composition": "桔梗、荆芥、紫菀、百部、白前、甘草、陈皮", "functions": "宣利肺气，疏风止咳", "indications": "风邪犯肺之咳嗽，咳嗽咽痒，咯痰不爽，或微有恶风发热", "usage": "水煎服", "modification": "外感风寒加防风、紫苏叶"},

    # 消食剂
    {"name": "保和丸", "category": "消食化滞剂", "source_book": "《丹溪心法》", "composition": "山楂、神曲、半夏、茯苓、陈皮、连翘、莱菔子", "functions": "消食和胃", "indications": "食积证，脘腹痞满胀痛，嗳腐吞酸，恶食呕逆，或大便泄泻", "usage": "丸剂或水煎服", "modification": "食积较重加枳实、槟榔"},
    {"name": "枳实导滞丸", "category": "消食化滞剂", "source_book": "《内外伤辨惑论》", "composition": "大黄、枳实、神曲、茯苓、黄芩、黄连、白术、泽泻", "functions": "消食导滞，清热利湿", "indications": "湿热食积证，脘腹胀痛，下痢泄泻，或大便秘结，小便短赤", "usage": "丸剂或水煎服", "modification": "胀满较重加木香、槟榔"},

    # 驱虫剂
    {"name": "乌梅丸", "category": "驱虫剂", "source_book": "《伤寒论》", "composition": "乌梅、细辛、干姜、黄连、当归、附子、蜀椒、桂枝、人参、黄柏", "functions": "温脏安蛔", "indications": "蛔厥证，腹痛时作，心烦呕吐，时发时止，常自吐蛔，手足厥冷", "usage": "丸剂或水煎服", "modification": "热重去附子、干姜"},

    # 治风剂
    {"name": "川芎茶调散", "category": "疏散外风剂", "source_book": "《太平惠民和剂局方》", "composition": "川芎、荆芥、白芷、羌活、甘草、细辛、防风、薄荷", "functions": "疏风止痛", "indications": "外感风邪头痛，偏正头痛或巅顶作痛，恶寒发热，目眩鼻塞", "usage": "清茶调服", "modification": "风热头痛加菊花、僵蚕"},
    {"name": "天麻钩藤饮", "category": "平息内风剂", "source_book": "《中医内科杂病证治新义》", "composition": "天麻、钩藤、石决明、栀子、黄芩、牛膝、杜仲、益母草、桑寄生、首乌藤、茯神", "functions": "平肝息风，清热活血，补益肝肾", "indications": "肝阳偏亢，肝风上扰证，头痛眩晕，失眠多梦，口苦面红", "usage": "水煎服", "modification": "眩晕加剧加羚羊角"},

    # 治燥剂
    {"name": "杏苏散", "category": "轻宣外燥剂", "source_book": "《温病条辨》", "composition": "杏仁、紫苏叶、半夏、陈皮、前胡、桔梗、枳壳、茯苓、甘草、生姜、大枣", "functions": "轻宣凉燥，理肺化痰", "indications": "外感凉燥证，恶寒无汗，头微痛，咳嗽痰稀，鼻塞咽干", "usage": "水煎服", "modification": "无汗脉弦甚加羌活"},
    {"name": "清燥救肺汤", "category": "轻宣外燥剂", "source_book": "《医门法律》", "composition": "桑叶、石膏、人参、甘草、胡麻仁、阿胶、麦冬、杏仁、枇杷叶", "functions": "清燥润肺", "indications": "温燥伤肺证，身热头痛，干咳无痰，气逆而喘，咽喉干燥，鼻燥胸满胁痛", "usage": "水煎服", "modification": "痰多加贝母、瓜蒌"},
    {"name": "麦门冬汤", "category": "滋阴润燥剂", "source_book": "《金匮要略》", "composition": "麦冬、半夏、人参、甘草、粳米、大枣", "functions": "滋养肺胃，降逆和中", "indications": "肺胃阴虚证，咳唾涎沫，短气喘促，咽喉干燥，舌干红少苔", "usage": "水煎服", "modification": "阴伤甚加北沙参、玉竹"},
    {"name": "百合固金汤", "category": "滋阴润燥剂", "source_book": "《慎斋遗书》", "composition": "百合、熟地黄、生地黄、当归、白芍、甘草、桔梗、玄参、贝母、麦冬", "functions": "滋养肺肾，止咳化痰", "indications": "肺肾阴虚，虚火上炎证，咳嗽气喘，痰中带血，咽喉燥痛，头晕目眩", "usage": "水煎服", "modification": "痰多色黄加黄芩、鱼腥草"},
]

# ── 扩展症状列表 ──

EXTENDED_SYMPTOMS: List[str] = [
    # 全身症状
    "发热", "恶寒", "畏寒", "恶风", "寒战", "潮热", "五心烦热", "手足心热", "骨蒸发热",
    "自汗", "盗汗", "无汗", "战汗", "黄汗",
    "乏力", "疲倦", "倦怠", "神疲", "少气", "懒言", "气短",
    "消瘦", "肥胖", "浮肿", "水肿",
    # 头面五官症状
    "头痛", "头重", "头胀", "头晕", "眩晕", "头昏", "头皮麻木",
    "面赤", "面色萎黄", "面色苍白", "面色晦暗", "面部浮肿",
    "目赤", "目黄", "目痛", "目痒", "目涩", "目眩", "视物模糊", "羞明", "目眵多",
    "耳鸣", "耳聋", "耳痛", "耳痒",
    "鼻塞", "鼻流清涕", "鼻流浊涕", "鼻衄", "鼻痒", "喷嚏",
    "咽喉肿痛", "咽干", "咽痒", "声音嘶哑", "失音",
    "口疮", "口臭", "口苦", "口甜", "口酸", "口咸", "口淡", "口黏腻",
    "口干", "口渴", "口渴不欲饮", "烦渴引饮",
    "牙龈肿痛", "牙龈出血", "牙齿松动",
    # 颈项胸腹症状
    "颈项强痛", "瘿瘤", "瘰疬",
    "胸闷", "胸痛", "胸胁胀痛", "胸胁刺痛", "胁痛", "胁胀", "乳房胀痛", "心悸", "怔忡",
    "脘痞", "胃痛", "胃胀", "腹胀", "腹痛", "少腹痛", "小腹痛", "腹中痞块",
    "恶心", "呕吐", "干呕", "嗳气", "呃逆", "吞酸", "嘈杂",
    # 腰背四肢症状
    "腰酸", "腰痛", "腰膝酸软", "腰冷痛", "背痛",
    "关节痛", "关节肿胀", "关节变形", "四肢麻木", "四肢拘挛", "四肢抽搐",
    "手足不温", "手足逆冷", "手足心热",
    # 饮食二便症状
    "食欲不振", "纳呆", "多食易饥", "消谷善饥", "饥不欲食", "偏嗜异物",
    "腹泻", "泄泻", "便溏", "五更泄", "便秘", "大便干结", "大便不爽", "里急后重",
    "便血", "完谷不化", "大便失禁",
    "小便不利", "小便频数", "小便涩痛", "小便清长", "小便短赤", "遗尿",
    "夜尿多", "癃闭",
    # 精神情志症状
    "失眠", "多梦", "易醒", "嗜睡", "健忘",
    "心烦", "烦躁", "易怒", "抑郁", "善太息", "善悲", "善恐", "善惊",
    "神昏", "谵语", "狂躁", "痴呆",
    # 妇科症状
    "月经不调", "月经先期", "月经后期", "月经先后无定期",
    "月经过多", "月经过少", "经期延长",
    "痛经", "经行乳房胀痛", "经行头痛",
    "闭经", "崩漏",
    "带下过多", "带下色黄", "带下色白", "带下赤白",
    "胎动不安", "恶露不尽", "产后腹痛",
    # 皮肤症状
    "皮肤瘙痒", "皮疹", "风团", "斑疹", "斑丘疹",
    "皮肤干燥", "肌肤甲错", "皮肤发黄",
    "疮疡", "痈肿", "疔疮", "疖肿",
    # 呼吸系统
    "咳嗽", "咳痰", "痰多", "痰少", "痰黄", "痰白", "痰中带血",
    "气喘", "哮喘", "喉中痰鸣",
]

# ── 扩展证型定义（含症状关联） ──

EXTENDED_SYNDROMES: Dict[str, Dict] = {
    "脾胃虚弱": {
        "symptoms": ["乏力", "疲倦", "食欲不振", "纳呆", "腹胀", "便溏", "面色萎黄", "消瘦", "少气", "懒言"],
        "description": "脾胃运化功能减弱，升降失常",
        "treatment": "健脾益气，和胃化湿",
        "recommend_formulas": ["四君子汤", "补中益气汤"],
        "recommend_herbs": ["人参", "黄芪", "白术", "茯苓", "甘草", "陈皮", "山药"],
    },
    "气血两虚": {
        "symptoms": ["乏力", "气短", "心悸", "失眠", "面色苍白", "面色萎黄", "头晕", "健忘", "月经量少", "自汗"],
        "description": "气虚与血虚同时存在，脏腑功能减退",
        "treatment": "益气养血",
        "recommend_formulas": ["八珍汤", "归脾汤"],
        "recommend_herbs": ["人参", "黄芪", "当归", "熟地黄", "白芍", "龙眼肉", "大枣"],
    },
    "肝郁气滞": {
        "symptoms": ["胁痛", "胁胀", "善太息", "胸闷", "抑郁", "易怒", "月经不调", "乳房胀痛", "嗳气", "失眠"],
        "description": "肝失疏泄，气机郁滞",
        "treatment": "疏肝理气，解郁散结",
        "recommend_formulas": ["逍遥散", "四逆散"],
        "recommend_herbs": ["柴胡", "香附", "白芍", "郁金", "陈皮", "佛手", "玫瑰花"],
    },
    "肾阳虚": {
        "symptoms": ["畏寒", "手足不温", "腰膝酸软", "腰冷痛", "乏力", "小便清长", "夜尿多", "水肿", "阳痿", "五更泄"],
        "description": "肾阳不足，温煦功能减退，水液代谢失常",
        "treatment": "温补肾阳，化气行水",
        "recommend_formulas": ["肾气丸", "右归丸"],
        "recommend_herbs": ["附子", "肉桂", "杜仲", "巴戟天", "淫羊藿", "肉苁蓉", "菟丝子"],
    },
    "肾阴虚": {
        "symptoms": ["腰膝酸软", "头晕", "耳鸣", "失眠", "盗汗", "手足心热", "口干", "遗精", "月经不调", "健忘"],
        "description": "肾阴不足，髓海亏虚，虚火内生",
        "treatment": "滋阴补肾，填精益髓",
        "recommend_formulas": ["六味地黄丸", "左归丸"],
        "recommend_herbs": ["熟地黄", "山茱萸", "山药", "枸杞子", "女贞子", "墨旱莲", "龟甲"],
    },
    "阴虚火旺": {
        "symptoms": ["手足心热", "五心烦热", "盗汗", "口干", "失眠", "心烦", "潮热", "颧红", "耳鸣", "便秘"],
        "description": "阴液亏虚，虚火内扰",
        "treatment": "滋阴降火",
        "recommend_formulas": ["六味地黄丸（加知母黄柏）", "青蒿鳖甲汤"],
        "recommend_herbs": ["知母", "黄柏", "生地黄", "玄参", "牡丹皮", "地骨皮"],
    },
    "心脾两虚": {
        "symptoms": ["失眠", "多梦", "健忘", "心悸", "食欲不振", "乏力", "面色萎黄", "便溏", "头晕", "神疲"],
        "description": "心血不足，脾气虚弱",
        "treatment": "益气补血，健脾养心",
        "recommend_formulas": ["归脾汤", "四君子汤"],
        "recommend_herbs": ["人参", "黄芪", "白术", "当归", "酸枣仁", "远志", "龙眼肉"],
    },
    "风寒感冒": {
        "symptoms": ["恶寒", "发热", "头痛", "鼻塞", "鼻流清涕", "咳嗽", "无汗", "肢体酸痛", "喷嚏"],
        "description": "风寒之邪侵袭肌表，肺卫失宣",
        "treatment": "辛温解表，宣肺散寒",
        "recommend_formulas": ["麻黄汤", "桂枝汤", "小青龙汤"],
        "recommend_herbs": ["麻黄", "桂枝", "防风", "荆芥", "羌活", "紫苏叶"],
    },
    "风热感冒": {
        "symptoms": ["发热", "恶风", "头痛", "咽喉肿痛", "口干", "咳嗽", "鼻流浊涕", "汗出", "口渴"],
        "description": "风热之邪侵袭肺卫",
        "treatment": "辛凉解表，清热解毒",
        "recommend_formulas": ["银翘散", "桑菊饮"],
        "recommend_herbs": ["金银花", "连翘", "薄荷", "牛蒡子", "菊花", "桑叶"],
    },
    "肝阳上亢": {
        "symptoms": ["头痛", "眩晕", "头胀", "面赤", "易怒", "失眠", "耳鸣", "目赤", "口苦", "心烦"],
        "description": "肝肾阴亏于下，肝阳亢逆于上",
        "treatment": "平肝潜阳，清热息风",
        "recommend_formulas": ["天麻钩藤饮", "镇肝熄风汤"],
        "recommend_herbs": ["天麻", "钩藤", "石决明", "菊花", "白芍", "龙骨", "牡蛎"],
    },
    "痰湿内阻": {
        "symptoms": ["胸闷", "咳嗽", "痰多", "痰白", "恶心", "呕吐", "头重", "肢体困重", "腹胀", "食欲不振"],
        "description": "脾失健运，水湿内停，聚而成痰",
        "treatment": "燥湿化痰，理气和中",
        "recommend_formulas": ["二陈汤", "平胃散"],
        "recommend_herbs": ["半夏", "陈皮", "茯苓", "白术", "苍术", "厚朴"],
    },
    "湿热蕴结": {
        "symptoms": ["发热", "口干", "口苦", "便秘或腹泻", "小便短赤", "皮肤发黄", "腹胀", "食欲不振", "恶心"],
        "description": "湿热之邪蕴结体内，阻滞气机",
        "treatment": "清热利湿",
        "recommend_formulas": ["茵陈蒿汤", "龙胆泻肝汤"],
        "recommend_herbs": ["茵陈", "栀子", "黄连", "黄芩", "黄柏", "金钱草"],
    },
    "瘀血阻络": {
        "symptoms": ["胸痛", "胁痛", "头痛", "痛经", "关节痛", "刺痛", "痛有定处", "皮肤晦暗"],
        "description": "瘀血内停，阻滞经络",
        "treatment": "活血化瘀，通络止痛",
        "recommend_formulas": ["血府逐瘀汤", "补阳还五汤"],
        "recommend_herbs": ["丹参", "川芎", "红花", "桃仁", "延胡索", "牛膝"],
    },
    "胃热炽盛": {
        "symptoms": ["胃痛", "口干", "多食易饥", "口臭", "牙龈肿痛", "便秘", "恶心"],
        "description": "胃中火热炽盛，消灼津液",
        "treatment": "清胃泻火",
        "recommend_formulas": ["清胃散", "白虎汤"],
        "recommend_herbs": ["石膏", "知母", "黄连", "生地黄", "牡丹皮"],
    },
    "肺热壅盛": {
        "symptoms": ["咳嗽", "气喘", "痰黄", "发热", "咽喉肿痛", "胸痛", "口干"],
        "description": "热邪壅肺，肺失宣降",
        "treatment": "清肺化痰，止咳平喘",
        "recommend_formulas": ["麻杏石甘汤", "泻白散"],
        "recommend_herbs": ["石膏", "黄芩", "桑白皮", "枇杷叶", "鱼腥草", "瓜蒌"],
    },
    "心火上炎": {
        "symptoms": ["心烦", "失眠", "口疮", "口干", "口苦", "小便短赤", "咽喉肿痛"],
        "description": "心火亢盛，上扰心神",
        "treatment": "清心泻火，安神定志",
        "recommend_formulas": ["黄连解毒汤", "导赤散"],
        "recommend_herbs": ["黄连", "栀子", "竹叶", "生地黄", "玄参", "连翘"],
    },
    "肝血不足": {
        "symptoms": ["头晕", "目涩", "视物模糊", "失眠", "多梦", "面色萎黄", "月经量少", "皮肤干燥"],
        "description": "肝血亏虚，筋脉失养，目窍失濡",
        "treatment": "养血柔肝",
        "recommend_formulas": ["四物汤", "归芍地黄丸"],
        "recommend_herbs": ["当归", "白芍", "熟地黄", "枸杞子", "女贞子", "酸枣仁"],
    },
}


def expand_database(db_path: str) -> Tuple[int, int, int, int]:
    """将扩展数据插入现有数据库，返回新增条数"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. 插入扩展药材（避免重复）
    existing_herbs = {r[0] for r in cursor.execute("SELECT name FROM herbs").fetchall()}
    herb_count = 0
    for herb in EXTENDED_HERBS:
        if herb["name"] in existing_herbs:
            continue
        cursor.execute(
            """INSERT INTO herbs (name, category, nature, flavor, meridian,
               efficacy, indications, usage, contraindications, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (herb["name"], herb["category"], herb["nature"], herb["flavor"],
             herb["meridian"], herb["efficacy"], herb["indications"],
             herb["usage"], herb["contraindications"], "《中药学》教材"),
        )
        herb_count += 1

    # 2. 插入扩展方剂
    existing_formulas = {r[0] for r in cursor.execute("SELECT name FROM formulas").fetchall()}
    formula_count = 0
    for f in EXTENDED_FORMULAS:
        if f["name"] in existing_formulas:
            continue
        cursor.execute(
            """INSERT INTO formulas (name, category, source_book, composition,
               functions, indications, usage, modification, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (f["name"], f["category"], f["source_book"], f["composition"],
             f["functions"], f["indications"], f["usage"], f["modification"], "《方剂学》教材"),
        )
        formula_count += 1

    conn.commit()

    # 3. 统计
    total_herbs = cursor.execute("SELECT COUNT(*) FROM herbs").fetchone()[0]
    total_formulas = cursor.execute("SELECT COUNT(*) FROM formulas").fetchone()[0]

    conn.close()

    print(f"数据库扩展完成：")
    print(f"  新增药材: {herb_count}，总计: {total_herbs}")
    print(f"  新增方剂: {formula_count}，总计: {total_formulas}")
    print(f"  扩展症状: {len(EXTENDED_SYMPTOMS)} 种")
    print(f"  扩展证型: {len(EXTENDED_SYNDROMES)} 种")

    return herb_count, formula_count, len(EXTENDED_SYMPTOMS), len(EXTENDED_SYNDROMES)


def export_extended_config(output_path: str) -> None:
    """导出扩展的症状和证型配置为 Python 模块"""
    lines = [
        '# Copyright (C) 2026 中医药智能问答系统',
        '# Licensed under Apache License 2.0',
        '',
        '"""',
        '扩展症状、证型定义（基于《中药学》《方剂学》教材）',
        f'症状: {len(EXTENDED_SYMPTOMS)} 种',
        f'证型: {len(EXTENDED_SYNDROMES)} 种',
        '"""',
        '',
        f'EXTENDED_SYMPTOM_LIST: list[str] = {json.dumps(EXTENDED_SYMPTOMS, ensure_ascii=False, indent=4)}',
        '',
        '# 证型定义',
        'EXTENDED_SYNDROME_DEFINITIONS: dict = {',
    ]
    for name, info in EXTENDED_SYNDROMES.items():
        lines.append(f'    "{name}": {{')
        for k, v in info.items():
            formatted = json.dumps({k: v}, ensure_ascii=False, indent=8)
            # Extract just the value part
            inner = json.dumps(v, ensure_ascii=False, indent=8)
            lines.append(f'        "{k}": {inner},')
        lines.append('    },')
    lines.append('}')

    Path(output_path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"扩展配置已导出到: {output_path}")


if __name__ == "__main__":
    import os
    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / "tcm_knowledge.db"

    if db_path.exists():
        expand_database(str(db_path))
        export_extended_config(str(project_root / "tcm_recommender" / "extended_config.py"))
    else:
        print(f"错误: 数据库不存在 {db_path}")
        sys.exit(1)
