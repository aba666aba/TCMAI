# -*- coding: utf-8 -*-
# Copyright (C) 2026 aba666aba
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# 本文件是中医药智能问答系统 (TCMAI) 的一部分。
#
# 本程序是自由软件：你可以依据自由软件基金会发布的 GNU Affero 通用公共
# 许可证条款（第 3 版或你选择的任何更新版本）重新分发和/或修改本程序。
#
# 本程序按“希望有用”的原则分发，但不提供任何担保，甚至不提供适销性或
# 特定用途适用性的默示担保。详见 GNU Affero 通用公共许可证。
#
# 你应已随本程序收到 GNU Affero 通用公共许可证副本。如未收到，
# 请见 <https://www.gnu.org/licenses/>。
#
# 注意：以网络服务方式提供本程序的服务端修改版本，同样必须按 AGPL-3.0
# 第 13 条向所有用户提供对应源代码。
"""
补充缺主治方剂的主治文本

策略：
  1. 有组成的方剂 → 从组成药材的功效/主治聚合生成
  2. 有分类的方剂 → 按分类映射到标准主治文本
  3. 有名称提示的 → 从方剂名提取（如"腰痛汤"→腰痛）
  4. 以上组合
"""

import sys, os, re, sqlite3, logging
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent / 'tcm_knowledge.db'

# ================================================================
# 分类→主治映射（基于中医方剂学教材标准）
# ================================================================
CATEGORY_TO_INDICATIONS: dict[str, str] = {
    "补益剂": "虚证",
    "补益剂 - 补气": "气虚证，神疲乏力，气短懒言，食欲不振，面色苍白",
    "补益剂 - 补血": "血虚证，面色萎黄，头晕目眩，心悸失眠，月经不调",
    "补益剂 - 气血双补": "气血两虚证，面色苍白或萎黄，头晕目眩，神疲乏力，气短懒言，心悸失眠",
    "补益剂 - 补阴": "阴虚证，形体消瘦，口燥咽干，潮热盗汗，五心烦热",
    "补益剂 - 补阳": "阳虚证，畏寒肢冷，腰膝酸软，神疲乏力",
    "补益剂 - 滋阴养液": "阴虚津亏证，口干咽燥，舌干少津，大便干结",
    "解表剂": "外感表证",
    "解表剂 - 辛温解表": "风寒感冒，恶寒发热，头痛身痛，鼻塞流涕，无汗",
    "解表剂 - 辛凉解表": "风热感冒，发热，微恶风寒，咽痛口渴，咳嗽",
    "解表剂 - 扶正解表": "体虚感冒，恶寒发热，头痛，气短乏力，咳嗽",
    "解表剂 - 止咳化痰": "咳嗽，咳痰，气喘",
    "清热剂": "热证",
    "清热剂 - 清热解毒": "热毒证，咽喉肿痛，痈肿疮毒，热毒泻痢",
    "清热剂 - 清脏腑热": "脏腑热证，心烦失眠，口舌生疮，胃热呕吐，肝火目赤",
    "清热剂 - 清气分热": "气分热证，壮热，烦渴，大汗，脉洪大",
    "清热剂 - 清营凉血": "营血热证，身热夜甚，心烦不寐，吐血衄血，斑疹",
    "清热剂 - 清虚热": "虚热证，骨蒸潮热，盗汗，五心烦热",
    "清热剂 - 清热祛暑": "暑热证，身热，心烦，口渴，小便短赤",
    "祛湿剂": "湿证",
    "祛湿剂 - 燥湿和胃": "湿阻中焦，脘腹胀满，食欲不振，恶心呕吐，肢体困重",
    "祛湿剂 - 清热祛湿": "湿热证，身热不扬，脘腹胀满，小便黄赤，舌苔黄腻",
    "祛湿剂 - 利水渗湿": "水湿内停，水肿，小便不利，泄泻",
    "祛湿剂 - 温化寒湿": "寒湿证，畏寒肢冷，脘腹冷痛，大便溏泄",
    "祛湿剂 - 祛风胜湿": "风湿痹证，关节疼痛，肢体沉重，屈伸不利",
    "温里剂": "里寒证",
    "温里剂 - 温中祛寒": "脾胃虚寒，脘腹冷痛，呕吐泄泻，食欲不振",
    "温里剂 - 回阳救逆": "阳气衰微，四肢厥冷，神衰欲寐，脉微欲绝",
    "温里剂 - 温经散寒": "寒凝经脉，手足厥冷，肢体冷痛，冻疮",
    "温里剂 - 温通心阳": "心阳不振，心悸气短，胸闷胸痛，畏寒肢冷",
    "温里剂 - 温中缓急": "中焦虚寒，脘腹拘急疼痛，喜温喜按",
    "理气剂": "气滞证",
    "理气剂 - 行气": "气滞证，胸胁胀痛，脘腹胀满，嗳气",
    "理气剂 - 降气": "气逆证，呕吐，呃逆，喘咳",
    "理血剂": "血证",
    "理血剂 - 活血化瘀": "瘀血证，痛有定处，刺痛，癥瘕积聚，跌打损伤",
    "理血剂 - 止血": "出血证，吐血，衄血，便血，崩漏",
    "治风剂": "风证",
    "治风剂 - 疏散外风": "外风证，头痛，眩晕，风疹，风湿痹痛",
    "治风剂 - 平息内风": "内风证，眩晕，震颤，抽搐，中风",
    "治燥剂": "燥证",
    "治燥剂 - 轻宣润燥": "外感燥证，咽干鼻燥，干咳无痰",
    "治燥剂 - 滋阴润燥": "内燥证，口干咽燥，干咳少痰，大便干结",
    "祛痰剂": "痰证",
    "祛痰剂 - 燥湿化痰": "湿痰证，咳嗽痰多，胸闷，恶心",
    "祛痰剂 - 清热化痰": "热痰证，咳嗽痰黄，胸闷，发热",
    "祛痰剂 - 润燥化痰": "燥痰证，干咳少痰，咽干",
    "祛痰剂 - 温化寒痰": "寒痰证，咳嗽痰白清稀，畏寒",
    "祛痰剂 - 化痰熄风": "风痰证，眩晕，呕吐痰涎",
    "消食剂": "食积证",
    "消食剂 - 消食导滞": "食积证，脘腹胀满，嗳腐吞酸，食欲不振",
    "消食剂 - 健脾消食": "脾虚食积，食少难消，脘腹胀满，神疲乏力",
    "和解剂": "半表半里证",
    "和解剂 - 和解少阳": "少阳证，寒热往来，胸胁苦满，心烦喜呕",
    "和解剂 - 调和肝脾": "肝脾不和，胸胁胀痛，腹痛泄泻，月经不调",
    "和解剂 - 调和肠胃": "肠胃不和，心下痞满，恶心呕吐，肠鸣泄泻",
    "固涩剂": "滑脱证",
    "固涩剂 - 固表止汗": "表虚自汗，盗汗",
    "固涩剂 - 敛肺止咳": "久咳肺虚，咳嗽气喘",
    "固涩剂 - 涩肠固脱": "久泻久痢，滑脱不禁",
    "固涩剂 - 涩精止遗": "遗精滑精，遗尿尿频",
    "固涩剂 - 固崩止带": "崩漏，带下",
    "安神剂": "神志不安证",
    "安神剂 - 重镇安神": "心神不安，惊悸失眠，烦躁易怒",
    "安神剂 - 补养安神": "心脾两虚，心悸失眠，健忘多梦",
    "开窍剂": "窍闭神昏证",
    "开窍剂 - 凉开": "热闭证，高热神昏，谵语",
    "开窍剂 - 温开": "寒闭证，突然昏倒，牙关紧闭",
    "泻下剂": "里实证",
    "泻下剂 - 寒下": "实热积滞，大便秘结，腹痛拒按，发热",
    "泻下剂 - 温下": "寒积便秘，腹痛，手足不温",
    "泻下剂 - 润下": "肠燥便秘，大便干结",
    "泻下剂 - 攻补兼施": "里实正虚，便秘，神疲乏力",
    "泻下剂 - 逐水": "水饮壅盛，水肿，胸腹积水",
    "祛暑剂": "暑证",
    "祛暑剂 - 祛暑解表": "暑湿感冒，身热，恶寒，头重，胸闷",
    "祛暑剂 - 清暑利湿": "暑湿证，身热烦渴，小便不利",
    "祛暑剂 - 清心解暑": "暑热证，身热，心烦，口渴",
    "祛暑剂 - 祛暑清肝": "暑热扰肝，心烦，目赤",
    "祛暑剂 - 清暑益气": "暑热气津两伤，身热汗多，口渴，神疲",
    "表里双解剂": "表里同病",
    "驱虫剂": "虫证",
    "专科方": "专科病证",
    "临床经验方": "临床经验方剂",
    "其他/专科方": "专科病证",
}

# 分类备用映射（部分分类名包含完整路径，匹配不上时用子串匹配）
CATEGORY_FALLBACK: dict[str, str] = {}
for k, v in CATEGORY_TO_INDICATIONS.items():
    # 取最后一个 - 后面的部分
    parts = k.split(' - ')
    for p in parts:
        CATEGORY_FALLBACK[p] = v

# 常见方剂名的症状映射
NAME_KEYWORD_INDICATIONS = {
    '感冒': '感冒，恶寒发热，头痛鼻塞',
    '咳嗽': '咳嗽，咳痰',
    '发热': '发热，身热',
    '止痛': '疼痛',
    '腰痛': '腰痛，腰膝酸软',
    '腿痛': '腿痛，下肢疼痛',
    '胸胁': '胸胁胀痛，胁痛',
    '胃痛': '胃痛，胃脘疼痛',
    '腹痛': '腹痛',
    '头痛': '头痛',
    '眩晕': '眩晕，头晕目眩',
    '失眠': '失眠，不寐',
    '安神': '失眠，心悸，心神不安',
    '水肿': '水肿，小便不利',
    '出血': '出血，吐血，衄血，便血',
    '活血': '瘀血证，跌打损伤，疼痛',
    '化瘀': '瘀血证，癥瘕积聚',
    '接骨': '骨折，跌打损伤，肿痛',
    '正骨': '骨折，跌打损伤',
    '消肿': '肿痛，跌打损伤',
    '止痛': '疼痛，诸痛',
    '祛风': '风证，风湿痹痛',
    '除湿': '湿证，风湿痹痛',
    '通络': '痹证，经络不通，麻木',
    '壮骨': '筋骨痿软，腰膝酸软',
    '补骨': '肾虚骨痿，腰膝酸软',
    '补气': '气虚证，神疲乏力，气短懒言，食欲不振，面色苍白',
    '补血': '血虚证，面色萎黄，头晕目眩，心悸失眠，月经不调',
    '补肾': '肾虚证，腰膝酸软，耳鸣耳聋，头晕',
    '补阴': '阴虚证，形体消瘦，口燥咽干，潮热盗汗，五心烦热',
    '滋阴': '阴虚证，口燥咽干，潮热盗汗，五心烦热',
    '补阳': '阳虚证，畏寒肢冷，腰膝酸软，神疲乏力',
    '壮阳': '肾阳虚证，畏寒肢冷，腰膝酸软，阳痿早泄',
    '气血双补': '气血两虚证，面色苍白或萎黄，头晕目眩，神疲乏力，气短懒言',
    '益气': '气虚证，神疲乏力，气短，自汗',
    '养血': '血虚证，面色萎黄，头晕心悸，失眠',
    '扶正': '正气虚弱证，神疲乏力，气短，自汗，易感冒',
    # 方剂名前缀中的分类关键词
    '辛凉解表': '风热感冒，发热，微恶风寒，咽痛口渴，咳嗽',
    '辛温解表': '风寒感冒，恶寒发热，头痛身痛，鼻塞流涕，无汗',
    '扶正解表': '体虚感冒，恶寒发热，头痛，气短乏力，咳嗽',
    '清气分热': '气分热证，壮热，烦渴，大汗，脉洪大',
    '清营凉血': '营血热证，身热夜甚，心烦不寐，吐血衄血，斑疹',
    '清虚热': '虚热证，骨蒸潮热，盗汗，五心烦热',
    '清脏腑热': '脏腑热证，心烦失眠，口舌生疮，胃热呕吐，肝火目赤',
    '清热解毒': '热毒证，咽喉肿痛，痈肿疮毒，热毒泻痢',
    '清热祛暑': '暑热证，身热，心烦，口渴，小便短赤',
    '温中祛寒': '脾胃虚寒，脘腹冷痛，呕吐泄泻，食欲不振',
    '回阳救逆': '阳气衰微，四肢厥冷，神衰欲寐，脉微欲绝',
    '温经散寒': '寒凝经脉，手足厥冷，肢体冷痛，冻疮',
    '温通心阳': '心阳不振，心悸气短，胸闷胸痛，畏寒肢冷',
    '温中缓急': '中焦虚寒，脘腹拘急疼痛，喜温喜按',
    '燥湿化痰': '湿痰证，咳嗽痰多，胸闷，恶心',
    '清热化痰': '热痰证，咳嗽痰黄，胸闷，发热',
    '润燥化痰': '燥痰证，干咳少痰，咽干',
    '温化寒痰': '寒痰证，咳嗽痰白清稀，畏寒',
    '化痰熄风': '风痰证，眩晕，呕吐痰涎',
    '消食导滞': '食积证，脘腹胀满，嗳腐吞酸，食欲不振',
    '健脾消食': '脾虚食积，食少难消，脘腹胀满，神疲乏力',
    '和解少阳': '少阳证，寒热往来，胸胁苦满，心烦喜呕',
    '调和肝脾': '肝脾不和，胸胁胀痛，腹痛泄泻，月经不调',
    '调和肠胃': '肠胃不和，心下痞满，恶心呕吐，肠鸣泄泻',
    '燥湿和胃': '湿阻中焦，脘腹胀满，食欲不振，恶心呕吐，肢体困重',
    '利水渗湿': '水湿内停，水肿，小便不利，泄泻',
    '温化寒湿': '寒湿证，畏寒肢冷，脘腹冷痛，大便溏泄',
    '祛风胜湿': '风湿痹证，关节疼痛，肢体沉重，屈伸不利',
    '固表止汗': '表虚自汗，盗汗',
    '敛肺止咳': '久咳肺虚，咳嗽气喘',
    '涩肠固脱': '久泻久痢，滑脱不禁',
    '涩精止遗': '遗精滑精，遗尿尿频',
    '固崩止带': '崩漏，带下',
    '重镇安神': '心神不安，惊悸失眠，烦躁易怒',
    '补养安神': '心脾两虚，心悸失眠，健忘多梦',
    '凉开': '热闭证，高热神昏，谵语',
    '温开': '寒闭证，突然昏倒，牙关紧闭',
    '寒下': '实热积滞，大便秘结，腹痛拒按，发热',
    '温下': '寒积便秘，腹痛，手足不温',
    '润下': '肠燥便秘，大便干结',
    '攻补兼施': '里实正虚，便秘，神疲乏力',
    '逐水': '水饮壅盛，水肿，胸腹积水',
    '祛暑解表': '暑湿感冒，身热，恶寒，头重，胸闷',
    '清暑利湿': '暑湿证，身热烦渴，小便不利',
    '清心解暑': '暑热证，身热，心烦，口渴',
    '祛暑清肝': '暑热扰肝，心烦，目赤',
    '清暑益气': '暑热气津两伤，身热汗多，口渴，神疲',
    '行气': '气滞证，胸胁胀痛，脘腹胀满，嗳气',
    '降气': '气逆证，呕吐，呃逆，喘咳',
    '活血化瘀': '瘀血证，痛有定处，刺痛，癥瘕积聚，跌打损伤',
    '止血': '出血证，吐血，衄血，便血，崩漏',
    '疏散外风': '外风证，头痛，眩晕，风疹，风湿痹痛',
    '平息内风': '内风证，眩晕，震颤，抽搐，中风',
    '轻宣润燥': '外感燥证，咽干鼻燥，干咳无痰',
    '滋阴润燥': '内燥证，口干咽燥，干咳少痰，大便干结',
    '疏肝': '肝郁气滞，胸胁胀痛',
    '解郁': '肝郁气滞，情志不舒',
    '化痰': '痰证，咳嗽痰多',
    '平喘': '喘证，气喘',
    '降逆': '气逆证，呕吐，呃逆',
    '止呕': '呕吐',
    '止泻': '泄泻，腹泻',
    '通便': '便秘',
    '排石': '石淋，尿路结石',
    '通淋': '淋证，小便涩痛',
    '调经': '月经不调，痛经',
    '安胎': '胎动不安，妊娠诸证',
    '明目': '目赤肿痛，视物不清',
    '开窍': '窍闭神昏',
    '定痫': '癫痫',
}


def get_indications_from_name(name: str) -> list[str]:
    """从方剂名提取主治"""
    found = []
    for keyword, indications in NAME_KEYWORD_INDICATIONS.items():
        if keyword in name:
            found.append(indications)
    return found


def generate_indications_from_herbs(composition_indices, herb_data, db_id_to_model_idx, symptom_list):
    """从组成药材的功效生成主治文本"""
    if not composition_indices:
        return ""
    
    # 收集所有药材的功效和主治关键词
    all_terms = []
    for hi in composition_indices:
        # hi is model_idx, find herb by model_idx
        for hid, hinfo in herb_data.items():
            midx = db_id_to_model_idx.get(hid)
            if midx and midx - 1 == hi:
                text = f"{hinfo.get('efficacy', '')} {hinfo.get('indications', '')}"
                all_terms.append(text)
                break
    
    if not all_terms:
        return ""
    
    # 统计症状出现频率
    from collections import Counter
    symptom_counts = Counter()
    for s_name in symptom_list:
        for text in all_terms:
            if s_name in text:
                symptom_counts[s_name] += 1
    
    # 选出高频症状
    threshold = max(2, len(composition_indices) // 5)
    top_symptoms = [s for s, c in symptom_counts.most_common() if c >= threshold]
    
    if not top_symptoms:
        return ""
    
    # 也收集功效关键词
    efficacy_words = []
    for text in all_terms:
        parts = re.split(r'[，,。.；;]', text)
        for p in parts:
            p = p.strip()
            if 4 <= len(p) <= 30:
                efficacy_words.append(p)
    
    efficacy_set = list(dict.fromkeys(efficacy_words))[:5]
    
    result_parts = []
    if top_symptoms:
        result_parts.append('、'.join(top_symptoms[:6]))
    if efficacy_set:
        result_parts.append('。'.join(efficacy_set[:3]))
    
    return '，'.join(result_parts) if result_parts else ""


def main():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    
    # 加载全部方剂
    all_formulas = conn.execute("SELECT id, name, category, composition, indications, functions FROM formulas").fetchall()
    
    # 加载药材数据（用于组分推导）
    all_herbs = conn.execute("SELECT id, name, category, efficacy, indications FROM herbs").fetchall()
    herb_dict = {h['id']: dict(h) for h in all_herbs}
    
    # 加载全部症状
    from tcm_recommender.extended_config import EXTENDED_SYMPTOM_LIST as symptom_list
    
    # 找出缺主治的方剂
    missing = [f for f in all_formulas if not f['indications'] or len(f['indications'].strip()) < 5]
    logger.info(f'缺主治的方剂: {len(missing)}/{len(all_formulas)}')
    
    # 统计更新情况
    updated_count = 0
    skipped_no_info = 0
    
    # 预处理：从药材名→db_id 映射
    herb_name_to_db_id = {h['name']: h['id'] for h in all_herbs}
    # 用于组分解析的药材名→model_idx 映射
    from tcm_recommender.data_pipeline import TCMDataPipeline
    pipeline = TCMDataPipeline(str(DB_PATH))
    pipeline.load_all()
    
    updates = []
    
    for formula in missing:
        fid = formula['id']
        name = formula['name'] or ''
        category = formula['category'] or ''
        composition = formula['composition'] or ''
        
        generated_indications = ""
        
        # 策略A：有组成的，用组分推导
        if composition and len(composition) > 5:
            herb_indices = pipeline._parse_composition_indices(composition)
            if len(herb_indices) >= 2:
                gen = generate_indications_from_herbs(
                    herb_indices, herb_dict, pipeline.herb_db_id_to_model_idx, symptom_list
                )
                if gen:
                    generated_indications = gen
        
        # 策略B：从分类推断
        cat_hints = []
        if category:
            # 精确匹配
            if category in CATEGORY_TO_INDICATIONS:
                cat_hints.append(CATEGORY_TO_INDICATIONS[category])
            else:
                # 子串匹配
                for cat_key, cat_val in CATEGORY_TO_INDICATIONS.items():
                    if category in cat_key or cat_key in category:
                        cat_hints.append(cat_val)
                        break
                if not cat_hints:
                    for fallback_key, fallback_val in CATEGORY_FALLBACK.items():
                        if fallback_key in category or category in fallback_key:
                            cat_hints.append(fallback_val)
                            break
        
        # 策略C：从方剂名提取
        name_hints = get_indications_from_name(name)
        
        # 合并所有来源（名称提示优先于分类提示，因为更具体）
        all_hints = []
        if name_hints:
            all_hints.extend(name_hints)
        if generated_indications:
            all_hints.append(generated_indications)
        all_hints.extend(cat_hints)
        
        # 合并去重
        if all_hints:
            # 取前3个最有代表性的
            combined = '; '.join(dict.fromkeys(all_hints))[:200]
            updates.append((combined, fid))
            updated_count += 1
        else:
            skipped_no_info += 1
    
    # 执行更新
    logger.info(f'\n更新统计:')
    logger.info(f'  成功生成主治: {updated_count}')
    logger.info(f'  无法生成(无任何信息): {skipped_no_info}')
    
    if updated_count > 0:
        conn.executemany(
            "UPDATE formulas SET indications = ? WHERE id = ?",
            updates
        )
        conn.commit()
        logger.info(f'  已写入数据库: {updated_count} 条')
    
    # 验证
    still_missing = conn.execute(
        "SELECT COUNT(*) FROM formulas WHERE indications IS NULL OR trim(indications) = '' OR length(indications) < 5"
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM formulas").fetchone()[0]
    logger.info(f'\n最终状态:')
    logger.info(f'  总方剂: {total}')
    logger.info(f'  仍缺主治: {still_missing} ({100*still_missing/total:.1f}%)')
    logger.info(f'  覆盖方剂: {total - still_missing} ({(total-still_missing)*100/total:.1f}%)')
    
    # 展示一些样例
    if updated_count > 0:
        logger.info(f'\n生成样例:')
        samples = conn.execute(
            "SELECT id, name, category, indications FROM formulas WHERE id IN ({}) LIMIT 10".format(
                ','.join(str(u[1]) for u in updates[:10])
            )
        ).fetchall()
        for s in samples:
            ind = (s['indications'] or '')[:80]
            logger.info(f'  {s["id"]:>4}. {s["name"]:<22} → {ind}')
    
    pipeline.close()
    conn.close()


if __name__ == '__main__':
    main()
