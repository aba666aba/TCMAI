# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0
# See LICENSE file for details

"""
降级回答模块

当 AI 模型无法正常工作时，使用基于规则的中医药知识提供回答
"""

import logging
import re
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# 中医药知识库（内置，不依赖外部数据库）
TCM_KNOWLEDGE = {
    'herbs': [
        {'name': '人参', 'alias': '人衔、鬼盖', 'efficacy': '大补元气，复脉固脱，补脾益肺，生津养血，安神益智', 'indications': '体虚欲脱，肢冷脉微，脾虚食少，肺虚喘咳', 'contraindications': '实证、热证而正气不虚者忌服'},
        {'name': '黄芪', 'alias': '黄耆', 'efficacy': '补气升阳，固表止汗，利水消肿，生津养血，托毒排脓', 'indications': '气虚乏力，食少便溏，中气下陷，表虚自汗', 'contraindications': '表实邪盛，气滞湿阻者慎用'},
        {'name': '白术', 'alias': '山蓟', 'efficacy': '健脾益气，燥湿利水，止汗，安胎', 'indications': '脾虚食少，腹胀泄泻，痰饮眩悸', 'contraindications': '阴虚燥渴者慎用'},
        {'name': '甘草', 'alias': '国老', 'efficacy': '补脾益气，清热解毒，祛痰止咳，缓急止痛，调和诸药', 'indications': '脾胃虚弱，倦怠乏力，咳嗽痰多', 'contraindications': '湿盛胀满者不宜'},
        {'name': '当归', 'alias': '干归', 'efficacy': '补血活血，调经止痛，润肠通便', 'indications': '血虚萎黄，月经不调，经闭痛经', 'contraindications': '湿盛中满，大便泄泻者忌服'},
        {'name': '熟地黄', 'alias': '熟地', 'efficacy': '补血滋阴，益精填髓', 'indications': '血虚萎黄，心悸怔忡，月经不调', 'contraindications': '气滞痰多者慎用'},
        {'name': '茯苓', 'alias': '茯菟', 'efficacy': '利水渗湿，健脾宁心', 'indications': '水肿尿少，痰饮眩悸，脾虚食少', 'contraindications': '阴虚而无湿热者慎用'},
        {'name': '陈皮', 'alias': '橘皮', 'efficacy': '理气健脾，燥湿化痰', 'indications': '脘腹胀满，食少吐泻，咳嗽痰多', 'contraindications': '吐血证慎用'},
        {'name': '柴胡', 'alias': '地熏', 'efficacy': '疏散退热，疏肝解郁，升举阳气', 'indications': '感冒发热，寒热往来，胸胁胀痛', 'contraindications': '真阴亏损者忌用'},
        {'name': '黄芩', 'alias': '腐肠', 'efficacy': '清热燥湿，泻火解毒，止血，安胎', 'indications': '湿温暑湿，肺热咳嗽，高热烦渴', 'contraindications': '脾胃虚寒者慎用'},
        {'name': '黄连', 'alias': '味连', 'efficacy': '清热燥湿，泻火解毒', 'indications': '湿热痞满，呕吐吞酸，泻痢黄疸', 'contraindications': '脾胃虚寒者慎用'},
        {'name': '生姜', 'alias': '姜', 'efficacy': '解表散寒，温中止呕，化痰止咳', 'indications': '风寒感冒，胃寒呕吐，寒痰咳嗽', 'contraindications': '阴虚内热者慎用'},
        {'name': '大枣', 'alias': '红枣', 'efficacy': '补中益气，养血安神', 'indications': '脾虚食少，乏力便溏，妇人脏躁', 'contraindications': '湿盛脘腹胀满者慎用'},
        {'name': '桂枝', 'alias': '柳桂', 'efficacy': '发汗解肌，温通经脉，助阳化气', 'indications': '风寒感冒，寒凝血滞，痰饮', 'contraindications': '温热病阴虚火旺者慎用'},
        {'name': '白芍', 'alias': '白芍药', 'efficacy': '养血调经，敛阴止汗，柔肝止痛，平抑肝阳', 'indications': '血虚萎黄，月经不调，自汗盗汗', 'contraindications': '阳衰虚寒者不宜'},
    ],
    'formulas': [
        {'name': '四君子汤', 'source': '《太平惠民和剂局方》', 'composition': '人参、白术、茯苓、甘草', 'efficacy': '益气健脾', 'indications': '脾胃气虚证，面色萎白，气短乏力，食少便溏'},
        {'name': '四物汤', 'source': '《太平惠民和剂局方》', 'composition': '熟地黄、当归、白芍、川芎', 'efficacy': '补血调血', 'indications': '营血虚滞证，头晕目眩，心悸失眠，月经不调'},
        {'name': '八珍汤', 'source': '《正体类要》', 'composition': '四君子汤 + 四物汤', 'efficacy': '益气补血', 'indications': '气血两虚证，面色苍白，头晕目眩，心悸气短'},
        {'name': '补中益气汤', 'source': '《脾胃论》', 'composition': '黄芪、人参、白术、甘草、当归、陈皮、升麻、柴胡', 'efficacy': '补中益气，升阳举陷', 'indications': '脾虚气陷证，气虚发热，乏力懒言'},
        {'name': '归脾汤', 'source': '《济生方》', 'composition': '黄芪、人参、白术、甘草、当归、龙眼肉、酸枣仁、远志、木香、茯苓', 'efficacy': '益气补血，健脾养心', 'indications': '心脾气血两虚证，失眠健忘，心悸怔忡'},
        {'name': '六味地黄丸', 'source': '《小儿药证直诀》', 'composition': '熟地黄、山茱萸、山药、泽泻、牡丹皮、茯苓', 'efficacy': '滋阴补肾', 'indications': '肾阴虚证，腰膝酸软，头晕目眩，盗汗遗精'},
        {'name': '逍遥散', 'source': '《太平惠民和剂局方》', 'composition': '柴胡、当归、白芍、白术、茯苓、甘草、薄荷、生姜', 'efficacy': '疏肝解郁，养血健脾', 'indications': '肝郁血虚脾弱证，两胁作痛，神疲食少'},
        {'name': '小柴胡汤', 'source': '《伤寒论》', 'composition': '柴胡、黄芩、人参、甘草、半夏、生姜、大枣', 'efficacy': '和解少阳', 'indications': '伤寒少阳证，寒热往来，胸胁苦满，心烦喜呕'},
        {'name': '桂枝汤', 'source': '《伤寒论》', 'composition': '桂枝、白芍、生姜、大枣、甘草', 'efficacy': '解肌发表，调和营卫', 'indications': '外感风寒表虚证，头痛发热，汗出恶风'},
        {'name': '银翘散', 'source': '《温病条辨》', 'composition': '金银花、连翘、薄荷、牛蒡子、荆芥、淡豆豉、桔梗、甘草', 'efficacy': '辛凉透表，清热解毒', 'indications': '温病初起，发热无汗，头痛口渴'},
        {'name': '藿香正气散', 'source': '《太平惠民和剂局方》', 'composition': '藿香、紫苏、白芷、桔梗、白术、厚朴、半夏曲、大腹皮、茯苓、陈皮、甘草', 'efficacy': '解表化湿，理气和中', 'indications': '外感风寒，内伤湿滞，脘腹疼痛，呕吐泄泻'},
        {'name': '龙胆泻肝汤', 'source': '《医方集解》', 'composition': '龙胆草、黄芩、栀子、泽泻、木通、车前子、当归、生地黄、柴胡、甘草', 'efficacy': '清泻肝胆实火，清利肝经湿热', 'indications': '肝胆实火上炎证，头痛目赤，胁痛口苦'},
        {'name': '温胆汤', 'source': '《三因极一病证方论》', 'composition': '半夏、竹茹、枳实、陈皮、甘草、茯苓', 'efficacy': '理气化痰，清胆和胃', 'indications': '胆胃不和，痰热内扰证，虚烦不眠，呕吐呃逆'},
        {'name': '生脉散', 'source': '《医学启源》', 'composition': '人参、麦冬、五味子', 'efficacy': '益气生津，敛阴止汗', 'indications': '气阴两虚证，汗多神疲，气短懒言，咽干口渴'},
    ],
    'symptoms': {
        '乏力': {'category': '全身症状', 'description': '身体疲倦无力，精神不振', 'syndromes': '脾胃虚弱，气血两虚，肾阳虚', 'recommendation': '建议补气健脾，可考虑四君子汤、八珍汤等'},
        '食欲不振': {'category': '消化系统', 'description': '不想进食，食量减少', 'syndromes': '脾胃虚弱，肝郁气滞', 'recommendation': '建议健脾开胃，可考虑四君子汤、逍遥散等'},
        '失眠': {'category': '精神症状', 'description': '难以入睡或睡而易醒', 'syndromes': '心脾两虚，肝郁化火，阴虚火旺', 'recommendation': '建议养心安神，可考虑归脾汤、温胆汤等'},
        '头痛': {'category': '头部症状', 'description': '头部疼痛', 'syndromes': '肝阳上亢，气血亏虚，痰浊上扰', 'recommendation': '需辨证施治，可考虑逍遥散、六味地黄丸等'},
        '咳嗽': {'category': '呼吸系统', 'description': '肺气上逆，发出咳声', 'syndromes': '风寒束肺，风热犯肺，痰湿阻肺', 'recommendation': '需辨证施治，可考虑小柴胡汤、银翘散等'},
        '腹痛': {'category': '消化系统', 'description': '腹部疼痛', 'syndromes': '寒邪内阻，湿热壅滞，气滞血瘀', 'recommendation': '需辨证施治，可考虑桂枝汤、藿香正气散等'},
        '腹泻': {'category': '消化系统', 'description': '大便次数增多，粪质稀薄', 'syndromes': '脾虚湿盛，肾阳虚衰', 'recommendation': '建议健脾祛湿，可考虑藿香正气散、四君子汤等'},
        '便秘': {'category': '消化系统', 'description': '大便干结，排便困难', 'syndromes': '热结肠道，津亏血少，气虚传导无力', 'recommendation': '建议润肠通便，可考虑温胆汤等'},
        '心悸': {'category': '心血管系统', 'description': '心中悸动，惊惕不安', 'syndromes': '心气虚，心血不足，心阳不振', 'recommendation': '建议益气养血，可考虑八珍汤、归脾汤等'},
        '头晕': {'category': '头部症状', 'description': '头目眩晕，视物旋转', 'syndromes': '肝阳上亢，气血亏虚，肾精不足', 'recommendation': '需辨证施治，可考虑六味地黄丸、八珍汤等'},
        '发热': {'category': '全身症状', 'description': '体温升高或自觉发热', 'syndromes': '外感风寒，外感风热，阴虚发热', 'recommendation': '需辨证施治，可考虑银翘散、小柴胡汤等'},
        '恶寒': {'category': '全身症状', 'description': '自觉怕冷，得温不减', 'syndromes': '外感风寒，阳虚', 'recommendation': '建议解表散寒，可考虑桂枝汤、麻黄汤等'},
        '胁痛': {'category': '消化系统', 'description': '胁肋部疼痛', 'syndromes': '肝郁气滞，瘀血阻络，肝胆湿热', 'recommendation': '建议疏肝理气，可考虑逍遥散、龙胆泻肝汤等'},
        '自汗': {'category': '全身症状', 'description': '白天不活动或轻微活动即出汗', 'syndromes': '气虚，阳虚', 'recommendation': '建议益气固表，可考虑补中益气汤等'},
        '恶心呕吐': {'category': '消化系统', 'description': '胃气上逆，欲吐或吐出物', 'syndromes': '胃寒，胃热，痰饮，食滞', 'recommendation': '建议和胃降逆，可考虑小柴胡汤、温胆汤等'},
    }
}

# 常见证型及治法
SYNDROMES = {
    '脾胃虚弱': {'description': '脾胃运化功能减弱', 'symptoms': '乏力，食欲不振，腹泻，腹胀', 'treatment': '健脾益气', 'formula': '四君子汤'},
    '气血两虚': {'description': '气血不足，脏腑功能减退', 'symptoms': '乏力，心悸，失眠，面色萎黄', 'treatment': '益气补血', 'formula': '八珍汤'},
    '肝郁气滞': {'description': '肝气郁结，气机不畅', 'symptoms': '胁痛，失眠，情绪抑郁', 'treatment': '疏肝理气', 'formula': '逍遥散'},
    '肾阳虚': {'description': '肾阳不足，温煦功能减退', 'symptoms': '畏寒肢冷，腰膝酸软，夜尿多', 'treatment': '温补肾阳', 'formula': '六味地黄丸'},
    '阴虚火旺': {'description': '阴液亏虚，虚火内扰', 'symptoms': '潮热盗汗，五心烦热，失眠', 'treatment': '滋阴降火', 'formula': '六味地黄丸'},
    '风寒感冒': {'description': '风寒之邪侵袭肌表', 'symptoms': '恶寒重，发热轻，无汗，头痛', 'treatment': '辛温解表', 'formula': '桂枝汤'},
    '风热感冒': {'description': '风热之邪侵袭肺卫', 'symptoms': '发热重，微恶风，咽喉肿痛', 'treatment': '辛凉解表', 'formula': '银翘散'},
}


def detect_symptoms(query: str) -> List[str]:
    """
    从用户问题中检测症状

    Args:
        query: 用户输入

    Returns:
        检测到的症状列表
    """
    # 注意：症状关键词列表需与前端 src/web/templates/static/js/main.js 中的
    # extractSymptoms() 方法保持同步。后端检测用于实际分析，前端提取用于 UI 展示。
    detected = []
    query_lower = query.lower()

    # 症状关键词映射
    symptom_keywords = {
        '乏力': ['乏力', '没力气', '疲倦', '无力', '累', '疲劳', '虚弱'],
        '食欲不振': ['食欲不振', '不想吃饭', '没胃口', '吃不下', '纳差'],
        '失眠': ['失眠', '睡不着', '入睡困难', '易醒', '睡眠不好'],
        '头痛': ['头痛', '头疼', '头昏'],
        '咳嗽': ['咳嗽', '咳痰'],
        '腹痛': ['腹痛', '肚子疼', '肚子痛', '胃痛', '胃疼'],
        '腹泻': ['腹泻', '拉肚子', '大便稀'],
        '便秘': ['便秘', '大便干', '排便困难'],
        '心悸': ['心悸', '心慌', '心跳快'],
        '头晕': ['头晕', '眩晕', '头昏'],
        '发热': ['发热', '发烧', '体温高'],
        '恶寒': ['恶寒', '怕冷', '畏寒'],
        '胁痛': ['胁痛', '胁肋痛', '两肋痛'],
        '自汗': ['自汗', '容易出汗', '盗汗'],
        '恶心呕吐': ['恶心', '呕吐', '想吐'],
    }

    for symptom, keywords in symptom_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                if symptom not in detected:
                    detected.append(symptom)
                break

    return detected


def search_herbs(keyword: str) -> List[Dict]:
    """
    搜索药材

    Args:
        keyword: 关键词

    Returns:
        药材列表
    """
    results = []
    for herb in TCM_KNOWLEDGE['herbs']:
        if (keyword in herb['name'] or
            keyword in herb.get('alias', '') or
            keyword in herb.get('efficacy', '')):
            results.append(herb)
    return results[:5]  # 最多返回 5 条


def search_formulas(keyword: str) -> List[Dict]:
    """
    搜索方剂

    Args:
        keyword: 关键词

    Returns:
        方剂列表
    """
    results = []
    for formula in TCM_KNOWLEDGE['formulas']:
        if (keyword in formula['name'] or
            keyword in formula.get('efficacy', '') or
            keyword in formula.get('indications', '')):
            results.append(formula)
    return results[:5]  # 最多返回 5 条


def generate_fallback_answer(query: str, kb_result: Optional[Dict] = None) -> Dict[str, Any]:
    """
    生成降级回答（基于规则的中医药知识回答）

    Args:
        query: 用户问题
        kb_result: 知识库查询结果（可选）

    Returns:
        回答字典，包含 answer, knowledge, disclaimer
    """
    logger.info(f"使用降级方案生成回答：{query[:50]}...")

    # 检测症状
    symptoms = detect_symptoms(query)

    # 构建回答
    answer_parts = []

    # 1. 症状分析
    if symptoms:
        answer_parts.append("【症状分析】")
        for symptom in symptoms:
            if symptom in TCM_KNOWLEDGE['symptoms']:
                info = TCM_KNOWLEDGE['symptoms'][symptom]
                answer_parts.append(f"• {symptom}：{info['description']}")
                answer_parts.append(f"  可能证型：{info['syndromes']}")
                answer_parts.append(f"  建议：{info['recommendation']}")
    else:
        answer_parts.append("【分析】")
        answer_parts.append("根据您的描述，我来提供相关的中医药知识参考。")

    # 2. 推荐方剂
    if symptoms:
        answer_parts.append("\n【推荐方剂】")
        for symptom in symptoms:
            if symptom in TCM_KNOWLEDGE['symptoms']:
                info = TCM_KNOWLEDGE['symptoms'][symptom]
                answer_parts.append(f"针对{symptom}，可考虑：{info['recommendation']}")

    # 3. 搜索相关知识
    answer_parts.append("\n【相关知识】")

    # 从查询中提取关键词
    keywords = extract_keywords(query)
    for keyword in keywords[:3]:  # 最多处理 3 个关键词
        herbs = search_herbs(keyword)
        formulas = search_formulas(keyword)

        if herbs:
            answer_parts.append(f"• 相关药材（{keyword}）：")
            for herb in herbs[:2]:
                answer_parts.append(f"  - {herb['name']}：{herb['efficacy']}")

        if formulas:
            answer_parts.append(f"• 相关方剂（{keyword}）：")
            for formula in formulas[:2]:
                answer_parts.append(f"  - {formula['name']}：{formula['efficacy']}")
                answer_parts.append(f"    组成：{formula['composition']}")

    # 4. 通用建议
    answer_parts.append("\n【养生建议】")
    answer_parts.append("• 饮食调理：饮食宜清淡，避免生冷油腻")
    answer_parts.append("• 起居调摄：保持规律作息，避免熬夜")
    answer_parts.append("• 情志调畅：保持心情舒畅，避免过度思虑")

    # 合并回答
    answer = '\n'.join(answer_parts)

    # 免责声明
    disclaimer = "温馨提示：本系统仅提供中医药知识科普和参考信息，不构成医疗诊断或治疗建议。用户如有健康问题，请咨询专业医疗机构或执业医师。"

    return {
        'answer': answer,
        'knowledge': {
            'symptoms': symptoms,
            'source': 'fallback_mode'
        },
        'disclaimer': disclaimer,
        'mode': 'fallback'  # 标记为降级模式
    }


def extract_keywords(query: str) -> List[str]:
    """
    从查询中提取关键词

    Args:
        query: 用户输入

    Returns:
        关键词列表
    """
    # 中医药领域关键词
    tcm_keywords = [
        '气虚', '血虚', '阴虚', '阳虚', '气滞', '血瘀', '痰湿', '湿热',
        '脾胃', '肝肾', '心肺', '气血', '阴阳', '表里', '寒热',
        '补气', '补血', '滋阴', '壮阳', '健脾', '养胃', '疏肝', '理气',
        '感冒', '咳嗽', '发热', '头痛', '失眠', '腹泻', '便秘',
        '人参', '黄芪', '当归', '甘草', '茯苓', '白术', '陈皮', '柴胡',
        '四君子汤', '四物汤', '八珍汤', '六味地黄丸', '逍遥散', '小柴胡汤'
    ]

    keywords = []
    for keyword in tcm_keywords:
        if keyword in query:
            keywords.append(keyword)

    return keywords


def is_model_output_valid(output: str) -> bool:
    """
    检查模型输出是否有效

    Args:
        output: 模型输出

    Returns:
        是否有效
    """
    if not output or not output.strip():
        return False

    # 检查是否有大量重复字符
    if len(output) >= 10:
        # 检查连续重复字符
        if re.search(r'(.)\1{4,}', output):
            return False

        # 检查重复片段
        for length in range(2, 10):
            for i in range(len(output) - length * 2):
                segment = output[i:i + length]
                if segment * 3 in output:
                    return False

        # 检查唯一字符比例（检测重复字符过多的无效输出）
        unique_chars = set(output)
        unique_ratio = len(unique_chars) / len(output)
        if unique_ratio < 0.3:  # 唯一字符占比低于30%，判定为无效输出
            return False

    # 检查是否包含乱码字符
    if re.search(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', output):
        return False

    # 检查是否有意义的内容（至少有一些汉字）
    if not re.search(r'[\u4e00-\u9fff]', output):
        return False

    return True


def get_fallback_info(topic: str) -> Dict[str, Any]:
    """
    获取特定主题的降级信息

    Args:
        topic: 主题（药材名、方剂名、症状名等）

    Returns:
        相关信息字典
    """
    result = {'found': False, 'type': None, 'data': None}

    # 搜索药材
    for herb in TCM_KNOWLEDGE['herbs']:
        if topic in herb['name'] or topic in herb.get('alias', ''):
            result['found'] = True
            result['type'] = 'herb'
            result['data'] = herb
            return result

    # 搜索方剂
    for formula in TCM_KNOWLEDGE['formulas']:
        if topic in formula['name']:
            result['found'] = True
            result['type'] = 'formula'
            result['data'] = formula
            return result

    # 搜索症状
    if topic in TCM_KNOWLEDGE['symptoms']:
        result['found'] = True
        result['type'] = 'symptom'
        result['data'] = TCM_KNOWLEDGE['symptoms'][topic]
        return result

    # 搜索证型
    if topic in SYNDROMES:
        result['found'] = True
        result['type'] = 'syndrome'
        result['data'] = SYNDROMES[topic]
        return result

    return result
