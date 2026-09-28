# -*- coding: utf-8 -*-
"""
数据增强：补充症状-方剂关联，提升训练数据质量

策略：
  1. 别名扩展匹配：用 TCM_SYNONYM_MAP + SYMPTOM_ALIAS_MAP 扩展现有关联
  2. 方剂名称提取：从方剂名中提取症状关键词
  3. 组分推导：用方剂组成的药材功效反推主治
  4. 管线知识：用 SYMPTOM_TO_FORMULA_MAP 直接注入已知关联
  5. 功效+主治联合搜索：用 functions 和 indications 的组合文本扩大匹配
  6. 方剂分类推断：基于方剂分类（如"解表剂-辛温解表"）注入关联
  7. 证型定义注入：从 extended_config 的证型定义中提取推荐方剂
  8. 民间验方疾病映射：将 folk_remedies 的疾病名匹配到症状
"""

import sys, os, re, logging
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['TCMAI_ROOT'] = str(Path(__file__).resolve().parents[1])

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 加载智能搜索的映射知识
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from core.tcm_config import (
    TCM_SYNONYM_MAP, SYMPTOM_ALIAS_MAP, SYMPTOM_ALIAS_REVERSE_MAP,
    SYMPTOM_TO_FORMULA_MAP, SYMPTOM_LIST,
)
from tcm_recommender.data_pipeline import TCMDataPipeline
from tcm_recommender.extended_config import EXTENDED_SYMPTOM_LIST

DB_PATH = Path(__file__).resolve().parent / 'tcm_knowledge.db'
OUTPUT_NPZ = Path(__file__).resolve().parent / 'train_data.npz'

# 构建完整的同义词映射（症状名 → 所有可能的写法）
def build_synonym_expander():
    """构建症状名→别名列表的映射，用于扩展文本匹配"""
    expander = {}
    # 1. TCM_SYNONYM_MAP 中的同义词
    for key, syns in TCM_SYNONYM_MAP.items():
        expander[key] = list(syns)
    # 2. SYMPTOM_ALIAS_REVERSE_MAP（标准症状的口语变体）
    for std, aliases in SYMPTOM_ALIAS_REVERSE_MAP.items():
        if std not in expander:
            expander[std] = []
        expander[std].extend(aliases)
    # 3. 确保每个标准症状至少包含自己和可能的相关写法
    for s in EXTENDED_SYMPTOM_LIST:
        if s not in expander:
            expander[s] = []
    return expander

# 从方剂名提取症状关键词
def extract_symptoms_from_name(formula_name: str) -> list:
    """从方剂名称中提取可能的症状关键词"""
    if not formula_name:
        return []
    found = set()
    for s in EXTENDED_SYMPTOM_LIST:
        if s in formula_name:
            found.add(s)
    # 特殊模式：以"汤/丸/散/丹/膏/片/胶囊"结尾，前面的词可能是症状
    name_clean = re.sub(r'[汤丸散丹膏片胶囊颗粒口服液]*$', '', formula_name)
    for s in EXTENDED_SYMPTOM_LIST:
        if s in name_clean:
            found.add(s)
    # 常见方剂名模式：XX止痛/安神/化痰/止咳
    pattern_symptoms = {
        '止痛': ['疼痛', '痛'],
        '安神': ['失眠', '心悸'],
        '化痰': ['咳嗽', '痰'],
        '止咳': ['咳嗽'],
        '祛痰': ['咳嗽', '痰'],
        '清热': ['发热', '热证'],
        '解毒': ['热毒'],
        '活血': ['瘀血', '血瘀'],
        '凉血': ['血热'],
        '止血': ['出血'],
        '舒筋': ['关节痛', '风湿'],
        '活络': ['风湿痹痛'],
        '疏肝': ['肝郁气滞', '胁痛'],
        '理气': ['气滞', '腹胀'],
        '健脾': ['脾胃虚弱', '食欲不振'],
        '养胃': ['胃痛', '胃脘痛'],
        '润肠': ['便秘'],
        '通便': ['便秘'],
        '利水': ['水肿', '小便不利'],
        '消肿': ['水肿', '肿痛'],
        '退黄': ['湿热黄疸'],
        '疏风': ['风热感冒', '风寒感冒'],
        '散寒': ['畏寒', '风寒'],
        '排石': ['石淋'],
        '通淋': ['热淋', '小便涩痛'],
        '调经': ['月经不调'],
        '安胎': ['胎动不安'],
        '明|目': ['目赤肿痛', '视物模糊'],
    }
    for pattern, symptoms in pattern_symptoms.items():
        if pattern in formula_name:
            for s in symptoms:
                # 找标准症状的精确匹配
                for std in EXTENDED_SYMPTOM_LIST:
                    if s in std:
                        found.add(std)
    return list(found)


def main():
    logger.info('=' * 60)
    logger.info('数据增强：补充症状-方剂关联')
    logger.info('=' * 60)

    # 1. 加载数据管线
    pipeline = TCMDataPipeline(str(DB_PATH))
    pipeline.load_all()

    ns = pipeline.num_symptoms  # 200
    nf = pipeline.num_formulas  # 1095
    nh = pipeline.num_herbs     # 3010

    # 2. 构建别名扩展器
    expander = build_synonym_expander()

    # ============ 增强策略1：别名扩展匹配 ============
    logger.info('\n[策略1] 别名扩展匹配...')
    enhanced_sym_formula = np.zeros((ns, nf), dtype=np.float32)

    for fid, formula in pipeline.formulas.items():
        f_idx = pipeline.formula_db_id_to_model_idx.get(fid)
        if f_idx is None:
            continue
        text = f"{formula.get('functions', '')} {formula.get('indications', '')} {formula.get('name', '')}"
        if not text.strip():
            continue
        for si, s_name in enumerate(pipeline.symptoms):
            # 直接匹配
            if s_name in text:
                enhanced_sym_formula[si, f_idx - 1] = 1.0
                continue
            # 别名扩展匹配
            aliases = expander.get(s_name, [])
            for alias in aliases:
                if alias and alias in text:
                    enhanced_sym_formula[si, f_idx - 1] = 1.0
                    break

    new_assoc = int(enhanced_sym_formula.sum())
    old_assoc = int(pipeline.symptom_formula_matrix.sum())
    logger.info(f'  原有关联: {old_assoc}')
    logger.info(f'  扩展后:   {new_assoc} (+{new_assoc - old_assoc})')

    # ============ 增强策略2：方剂名称提取 ============
    logger.info('\n[策略2] 方剂名称症状提取...')
    name_matches = 0
    for fid, formula in pipeline.formulas.items():
        f_idx = pipeline.formula_db_id_to_model_idx.get(fid)
        if f_idx is None:
            continue
        name = formula.get('name', '')
        symptoms_from_name = extract_symptoms_from_name(name)
        for s in symptoms_from_name:
            for si, s_name in enumerate(pipeline.symptoms):
                if s_name == s or s_name in s:
                    if enhanced_sym_formula[si, f_idx - 1] == 0:
                        enhanced_sym_formula[si, f_idx - 1] = 0.8
                        name_matches += 1
                    break

    logger.info(f'  方剂名称新增关联: {name_matches}')
    new_assoc2 = int(enhanced_sym_formula.sum())
    logger.info(f'  累计关联: {new_assoc2}')

    # ============ 增强策略3：组分药材推导 ============
    logger.info('\n[策略3] 组分药材功效推导...')
    # 对没有主治/功效但有组成的方剂，用组成药材的功效推导
    herb_symptom_matrix = np.zeros((ns, nh), dtype=np.float32)
    for hid, herb in pipeline.herbs.items():
        h_idx = pipeline.herb_db_id_to_model_idx.get(hid)
        if h_idx is None:
            continue
        text = f"{herb.get('efficacy', '')} {herb.get('indications', '')}"
        for si, s_name in enumerate(pipeline.symptoms):
            if s_name in text:
                herb_symptom_matrix[si, h_idx - 1] = 1.0
            else:
                # 别名扩展
                aliases = expander.get(s_name, [])
                for alias in aliases:
                    if alias and alias in text:
                        herb_symptom_matrix[si, h_idx - 1] = 1.0
                        break

    herb_inferred = 0
    for fid, formula in pipeline.formulas.items():
        f_idx = pipeline.formula_db_id_to_model_idx.get(fid)
        if f_idx is None:
            continue
        # 只对没有症状关联的方剂做推导
        if enhanced_sym_formula[:, f_idx - 1].sum() > 0:
            continue
        comp = formula.get("composition", "")
        herb_indices = pipeline._parse_composition_indices(comp)
        if len(herb_indices) < 2:
            continue
        # 聚合所有组成药材的症状
        formula_symptom_scores = np.zeros(ns)
        for hi in herb_indices:
            if 0 <= hi < nh:
                formula_symptom_scores += herb_symptom_matrix[:, hi]
        # 取出现次数 >= 2 的症状（至少两种药材共同指向）
        threshold = max(2, int(len(herb_indices) * 0.2))
        for si in range(ns):
            if formula_symptom_scores[si] >= threshold:
                enhanced_sym_formula[si, f_idx - 1] = 0.7
                herb_inferred += 1

    logger.info(f'  组分推导新增关联: {herb_inferred}')
    new_assoc3 = int(enhanced_sym_formula.sum())
    logger.info(f'  累计关联: {new_assoc3}')

    # ============ 增强策略4：知识库直接注入 ============
    logger.info('\n[策略4] 知识库映射注入...')
    kb_injected = 0
    for symptom, formula_refs in SYMPTOM_TO_FORMULA_MAP.items():
        for si, s_name in enumerate(pipeline.symptoms):
            if s_name != symptom:
                continue
            for ref in formula_refs:
                fname = ref['formula']
                # 在数据库中找这个方剂
                for fid, formula in pipeline.formulas.items():
                    if formula['name'] == fname:
                        f_idx = pipeline.formula_db_id_to_model_idx.get(fid)
                        if f_idx and enhanced_sym_formula[si, f_idx - 1] == 0:
                            enhanced_sym_formula[si, f_idx - 1] = 0.9
                            kb_injected += 1
                        break
            break

    logger.info(f'  知识库注入新增关联: {kb_injected}')

    # ============ 增强策略5：功效+主治联合深度匹配 ============
    logger.info('\n[策略5] 功效+主治联合深度匹配...')
    # 对还没有任何关联的方剂，用 functions+indications+name 全文本再次搜索
    # 这次降低匹配门槛，只要症状子串出现在任意字段就建立关联
    deep_matched = 0
    for fid, formula in pipeline.formulas.items():
        f_idx = pipeline.formula_db_id_to_model_idx.get(fid)
        if f_idx is None:
            continue
        # 跳过已有大量关联的
        if enhanced_sym_formula[:, f_idx - 1].sum() >= 3:
            continue
        text = ' '.join(filter(None, [
            formula.get('functions', ''),
            formula.get('indications', ''),
            formula.get('name', ''),
            formula.get('composition', ''),
            formula.get('modification', ''),
            formula.get('source_book', ''),
        ]))
        if not text.strip():
            continue
        for si, s_name in enumerate(pipeline.symptoms):
            if s_name in text:
                if enhanced_sym_formula[si, f_idx - 1] == 0:
                    enhanced_sym_formula[si, f_idx - 1] = 0.6
                    deep_matched += 1

    logger.info(f'  深度匹配新增关联: {deep_matched}')
    new_assoc5 = int(enhanced_sym_formula.sum())
    logger.info(f'  累计关联: {new_assoc5}')

    # ============ 增强策略6：方剂分类推断 ============
    logger.info('\n[策略6] 方剂分类推断...')
    # 分类→症状映射
    CATEGORY_SYMPTOM_MAP = {
        '解表剂': ['感冒', '发热', '头痛', '恶寒'],
        '辛温解表': ['恶寒', '发热', '头痛', '身痛', '鼻塞', '无汗'],
        '辛凉解表': ['发热', '咽痛', '口渴', '咳嗽'],
        '扶正解表': ['感冒', '发热', '头痛', '气短', '乏力'],
        '清热剂': ['发热', '口干', '心烦', '咽痛'],
        '清热解毒': ['咽喉肿痛', '发热', '口干', '疮疡'],
        '清脏腑热': ['心烦', '失眠', '口疮', '目赤'],
        '清营凉血': ['发热', '心烦', '吐血', '斑疹'],
        '清虚热': ['潮热', '盗汗', '五心烦热'],
        '清气分热': ['壮热', '烦渴', '大汗'],
        '泻下剂': ['便秘', '腹痛', '腹胀'],
        '寒下': ['便秘', '腹痛', '发热'],
        '温下': ['便秘', '腹痛', '畏寒'],
        '润下': ['便秘', '大便干'],
        '和解剂': ['寒热往来', '胸胁苦满'],
        '和解少阳': ['寒热往来', '胸胁苦满', '心烦'],
        '调和肝脾': ['胁痛', '腹痛', '泄泻', '月经不调'],
        '温里剂': ['畏寒', '肢冷', '腹痛', '泄泻'],
        '温中祛寒': ['脘腹冷痛', '呕吐', '泄泻', '食欲不振'],
        '回阳救逆': ['四肢厥冷', '神衰', '脉微'],
        '补益剂': ['乏力', '气短', '面色萎黄'],
        '补气': ['乏力', '气短', '食欲不振', '面色苍白', '自汗'],
        '补血': ['面色萎黄', '头晕', '心悸', '失眠', '月经不调'],
        '气血双补': ['面色苍白', '头晕', '乏力', '气短', '心悸'],
        '补阴': ['潮热', '盗汗', '口干', '五心烦热'],
        '补阳': ['畏寒', '腰膝酸软', '乏力'],
        '安神剂': ['失眠', '心悸', '健忘', '烦躁'],
        '重镇安神': ['心悸', '失眠', '烦躁', '易怒'],
        '补养安神': ['心悸', '失眠', '健忘', '多梦'],
        '开窍剂': ['神昏', '谵语'],
        '固涩剂': ['自汗', '盗汗', '遗精', '带下', '久泻'],
        '理气剂': ['胸胁胀痛', '腹胀', '嗳气', '呕吐'],
        '理血剂': ['瘀血', '出血', '疼痛', '月经不调'],
        '活血化瘀': ['瘀血', '疼痛', '癥瘕', '跌打损伤'],
        '止血': ['吐血', '衄血', '便血', '崩漏'],
        '治风剂': ['头痛', '眩晕', '抽搐', '风湿'],
        '祛湿剂': ['水肿', '小便不利', '泄泻', '关节痛'],
        '祛痰剂': ['咳嗽', '痰多', '胸闷'],
        '消食剂': ['食积', '腹胀', '食欲不振'],
        '驱虫剂': ['虫积', '腹痛'],
        '治燥剂': ['咽干', '干咳', '口渴'],
        '祛暑剂': ['发热', '心烦', '口渴', '小便短赤'],
        '专科方': ['疼痛', '肿痛'],
        '临床经验方': ['疼痛', '肿胀'],
    }

    cat_matched = 0
    for fid, formula in pipeline.formulas.items():
        f_idx = pipeline.formula_db_id_to_model_idx.get(fid)
        if f_idx is None:
            continue
        category = formula.get('category', '') or ''
        if not category:
            continue

        # 收集该分类对应的症状
        target_symptoms = set()
        for cat_key, syms in CATEGORY_SYMPTOM_MAP.items():
            if cat_key in category or category in cat_key:
                for s in syms:
                    target_symptoms.add(s)

        if not target_symptoms:
            continue

        for s_name in target_symptoms:
            for si, ps_name in enumerate(pipeline.symptoms):
                if ps_name == s_name or s_name in ps_name:
                    if enhanced_sym_formula[si, f_idx - 1] == 0:
                        enhanced_sym_formula[si, f_idx - 1] = 0.65
                        cat_matched += 1
                    break

    logger.info(f'  分类推断新增关联: {cat_matched}')
    new_assoc6 = int(enhanced_sym_formula.sum())
    logger.info(f'  累计关联: {new_assoc6}')

    # ============ 增强策略7：证型定义注入 ============
    logger.info('\n[策略7] 证型定义注入...')
    syn_injected = 0
    from tcm_recommender.extended_config import EXTENDED_SYNDROME_DEFINITIONS
    for syn_name, syn_info in EXTENDED_SYNDROME_DEFINITIONS.items():
        # 证型本身的症状
        syn_symptoms = syn_info.get('symptoms', [])
        # 推荐的方剂
        syn_formulas = syn_info.get('recommend_formulas', [])
        syn_herbs = syn_info.get('recommend_herbs', [])

        if not syn_formulas:
            continue

        for fname in syn_formulas:
            for fid, formula in pipeline.formulas.items():
                if formula['name'] != fname:
                    continue
                f_idx = pipeline.formula_db_id_to_model_idx.get(fid)
                if f_idx is None:
                    continue
                for s_name in syn_symptoms:
                    for si, ps_name in enumerate(pipeline.symptoms):
                        if ps_name == s_name or s_name in ps_name:
                            if enhanced_sym_formula[si, f_idx - 1] == 0:
                                enhanced_sym_formula[si, f_idx - 1] = 0.85
                                syn_injected += 1
                            break
                break

    logger.info(f'  证型注入新增关联: {syn_injected}')
    new_assoc7 = int(enhanced_sym_formula.sum())
    logger.info(f'  累计关联: {new_assoc7}')

    # ============ 增强策略8：民间验方疾病→症状映射 ============
    logger.info('\n[策略8] 民间验方疾病映射...')
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    folk_diseases = conn.execute(
        "SELECT disease, remedy FROM folk_remedies WHERE disease IS NOT NULL"
    ).fetchall()
    conn.close()

    folk_matched = 0
    for disease, remedy in folk_diseases:
        disease = disease or ''
        remedy = remedy or ''
        text = f'{disease} {remedy}'
        for si, s_name in enumerate(pipeline.symptoms):
            if s_name in text:
                # 民间验方不直接映射到方剂，但可以给 herb_symptom_matrix 增加证据
                # 跳过，因为不直接对应 formulas
                pass

    # 但我们可以用民间验方的疾病名来补充药材-症状关联
    # 这对组分推导有帮助
    folk_herb_boost = 0
    for disease, remedy in folk_diseases:
        disease = disease or ''
        for si, s_name in enumerate(pipeline.symptoms):
            if s_name in disease or disease in s_name:
                # 增强 herb_symptom_matrix
                # 查找该验方中提到的药材
                for hid, herb in pipeline.herbs.items():
                    h_name = herb['name']
                    if h_name in (remedy or ''):
                        h_idx = pipeline.herb_db_id_to_model_idx.get(hid)
                        if h_idx and herb_symptom_matrix[si, h_idx - 1] == 0:
                            herb_symptom_matrix[si, h_idx - 1] = 0.5
                            folk_herb_boost += 1
                        break

    logger.info(f'  民间验方增强药材关联: {folk_herb_boost}')

    # ============ 再用增强后的 herb_symptom_matrix 重新做组分推导 ============
    logger.info('\n[再推导] 用增强后的药材矩阵重新组分推导...')
    re_inferred = 0
    for fid, formula in pipeline.formulas.items():
        f_idx = pipeline.formula_db_id_to_model_idx.get(fid)
        if f_idx is None:
            continue
        comp = formula.get("composition", "")
        herb_indices = pipeline._parse_composition_indices(comp)
        if len(herb_indices) < 2:
            continue
        scores = np.zeros(ns)
        for hi in herb_indices:
            if 0 <= hi < nh:
                scores += herb_symptom_matrix[:, hi]
        threshold = max(1, int(len(herb_indices) * 0.15))
        for si in range(ns):
            if scores[si] >= threshold and enhanced_sym_formula[si, f_idx - 1] == 0:
                enhanced_sym_formula[si, f_idx - 1] = 0.6
                re_inferred += 1

    logger.info(f'  再推导新增关联: {re_inferred}')

    # ============ 最终统计 ============
    final_assoc = int(enhanced_sym_formula.sum())
    formulas_with_symptoms = int((enhanced_sym_formula.sum(axis=0) > 0).sum())
    symptoms_with_formulas = int((enhanced_sym_formula.sum(axis=1) > 0).sum())
    logger.info(f'\n{"=" * 60}')
    logger.info(f'增强统计')
    logger.info(f'{"=" * 60}')
    logger.info(f'  关联总数:      {old_assoc:>5} → {final_assoc:>5} (+{final_assoc - old_assoc})')
    logger.info(f'  有症状关联的方剂: {int((pipeline.symptom_formula_matrix.sum(axis=0) > 0).sum()):>5} → {formulas_with_symptoms:>5}')
    logger.info(f'  有方剂关联的症状: {int((pipeline.symptom_formula_matrix.sum(axis=1) > 0).sum()):>5} → {symptoms_with_formulas:>5}')

    # ============ 生成增强版训练数据 ============
    logger.info(f'\n生成增强训练数据...')

    # 替换管线的关联矩阵
    old_matrix = pipeline.symptom_formula_matrix.copy()
    pipeline.symptom_formula_matrix = enhanced_sym_formula

    # 重新生成训练样本
    target_count = 150000
    data = pipeline.generate_samples(target_count)

    # 检查方剂标签分布
    fl = data['formula_labels']
    from collections import Counter
    fc = Counter(fl.tolist())
    unique_formulas = len([l for l in fc if l > 0])
    bg_count = fc.get(0, 0)
    logger.info(f'\n训练数据统计:')
    logger.info(f'  总样本:    {len(fl)}')
    logger.info(f'  背景标签:  {bg_count} ({100*bg_count/len(fl):.1f}%)')
    logger.info(f'  唯一方剂:  {unique_formulas}/{nf}')
    logger.info(f'  有效方剂:  {len([l for l in fc if l>0])} 种')

    # 保存
    pipeline.save(data, str(OUTPUT_NPZ))
    logger.info(f'\n已保存: {OUTPUT_NPZ}')

    # 恢复原矩阵
    pipeline.symptom_formula_matrix = old_matrix
    pipeline.close()

    logger.info('\n✅ 数据增强完成')


if __name__ == '__main__':
    main()
