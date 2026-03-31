# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0

"""
中医药推荐系统 - 数据加载模块

从数据库加载药材、方剂、症状数据，构建训练数据集
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class TCMDataLoader:
    """中医药数据加载器"""
    
    def __init__(self, db_path: str):
        """
        初始化数据加载器
        
        Args:
            db_path: 数据库路径
        """
        self.db_path = db_path
        self.conn = None
        self._connect()
        
        # 数据存储
        self.herbs = {}  # 药材数据
        self.formulas = {}  # 方剂数据
        self.symptoms = {}  # 症状数据
        self.symptom_herb_map = defaultdict(list)  # 症状→药材映射
        self.symptom_formula_map = defaultdict(list)  # 症状→方剂映射
        self.herb_symptom_map = defaultdict(list)  # 药材→症状映射
        
        # 配伍禁忌
        self.incompatible_herbs = {}  # 不相容药材对
        
    def _connect(self):
        """连接数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"数据库连接成功：{self.db_path}")
        except Exception as e:
            logger.error(f"数据库连接失败：{e}")
            raise
    
    def load_all_data(self):
        """加载所有数据"""
        logger.info("开始加载数据...")
        
        self._load_herbs()
        self._load_formulas()
        self._load_symptoms()
        self._load_incompatible_herbs()
        self._build_mappings()
        
        logger.info(f"数据加载完成:")
        logger.info(f"  - 药材：{len(self.herbs)}")
        logger.info(f"  - 方剂：{len(self.formulas)}")
        logger.info(f"  - 症状：{len(self.symptoms)}")
        logger.info(f"  - 配伍禁忌：{len(self.incompatible_herbs)}")
    
    def _load_herbs(self):
        """加载药材数据"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, name, category, nature, flavor, meridian, 
                   efficacy, indications, usage, contraindications
            FROM herbs
        """)
        
        for row in cursor.fetchall():
            self.herbs[row['id']] = {
                'id': row['id'],
                'name': row['name'],
                'category': row['category'],
                'nature': row['nature'],
                'flavor': row['flavor'],
                'meridian': row['meridian'],
                'efficacy': row['efficacy'],
                'indications': row['indications'],
                'usage': row['usage'],
                'contraindications': row['contraindications'],
                'is_toxic': self._check_toxic(row['contraindications'])
            }
    
    def _load_formulas(self):
        """加载方剂数据"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, name, category, source_book, composition, 
                   dosage, functions, indications, usage, modification
            FROM formulas
        """)
        
        for row in cursor.fetchall():
            self.formulas[row['id']] = {
                'id': row['id'],
                'name': row['name'],
                'category': row['category'],
                'source_book': row['source_book'],
                'composition': row['composition'],
                'dosage': row['dosage'],
                'functions': row['functions'],
                'indications': row['indications'],
                'usage': row['usage'],
                'modification': row['modification']
            }
    
    def _load_symptoms(self):
        """加载症状数据（从内置列表）"""
        # 常见症状列表
        common_symptoms = [
            '乏力', '疲倦', '食欲不振', '失眠', '头痛', '咳嗽',
            '发热', '腹痛', '腹泻', '便秘', '心悸', '头晕',
            '恶心', '呕吐', '胃痛', '腹胀', '腰膝酸软', '耳鸣',
            '健忘', '气短', '畏寒', '手足心热', '口干', '自汗',
            '盗汗', '胁痛', '关节痛', '皮肤瘙痒', '月经不调',
            '痛经', '白带异常'
        ]
        
        for i, symptom in enumerate(common_symptoms, 1):
            self.symptoms[i] = {
                'id': i,
                'name': symptom
            }
    
    def _load_incompatible_herbs(self):
        """加载配伍禁忌（十八反、十九畏）"""
        # 十八反
        self.incompatible_herbs['十八反'] = [
            ('甘草', '大戟'), ('甘草', '芫花'), ('甘草', '甘遂'),
            ('乌头', '半夏'), ('乌头', '瓜蒌'), ('乌头', '贝母'),
            ('藜芦', '人参'), ('藜芦', '丹参'), ('藜芦', '沙参'),
            ('藜芦', '玄参'), ('藜芦', '苦参'),
        ]
        
        # 十九畏
        self.incompatible_herbs['十九畏'] = [
            ('硫黄', '朴硝'), ('水银', '砒霜'), ('狼毒', '密陀僧'),
            ('巴豆', '牵牛'), ('丁香', '郁金'), ('牙硝', '三棱'),
            ('川乌', '草乌'), ('人参', '五灵脂'), ('肉桂', '赤石脂'),
        ]
        
        # 孕妇禁用
        self.incompatible_herbs['孕妇禁用'] = [
            '麝香', '三棱', '莪术', '水蛭', '虻虫', '巴豆', '牵牛',
            '大戟', '芫花', '甘遂', '商陆', '斑蝥', '穿山甲'
        ]
        
        logger.info(f"加载配伍禁忌：{sum(len(v) for v in self.incompatible_herbs.values())} 条")
    
    def _build_mappings(self):
        """构建症状 - 药材映射"""
        # 重新映射药材 ID 为连续索引
        self.herb_id_to_idx = {}
        self.idx_to_herb_id = {}
        
        for idx, (herb_id, herb) in enumerate(self.herbs.items(), 1):
            self.herb_id_to_idx[herb_id] = idx
            self.idx_to_herb_id[idx] = herb_id
        
        # 重新映射方剂 ID
        self.formula_id_to_idx = {}
        self.idx_to_formula_id = {}
        
        for idx, (formula_id, formula) in enumerate(self.formulas.items(), 1):
            self.formula_id_to_idx[formula_id] = idx
            self.idx_to_formula_id[idx] = formula_id
        
        # 从药材的 indications 字段提取症状映射
        for herb_id, herb in self.herbs.items():
            indications = herb['indications']
            herb_idx = self.herb_id_to_idx[herb_id]
            
            # 匹配症状
            for symp_id, symp in self.symptoms.items():
                if symp['name'] in indications:
                    self.symptom_herb_map[symp_id].append(herb_idx)
                    self.herb_symptom_map[herb_idx].append(symp_id)
        
        # 从方剂的 indications 字段提取症状映射
        for formula_id, formula in self.formulas.items():
            indications = formula['indications']
            formula_idx = self.formula_id_to_idx[formula_id]
            
            for symp_id, symp in self.symptoms.items():
                if symp['name'] in indications:
                    self.symptom_formula_map[symp_id].append(formula_idx)
    
    def _check_toxic(self, contraindications: str) -> bool:
        """检查药材是否有毒"""
        if not contraindications:
            return False
        
        toxic_keywords = ['毒', '有毒', '小毒', '大毒', '慎用', '禁用']
        return any(kw in contraindications for kw in toxic_keywords)
    
    def get_training_data(self) -> Tuple:
        """
        获取训练数据
        
        Returns:
            (symptoms_list, herbs_list, formulas_list) - 训练数据三元组
        """
        symptoms_list = []
        herbs_list = []
        formulas_list = []
        
        # 为每个症状生成训练样本
        for symp_id, herb_indices in self.symptom_herb_map.items():
            for herb_idx in herb_indices:
                symptoms_list.append([symp_id])
                herbs_list.append(herb_idx)
                formulas_list.append(0)  # 无方剂
        
        # 为每个症状 - 方剂对生成训练样本
        for symp_id, formula_indices in self.symptom_formula_map.items():
            for formula_idx in formula_indices:
                symptoms_list.append([symp_id])
                herbs_list.append(0)  # 无特定药材
                formulas_list.append(formula_idx)
        
        logger.info(f"生成训练样本：{len(symptoms_list)} 条")
        
        return symptoms_list, herbs_list, formulas_list
    
    def get_herb_names(self) -> Dict[int, str]:
        """获取药材 ID→名称映射"""
        return {self.idx_to_herb_id.get(h['id'], h['id']): h['name'] for h in self.herbs.values()}
    
    def get_formula_names(self) -> Dict[int, str]:
        """获取方剂 ID→名称映射"""
        return {self.idx_to_formula_id.get(f['id'], f['id']): f['name'] for f in self.formulas.values()}
    
    def get_symptom_names(self) -> Dict[int, str]:
        """获取症状 ID→名称映射"""
        return {s['id']: s['name'] for s in self.symptoms.values()}
    
    def close(self):
        """关闭数据库"""
        if self.conn:
            self.conn.close()


def load_data(db_path: str) -> TCMDataLoader:
    """
    加载数据的便捷函数
    
    Args:
        db_path: 数据库路径
    
    Returns:
        TCMDataLoader 实例
    """
    loader = TCMDataLoader(db_path)
    loader.load_all_data()
    return loader


if __name__ == '__main__':
    # 测试数据加载
    import os
    import sys
    
    # 添加项目路径
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    
    db_path = os.path.join(PROJECT_ROOT, 'data', 'tcm_knowledge.db')
    
    if os.path.exists(db_path):
        loader = load_data(db_path)
        
        print(f"\n数据加载完成:")
        print(f"  药材：{len(loader.herbs)}")
        print(f"  方剂：{len(loader.formulas)}")
        print(f"  症状：{len(loader.symptoms)}")
        
        # 显示部分数据
        print("\n前 5 个药材:")
        for i, (id_, herb) in enumerate(list(loader.herbs.items())[:5]):
            print(f"  {i+1}. {herb['name']} - {herb['efficacy'][:30]}...")
        
        print("\n配伍禁忌:")
        for category, pairs in loader.incompatible_herbs.items():
            print(f"  {category}: {len(pairs)} 条")
        
        loader.close()
    else:
        print(f"数据库不存在：{db_path}")
