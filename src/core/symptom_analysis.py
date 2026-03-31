# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0
# See LICENSE file for details

"""
症状分析模块

原创功能：症状组合分析、证型判断、禁忌校验
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SymptomInfo:
    """症状信息"""
    name: str
    standardized_name: str
    category: str  # 症状类别
    severity: int  # 严重程度 1-5


@dataclass
class SyndromeResult:
    """证型判断结果"""
    syndrome_name: str  # 证型名称
    confidence: float  # 置信度
    symptoms: List[str]  # 相关症状
    explanation: str  # 证型解释


class SymptomAnalyzer:
    """
    症状分析器
    
    基于中医辨证论治理论进行症状分析
    """
    
    def __init__(self):
        """初始化分析器"""
        # 症状标准化映射表（示例）
        self.symptom_map = {
            '肚子疼': '腹痛',
            '胃疼': '胃痛',
            '没力气': '乏力',
            '累': '乏力',
            '睡不着': '失眠',
            '不想吃饭': '食欲不振',
            '拉肚子': '腹泻',
            '大便干': '便秘',
        }
        
        # 症状类别
        self.symptom_categories = {
            '乏力': '全身症状',
            '食欲不振': '消化系统',
            '失眠': '精神症状',
            '头痛': '头部症状',
            '咳嗽': '呼吸系统',
            '腹痛': '消化系统',
            '腹泻': '消化系统',
            '便秘': '消化系统',
        }
        
        # 证型判断规则（简化示例）
        self.syndrome_rules = {
            '脾胃虚弱': {
                'symptoms': ['乏力', '食欲不振', '腹泻'],
                'explanation': '脾胃运化功能减弱，表现为消化功能低下'
            },
            '肝郁气滞': {
                'symptoms': ['头痛', '失眠', '乏力'],
                'explanation': '肝气郁结，气机不畅'
            },
            '气血两虚': {
                'symptoms': ['乏力', '失眠', '食欲不振'],
                'explanation': '气血不足，脏腑功能减退'
            }
        }
        
        # 禁忌规则
        self.contraindications = {
            '孕妇禁用': ['活血化瘀', '攻下'],
            '脾胃虚寒慎用': ['寒凉'],
        }
    
    def analyze(self, symptoms: List[str]) -> Dict:
        """
        分析症状
        
        Args:
            symptoms: 用户输入的症状列表
            
        Returns:
            分析结果
        """
        if not symptoms:
            return {
                'standardized': [],
                'syndromes': [],
                'contraindications': [],
                'suggestions': []
            }
        
        # 症状标准化
        standardized = self._standardize_symptoms(symptoms)
        
        # 证型判断
        syndromes = self._identify_syndromes(standardized)
        
        # 禁忌检查
        contraindications = self._check_contraindications(syndromes)
        
        # 生成建议
        suggestions = self._generate_suggestions(syndromes)
        
        return {
            'standardized': standardized,
            'syndromes': syndromes,
            'contraindications': contraindications,
            'suggestions': suggestions
        }
    
    def _standardize_symptoms(self, symptoms: List[str]) -> List[SymptomInfo]:
        """
        症状标准化
        
        将日常用语映射到中医专业术语
        
        Args:
            symptoms: 原始症状列表
            
        Returns:
            标准化后的症状列表
        """
        standardized = []
        
        for symptom in symptoms:
            # 查找标准化名称
            std_name = self.symptom_map.get(symptom, symptom)
            
            # 获取类别
            category = self.symptom_categories.get(std_name, '其他')
            
            symptom_info = SymptomInfo(
                name=symptom,
                standardized_name=std_name,
                category=category,
                severity=3  # 默认中等程度
            )
            standardized.append(symptom_info)
        
        logger.info(f"症状标准化完成：{[s.standardized_name for s in standardized]}")
        return standardized
    
    def _identify_syndromes(self, symptoms: List[SymptomInfo]) -> List[SyndromeResult]:
        """
        证型判断
        
        Args:
            symptoms: 标准化症状列表
            
        Returns:
            证型判断结果列表
        """
        symptom_names = [s.standardized_name for s in symptoms]
        results = []
        
        for syndrome_name, rule in self.syndrome_rules.items():
            # 计算匹配度
            matched = [s for s in rule['symptoms'] if s in symptom_names]
            confidence = len(matched) / len(rule['symptoms'])
            
            if confidence >= 0.5:  # 匹配度超过 50%
                result = SyndromeResult(
                    syndrome_name=syndrome_name,
                    confidence=confidence,
                    symptoms=matched,
                    explanation=rule['explanation']
                )
                results.append(result)
        
        # 按置信度排序
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"证型判断完成：{[r.syndrome_name for r in results]}")
        return results
    
    def _check_contraindications(self, syndromes: List[SyndromeResult]) -> List[str]:
        """
        禁忌检查
        
        Args:
            syndromes: 证型判断结果
            
        Returns:
            禁忌列表
        """
        contraindications = []
        
        # 根据证型检查禁忌
        for syndrome in syndromes:
            if syndrome.syndrome_name == '脾胃虚弱':
                contraindications.append('脾胃虚寒者慎用寒凉药物')
        
        logger.info(f"禁忌检查完成：{contraindications}")
        return contraindications
    
    def _generate_suggestions(self, syndromes: List[SyndromeResult]) -> List[str]:
        """
        生成调理建议
        
        Args:
            syndromes: 证型判断结果
            
        Returns:
            建议列表
        """
        suggestions = []
        
        if not syndromes:
            return ['建议咨询专业中医师进行辨证施治']
        
        top_syndrome = syndromes[0]
        
        if top_syndrome.syndrome_name == '脾胃虚弱':
            suggestions.extend([
                '饮食宜清淡易消化',
                '避免生冷油腻食物',
                '可适当食用山药、薏米等健脾食材'
            ])
        elif top_syndrome.syndrome_name == '肝郁气滞':
            suggestions.extend([
                '保持心情舒畅',
                '适当运动疏解压力',
                '可饮用玫瑰花茶疏肝理气'
            ])
        
        return suggestions
    
    def get_symptom_info(self, symptom_name: str) -> Optional[Dict]:
        """
        获取症状信息

        注意：新版本的数据库已移除 symptoms 表，
        此方法现在使用内置的症状信息表作为数据源。

        Args:
            symptom_name: 症状名称

        Returns:
            症状详细信息
        """
        return self._get_basic_symptom_info(symptom_name)
    
    def _get_basic_symptom_info(self, symptom_name: str) -> Optional[Dict]:
        """
        获取基础症状信息（当数据库不可用时的备选方案）

        Args:
            symptom_name: 症状名称

        Returns:
            症状基础信息
        """
        # 常见症状的简单描述
        basic_info = {
            '乏力': {'category': '全身症状', 'description': '身体疲倦无力，精神不振', 'related_syndromes': '脾胃虚弱，气血两虚，肾阳虚'},
            '食欲不振': {'category': '消化系统', 'description': '不想进食，食量减少', 'related_syndromes': '脾胃虚弱，肝郁气滞'},
            '失眠': {'category': '精神症状', 'description': '难以入睡或睡而易醒', 'related_syndromes': '心脾两虚，肝郁化火，阴虚火旺'},
            '头痛': {'category': '头部症状', 'description': '头部疼痛', 'related_syndromes': '肝阳上亢，气血亏虚，痰浊上扰'},
            '咳嗽': {'category': '呼吸系统', 'description': '肺气上逆，发出咳声', 'related_syndromes': '风寒束肺，风热犯肺，痰湿阻肺'},
            '腹痛': {'category': '消化系统', 'description': '腹部疼痛', 'related_syndromes': '寒邪内阻，湿热壅滞，气滞血瘀'},
            '腹泻': {'category': '消化系统', 'description': '大便次数增多，粪质稀薄', 'related_syndromes': '脾虚湿盛，肾阳虚衰'},
            '便秘': {'category': '消化系统', 'description': '大便干结，排便困难', 'related_syndromes': '热结肠道，津亏血少，气虚传导无力'},
            '心悸': {'category': '心血管系统', 'description': '心中悸动，惊惕不安', 'related_syndromes': '心气虚，心血不足，心阳不振'},
            '头晕': {'category': '头部症状', 'description': '头目眩晕，视物旋转', 'related_syndromes': '肝阳上亢，气血亏虚，肾精不足'},
            '发热': {'category': '全身症状', 'description': '体温升高或自觉发热', 'related_syndromes': '外感风寒，外感风热，阴虚发热'},
            '恶寒': {'category': '全身症状', 'description': '自觉怕冷，得温不减', 'related_syndromes': '外感风寒，阳虚'},
            '胁痛': {'category': '消化系统', 'description': '胁肋部疼痛', 'related_syndromes': '肝郁气滞，瘀血阻络，肝胆湿热'},
            '自汗': {'category': '全身症状', 'description': '白天不活动或轻微活动即出汗', 'related_syndromes': '气虚，阳虚'},
            '恶心呕吐': {'category': '消化系统', 'description': '胃气上逆，欲吐或吐出物', 'related_syndromes': '胃寒，胃热，痰饮，食滞'},
            '胃痛': {'category': '消化系统', 'description': '胃部疼痛', 'related_syndromes': '寒邪客胃，饮食伤胃，肝气犯胃'},
            '腹胀': {'category': '消化系统', 'description': '腹部胀满不适', 'related_syndromes': '脾虚气滞，食积，湿阻'},
            '腰膝酸软': {'category': '全身症状', 'description': '腰部和膝部酸软无力', 'related_syndromes': '肾阳虚，肾阴虚'},
            '盗汗': {'category': '全身症状', 'description': '睡中出汗，醒后即止', 'related_syndromes': '阴虚火旺，气阴两虚'},
            '耳鸣': {'category': '头部症状', 'description': '耳中鸣响', 'related_syndromes': '肾精不足，肝火上炎'},
            '健忘': {'category': '精神症状', 'description': '记忆力减退', 'related_syndromes': '心脾两虚，肾精不足'},
            '气短': {'category': '全身症状', 'description': '呼吸短促', 'related_syndromes': '肺气虚，肾不纳气'},
            '畏寒': {'category': '全身症状', 'description': '怕冷，得温可减', 'related_syndromes': '阳虚'},
            '手足心热': {'category': '全身症状', 'description': '手心脚心发热', 'related_syndromes': '阴虚火旺'},
            '口干': {'category': '全身症状', 'description': '口中干燥', 'related_syndromes': '阴虚，热盛津伤'},
        }

        info = basic_info.get(symptom_name)
        if info:
            return {
                'name': symptom_name,
                'category': info['category'],
                'description': info['description'],
                'related_syndromes': info['related_syndromes']
            }
        return None
