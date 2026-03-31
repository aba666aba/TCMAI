# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0

"""
中医药推荐系统 - 深度学习模型

基于症状推荐中药材和方剂
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SymptomEncoder(nn.Module):
    """症状编码器 - 将症状列表编码为向量"""
    
    def __init__(self, num_symptoms: int, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_symptoms + 1, embed_dim, padding_idx=0)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, symptoms: torch.Tensor) -> torch.Tensor:
        """
        编码症状
        
        Args:
            symptoms: [batch_size, seq_len] 症状 ID 序列
        
        Returns:
            [batch_size, embed_dim] 编码后的症状向量
        """
        # 嵌入编码
        embedded = self.embedding(symptoms)  # [batch, seq, embed]
        
        # 自注意力
        attended, _ = self.attention(embedded, embedded, embedded)
        
        # 层归一化
        attended = self.layer_norm(attended + embedded)
        
        # 全局池化（取平均）
        mask = (symptoms != 0).unsqueeze(-1).float()
        pooled = (attended * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        
        return pooled


class HerbRecommender(nn.Module):
    """药材推荐器"""
    
    def __init__(self, symptom_dim: int, num_herbs: int, hidden_dim: int = 128):
        super().__init__()
        self.num_herbs = num_herbs
        
        # 推荐网络
        self.recommender = nn.Sequential(
            nn.Linear(symptom_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_herbs)
        )
        
        # 药材特征嵌入（可选）
        self.herb_embedding = nn.Embedding(num_herbs + 1, 32)
    
    def forward(self, symptom_vector: torch.Tensor) -> torch.Tensor:
        """
        推荐药材
        
        Args:
            symptom_vector: [batch_size, symptom_dim] 症状向量
        
        Returns:
            [batch_size, num_herbs] 每个药材的推荐分数
        """
        scores = self.recommender(symptom_vector)
        return scores


class FormulaRecommender(nn.Module):
    """方剂推荐器"""
    
    def __init__(self, symptom_dim: int, num_formulas: int, hidden_dim: int = 64):
        super().__init__()
        self.recommender = nn.Sequential(
            nn.Linear(symptom_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_formulas)
        )
    
    def forward(self, symptom_vector: torch.Tensor) -> torch.Tensor:
        """
        推荐方剂
        
        Args:
            symptom_vector: [batch_size, symptom_dim] 症状向量
        
        Returns:
            [batch_size, num_formulas] 每个方剂的推荐分数
        """
        scores = self.recommender(symptom_vector)
        return scores


class TCMRecommenderSystem(nn.Module):
    """
    中医药推荐系统（主模型）
    
    输入：症状列表
    输出：推荐的药材和方剂
    """
    
    def __init__(
        self,
        num_symptoms: int,
        num_herbs: int,
        num_formulas: int,
        symptom_embed_dim: int = 64,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # 症状编码器
        self.symptom_encoder = SymptomEncoder(
            num_symptoms=num_symptoms,
            embed_dim=symptom_embed_dim
        )
        
        # 药材推荐器
        self.herb_recommender = HerbRecommender(
            symptom_dim=symptom_embed_dim,
            num_herbs=num_herbs,
            hidden_dim=hidden_dim
        )
        
        # 方剂推荐器
        self.formula_recommender = FormulaRecommender(
            symptom_dim=symptom_embed_dim,
            num_formulas=num_formulas,
            hidden_dim=hidden_dim // 2
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(
        self,
        symptoms: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            symptoms: [batch_size, seq_len] 症状 ID 序列
        
        Returns:
            (herb_scores, formula_scores) - 药材和方剂推荐分数
        """
        # 编码症状
        symptom_vector = self.symptom_encoder(symptoms)
        
        # 推荐药材和方剂
        herb_scores = self.herb_recommender(symptom_vector)
        formula_scores = self.formula_recommender(symptom_vector)
        
        return herb_scores, formula_scores
    
    def predict(
        self,
        symptoms: torch.Tensor,
        top_k: int = 10
    ) -> Dict[str, List]:
        """
        预测推荐
        
        Args:
            symptoms: [batch_size, seq_len] 症状 ID 序列
            top_k: 返回前 K 个推荐
        
        Returns:
            推荐结果字典
        """
        self.eval()
        with torch.no_grad():
            herb_scores, formula_scores = self.forward(symptoms)
            
            # 获取 Top-K
            herb_topk = torch.topk(herb_scores[0], min(top_k, herb_scores.shape[1]))
            formula_topk = torch.topk(formula_scores[0], min(top_k, formula_scores.shape[1]))
            
            return {
                'herbs': list(zip(herb_topk.indices.tolist(), herb_topk.values.tolist())),
                'formulas': list(zip(formula_topk.indices.tolist(), formula_topk.values.tolist()))
            }


class SafetyChecker:
    """
    安全校验器 - 检查配伍禁忌
    """
    
    def __init__(self, incompatible_herbs: Dict[str, List]):
        """
        初始化安全校验器
        
        Args:
            incompatible_herbs: 配伍禁忌字典
        """
        self.incompatible_pairs = set()
        self.pregnancy_forbidden = set()
        
        # 解析配伍禁忌
        for category, items in incompatible_herbs.items():
            if category == '孕妇禁用':
                self.pregnancy_forbidden = set(items)
            else:
                for pair in items:
                    if isinstance(pair, tuple):
                        self.incompatible_pairs.add(pair)
                        self.incompatible_pairs.add((pair[1], pair[0]))  # 双向
    
    def check_compatibility(self, herb_names: List[str]) -> Dict:
        """
        检查药材配伍兼容性
        
        Args:
            herb_names: 药材名称列表
        
        Returns:
            检查结果字典
        """
        warnings = []
        incompatible = []
        toxic = []
        
        # 检查配伍禁忌
        for i, herb1 in enumerate(herb_names):
            for herb2 in herb_names[i+1:]:
                if (herb1, herb2) in self.incompatible_pairs:
                    incompatible.append((herb1, herb2))
                    warnings.append(f"⚠️ 配伍禁忌：{herb1} 与 {herb2} 不可同用（十八反/十九畏）")
        
        # 检查孕妇禁用
        for herb in herb_names:
            if herb in self.pregnancy_forbidden:
                warnings.append(f"⚠️ 孕妇禁用：{herb}")
        
        # 检查毒性
        toxic_herbs = ['生川乌', '生草乌', '附子', '巴豆', '甘遂', '大戟', '芫花']
        for herb in herb_names:
            if herb in toxic_herbs:
                toxic.append(herb)
                warnings.append(f"☠️ 有毒性：{herb}（需谨慎使用）")
        
        return {
            'is_safe': len(incompatible) == 0,
            'warnings': warnings,
            'incompatible_pairs': incompatible,
            'toxic_herbs': toxic
        }


def create_model(
    num_symptoms: int,
    num_herbs: int,
    num_formulas: int
) -> TCMRecommenderSystem:
    """
    创建模型的便捷函数
    
    Args:
        num_symptoms: 症状数量
        num_herbs: 药材数量
        num_formulas: 方剂数量
    
    Returns:
        TCMRecommenderSystem 实例
    """
    model = TCMRecommenderSystem(
        num_symptoms=num_symptoms,
        num_herbs=num_herbs,
        num_formulas=num_formulas,
        symptom_embed_dim=64,
        hidden_dim=128
    )
    
    logger.info(f"创建模型:")
    logger.info(f"  - 症状编码维度：64")
    logger.info(f"  - 隐藏层维度：128")
    logger.info(f"  - 药材数量：{num_herbs}")
    logger.info(f"  - 方剂数量：{num_formulas}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  - 总参数量：{total_params:,}")
    
    return model


if __name__ == '__main__':
    # 测试模型
    print("=== 测试中医药推荐模型 ===\n")
    
    # 创建模型
    model = create_model(
        num_symptoms=32,
        num_herbs=1010,
        num_formulas=347
    )
    
    # 测试前向传播
    batch_size = 2
    seq_len = 5
    symptoms = torch.randint(0, 32, (batch_size, seq_len))
    
    print(f"\n输入症状：{symptoms.shape}")
    
    herb_scores, formula_scores = model.forward(symptoms)
    
    print(f"药材推荐分数：{herb_scores.shape}")
    print(f"方剂推荐分数：{formula_scores.shape}")
    
    # 测试预测
    result = model.predict(symptoms, top_k=5)
    print(f"\nTop-5 药材推荐：{len(result['herbs'])} 个")
    print(f"Top-5 方剂推荐：{len(result['formulas'])} 个")
    
    # 测试安全校验
    safety_checker = SafetyChecker({
        '十八反': [('甘草', '大戟'), ('乌头', '半夏')],
        '孕妇禁用': ['麝香', '三棱', '莪术']
    })
    
    check_result = safety_checker.check_compatibility(['甘草', '大戟', '人参'])
    print(f"\n安全检查结果:")
    print(f"  是否安全：{check_result['is_safe']}")
    print(f"  警告：{len(check_result['warnings'])} 条")
    for w in check_result['warnings']:
        print(f"    - {w}")
