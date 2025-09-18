import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch.nn.functional as F


class TCMHerbDecoder(nn.Module):
    """
    中药处方解码器
    根据症状表示和知识图谱信息生成药材推荐
    """
    
    def __init__(self, 
                 feature_dim: int = 128,
                 herb_vocab_size: int = 1000,
                 max_prescription_length: int = 15,
                 dropout: float = 0.1):
        """
        初始化解码器
        
        Args:
            feature_dim (int): 特征维度
            herb_vocab_size (int): 药材词汇表大小
            max_prescription_length (int): 最大处方长度
            dropout (float): Dropout概率
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.herb_vocab_size = herb_vocab_size
        self.max_prescription_length = max_prescription_length
        
        # 药材嵌入层
        self.herb_embedding = nn.Embedding(herb_vocab_size, feature_dim)
        
        # 解码器LSTM
        self.decoder_lstm = nn.LSTM(
            feature_dim, 
            feature_dim, 
            num_layers=2, 
            batch_first=True,
            dropout=dropout if 2 > 1 else 0
        )
        
        # 注意力机制（用于关注症状和知识图谱信息）
        self.attention = nn.MultiheadAttention(
            feature_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层（生成药材概率分布）
        self.output_projection = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),  # 连接LSTM输出和注意力上下文
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, herb_vocab_size)
        )
        
        # 处方级注意力（用于捕捉药材间的相互作用）
        self.prescription_attention = nn.MultiheadAttention(
            feature_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(self, 
                symptom_representation: torch.Tensor,
                kg_herb_embeddings: torch.Tensor,
                herb_sequence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            symptom_representation (torch.Tensor): 症状表示, shape: (batch_size, feature_dim)
            kg_herb_embeddings (torch.Tensor): 知识图谱中的药材嵌入, shape: (num_herbs, feature_dim)
            herb_sequence (Optional[torch.Tensor]): 已生成的药材序列, shape: (batch_size, seq_len)
            
        Returns:
            torch.Tensor: 药材概率分布, shape: (batch_size, max_prescription_length, herb_vocab_size)
        """
        batch_size = symptom_representation.size(0)
        
        # 如果没有提供药材序列，则从零开始生成
        if herb_sequence is None:
            # 初始化隐藏状态和细胞状态
            h0 = symptom_representation.unsqueeze(0).repeat(2, 1, 1)  # (num_layers, batch_size, feature_dim)
            c0 = torch.zeros_like(h0)
            
            # 存储每个时间步的输出
            outputs = []
            hidden = (h0, c0)
            
            # 初始化输入（使用症状表示作为第一个输入）
            decoder_input = symptom_representation.unsqueeze(1)  # (batch_size, 1, feature_dim)
            
            # 逐步生成药材序列
            for _ in range(self.max_prescription_length):
                # LSTM解码
                lstm_output, hidden = self.decoder_lstm(decoder_input, hidden)
                
                # 注意力机制：关注知识图谱中的药材
                attended_herbs, _ = self.attention(
                    lstm_output,  # 查询
                    kg_herb_embeddings.unsqueeze(0).expand(batch_size, -1, -1),  # 键
                    kg_herb_embeddings.unsqueeze(0).expand(batch_size, -1, -1)   # 值
                )
                
                # 连接LSTM输出和注意力上下文
                combined_output = torch.cat([lstm_output, attended_herbs], dim=-1)
                
                # 生成药材概率分布
                herb_logits = self.output_projection(combined_output)
                outputs.append(herb_logits)
                
                # 使用当前输出作为下一个时间步的输入（实际应用中可以使用采样或贪婪策略）
                # 这里简化处理，直接使用概率最高的药材
                _, predicted_herb = torch.max(herb_logits, dim=-1)
                # 确保预测的药材ID在有效范围内
                predicted_herb_clamped = torch.clamp(predicted_herb, 0, self.herb_vocab_size - 1)
                decoder_input = self.herb_embedding(predicted_herb_clamped)
        
            # 堆叠所有时间步的输出
            outputs = torch.stack(outputs, dim=1)  # (batch_size, max_prescription_length, herb_vocab_size)
        else:
            # 如果提供了药材序列，则计算其概率分布
            seq_len = herb_sequence.size(1)
            
            # 确保所有药材ID都在词汇表范围内
            herb_sequence_clamped = torch.clamp(herb_sequence, 0, self.herb_vocab_size - 1)
            
            # 获取药材嵌入
            herb_embeddings = self.herb_embedding(herb_sequence_clamped)  # (batch_size, seq_len, feature_dim)
            
            # LSTM编码
            lstm_outputs, _ = self.decoder_lstm(herb_embeddings)
            
            # 注意力机制：关注知识图谱中的药材
            attended_herbs, _ = self.attention(
                lstm_outputs,  # 查询
                kg_herb_embeddings.unsqueeze(0).expand(batch_size, -1, -1),  # 键
                kg_herb_embeddings.unsqueeze(0).expand(batch_size, -1, -1)   # 值
            )
            
            # 连接LSTM输出和注意力上下文
            combined_outputs = torch.cat([lstm_outputs, attended_herbs], dim=-1)
            
            # 生成药材概率分布
            outputs = self.output_projection(combined_outputs)  # (batch_size, seq_len, herb_vocab_size)
            
            # 如果序列长度小于最大处方长度，进行填充
            if seq_len < self.max_prescription_length:
                padding = torch.zeros(
                    batch_size, 
                    self.max_prescription_length - seq_len, 
                    self.herb_vocab_size,
                    device=outputs.device
                )
                outputs = torch.cat([outputs, padding], dim=1)
        
        return outputs


class TCMInteractionAwareDecoder(nn.Module):
    """
    考虑药材相互作用的解码器
    在生成处方时考虑药材间的相生相克关系
    """
    
    def __init__(self, 
                 feature_dim: int = 128,
                 herb_vocab_size: int = 1000,
                 interaction_matrix: Optional[torch.Tensor] = None,
                 max_prescription_length: int = 15,
                 dropout: float = 0.1):
        """
        初始化考虑相互作用的解码器
        
        Args:
            feature_dim (int): 特征维度
            herb_vocab_size (int): 药材词汇表大小
            interaction_matrix (Optional[torch.Tensor]): 药材相互作用矩阵, shape: (herb_vocab_size, herb_vocab_size)
            max_prescription_length (int): 最大处方长度
            dropout (float): Dropout概率
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.herb_vocab_size = herb_vocab_size
        self.max_prescription_length = max_prescription_length
        
        # 基础解码器
        self.base_decoder = TCMHerbDecoder(
            feature_dim, herb_vocab_size, max_prescription_length, dropout
        )
        
        # 药材相互作用矩阵
        if interaction_matrix is not None:
            self.interaction_matrix = nn.Parameter(interaction_matrix, requires_grad=False)
        else:
            # 初始化为单位矩阵（无相互作用）
            self.interaction_matrix = nn.Parameter(
                torch.eye(herb_vocab_size), 
                requires_grad=False
            )
        
        # 相互作用调整层
        self.interaction_adjustment = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                symptom_representation: torch.Tensor,
                kg_herb_embeddings: torch.Tensor,
                herb_sequence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            symptom_representation (torch.Tensor): 症状表示, shape: (batch_size, feature_dim)
            kg_herb_embeddings (torch.Tensor): 知识图谱中的药材嵌入, shape: (num_herbs, feature_dim)
            herb_sequence (Optional[torch.Tensor]): 已生成的药材序列, shape: (batch_size, seq_len)
            
        Returns:
            torch.Tensor: 调整后的药材概率分布, shape: (batch_size, max_prescription_length, herb_vocab_size)
        """
        # 获取基础解码器输出
        base_logits = self.base_decoder(symptom_representation, kg_herb_embeddings, herb_sequence)
        
        # 如果有已生成的药材序列，调整后续药材的概率分布
        if herb_sequence is not None:
            batch_size, seq_len = herb_sequence.shape
            
            # 对于每个已生成的药材，调整后续药材的概率
            adjusted_logits = base_logits.clone()
            
            for i in range(seq_len):
                current_herb = herb_sequence[:, i]  # 当前药材 (batch_size,)
                
                # 获取当前药材与其他药材的相互作用
                interactions = self.interaction_matrix[current_herb, :]  # (batch_size, herb_vocab_size)
                
                # 扩展维度以匹配logits形状
                interactions = interactions.unsqueeze(1)  # (batch_size, 1, herb_vocab_size)
                
                # 调整后续位置的logits
                adjustment_factor = 1.0 + interactions * 0.5  # 可调整的相互作用强度
                adjusted_logits[:, i+1:, :] = adjusted_logits[:, i+1:, :] * adjustment_factor
                
            return adjusted_logits
        
        return base_logits


class TCMFinalOutputLayer(nn.Module):
    """
    最终输出层
    生成处方得分和处方组合优化
    """
    
    def __init__(self, 
                 feature_dim: int = 128,
                 herb_vocab_size: int = 1000,
                 max_prescription_length: int = 15,
                 dropout: float = 0.1):
        """
        初始化最终输出层
        
        Args:
            feature_dim (int): 特征维度
            herb_vocab_size (int): 药材词汇表大小
            max_prescription_length (int): 最大处方长度
            dropout (float): Dropout概率
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.herb_vocab_size = herb_vocab_size
        self.max_prescription_length = max_prescription_length
        
        # 处方组合评分网络
        self.prescription_scorer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),  # 连接症状表示和处方表示
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # 药材重要性评分
        self.herb_importance_scorer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 处方多样性增强
        self.diversity_enhancer = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, 
                symptom_representation: torch.Tensor,
                herb_logits: torch.Tensor,
                kg_herb_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            symptom_representation (torch.Tensor): 症状表示, shape: (batch_size, feature_dim)
            herb_logits (torch.Tensor): 药材logits, shape: (batch_size, max_prescription_length, herb_vocab_size)
            kg_herb_embeddings (torch.Tensor): 知识图谱中的实体嵌入, shape: (num_entities, feature_dim)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - 处方得分, shape: (batch_size, max_prescription_length)
                - 药材重要性得分, shape: (num_entities,)
        """
        batch_size = symptom_representation.size(0)
        num_entities = kg_herb_embeddings.size(0)
        
        # 计算每个位置的药材重要性得分
        # 注意：这里我们假设所有实体都可能是药材，实际应用中应该根据实体类型过滤
        herb_importance = self.herb_importance_scorer(kg_herb_embeddings)  # (num_entities, 1)
        herb_importance = herb_importance.squeeze(-1)  # (num_entities,)
        
        # 扩展维度以匹配logits
        # 注意：这里我们需要确保维度匹配，实际应用中可能需要更复杂的处理
        herb_importance_expanded = herb_importance.unsqueeze(0).unsqueeze(0)  # (1, 1, num_entities)
        herb_importance_expanded = herb_importance_expanded.expand(
            batch_size, self.max_prescription_length, -1
        )  # (batch_size, max_prescription_length, num_entities)
        
        # 调整logits以考虑药材重要性
        # 注意：这里假设logits的最后一个维度与实体数量匹配
        adjusted_logits = herb_logits * herb_importance_expanded[:, :, :herb_logits.size(2)]
        
        # 对每个处方位置计算得分
        # 使用softmax获取概率分布
        herb_probs = F.softmax(adjusted_logits, dim=-1)  # (batch_size, max_prescription_length, herb_vocab_size)
        
        # 计算每个位置的处方表示（通过加权平均实体嵌入）
        # 注意：这里我们使用所有实体嵌入，实际应用中应该根据实体类型过滤药材实体
        prescription_embeddings = torch.matmul(
            herb_probs, 
            kg_herb_embeddings[:herb_probs.size(2)]  # 确保维度匹配
        )  # (batch_size, max_prescription_length, feature_dim)
        
        # 计算处方得分
        # 连接症状表示和处方表示
        symptom_expanded = symptom_representation.unsqueeze(1).expand(
            -1, self.max_prescription_length, -1
        )  # (batch_size, max_prescription_length, feature_dim)
        
        combined_representation = torch.cat([
            symptom_expanded, 
            prescription_embeddings
        ], dim=-1)  # (batch_size, max_prescription_length, feature_dim * 2)
        
        # 计算处方得分
        prescription_scores = self.prescription_scorer(combined_representation)  # (batch_size, max_prescription_length, 1)
        prescription_scores = prescription_scores.squeeze(-1)  # (batch_size, max_prescription_length)
        
        # 返回处方得分和药材重要性得分
        return prescription_scores, herb_importance  # (batch_size, max_prescription_length), (num_entities,)


# 测试代码
def test_output_layers():
    """测试输出层"""
    # 测试参数
    batch_size = 4
    feature_dim = 128
    herb_vocab_size = 1000
    max_prescription_length = 15
    num_herbs = 1000
    
    # 创建测试数据
    symptom_representation = torch.randn(batch_size, feature_dim)
    herb_logits = torch.randn(batch_size, max_prescription_length, herb_vocab_size)
    kg_herb_embeddings = torch.randn(num_herbs, feature_dim)
    
    # 测试基础解码器
    print("测试基础解码器...")
    decoder = TCMHerbDecoder(feature_dim, herb_vocab_size, max_prescription_length)
    decoder_output = decoder(symptom_representation, kg_herb_embeddings)
    print(f"解码器输出形状: {decoder_output.shape}")
    
    # 测试考虑相互作用的解码器
    print("\n测试考虑相互作用的解码器...")
    interaction_matrix = torch.randn(herb_vocab_size, herb_vocab_size)
    interaction_decoder = TCMInteractionAwareDecoder(
        feature_dim, herb_vocab_size, interaction_matrix, max_prescription_length
    )
    interaction_output = interaction_decoder(symptom_representation, kg_herb_embeddings)
    print(f"相互作用解码器输出形状: {interaction_output.shape}")
    
    # 测试最终输出层
    print("\n测试最终输出层...")
    final_output_layer = TCMFinalOutputLayer(feature_dim, herb_vocab_size, max_prescription_length)
    prescription_scores, herb_importance = final_output_layer(
        symptom_representation, herb_logits, kg_herb_embeddings
    )
    print(f"处方得分形状: {prescription_scores.shape}")
    print(f"药材重要性得分形状: {herb_importance.shape}")


if __name__ == "__main__":
    test_output_layers()