import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv
import torch.nn.functional as F
import logging

# 创建logger实例
logger = logging.getLogger(__name__)

class TCMKGEmbeddingLayer(nn.Module):
    """
    中药知识图谱嵌入层
    为不同类型的实体和关系创建嵌入表示
    """
    
    def __init__(self, entity_type_sizes: Dict[str, int], relation_type_size: int, embedding_dim: int = 128):
        """
        初始化嵌入层
        
        Args:
            entity_type_sizes (Dict[str, int]): 不同实体类型的数量
            relation_type_size (int): 关系类型数量
            embedding_dim (int): 嵌入维度
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.entity_type_sizes = entity_type_sizes
        
        # 为每种实体类型创建嵌入层
        self.entity_embeddings = nn.ModuleDict({
            entity_type: nn.Embedding(size, embedding_dim)
            for entity_type, size in entity_type_sizes.items()
        })
        
        # 关系嵌入层
        self.relation_embedding = nn.Embedding(relation_type_size, embedding_dim)
        
        # 实体类型嵌入（用于区分不同类型的实体）
        self.entity_type_embedding = nn.Embedding(len(entity_type_sizes), embedding_dim)
        
        # 初始化嵌入
        self._init_embeddings()
    
    def _init_embeddings(self):
        """初始化嵌入权重"""
        for emb in self.entity_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        nn.init.xavier_uniform_(self.entity_type_embedding.weight)
    
    def forward(self, entity_ids: torch.Tensor, entity_types: torch.Tensor, 
                relation_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            entity_ids (torch.Tensor): 实体ID张量, shape: (num_entities,)
            entity_types (torch.Tensor): 实体类型张量, shape: (num_entities,)
            relation_ids (Optional[torch.Tensor]): 关系ID张量, shape: (num_relations,)
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 实体嵌入和关系嵌入
        """
        # 获取实体类型对应的嵌入层
        entity_embeddings = []
        entity_type_keys = list(self.entity_embeddings.keys())
        num_entity_types = len(entity_type_keys)
        
        for i, (entity_id, entity_type) in enumerate(zip(entity_ids, entity_types)):
            # 获取实体类型名称，添加边界检查
            entity_type_idx = entity_type.item()
            if entity_type_idx >= num_entity_types:
                # 如果索引超出范围，默认使用第一个实体类型
                entity_type_idx = 0
                logger.warning(f"警告: 实体类型索引 {entity_type.item()} 超出范围，使用默认类型 0")
            entity_type_name = entity_type_keys[entity_type_idx]
            
            # 获取实体ID并检查范围
            entity_id_val = entity_id.item()
            entity_emb_layer = self.entity_embeddings[entity_type_name]
            
            # 正确处理实体ID范围检查
            if entity_id_val >= entity_emb_layer.num_embeddings:
                # 如果实体ID超出范围，使用模运算确保在范围内
                entity_id_val = entity_id_val % max(1, entity_emb_layer.num_embeddings)
                logger.warning(f"警告: 实体ID {entity_id.item()} 超出 {entity_type_name} 类型范围，调整为 {entity_id_val}")
            
            # 获取实体嵌入
            entity_emb = entity_emb_layer(torch.tensor(entity_id_val, device=entity_id.device))
            
            # 添加实体类型嵌入
            # 确保实体类型索引在有效范围内
            type_emb_idx = min(entity_type_idx, self.entity_type_embedding.num_embeddings - 1)
            type_emb = self.entity_type_embedding(torch.tensor(type_emb_idx, device=entity_type.device))
            entity_embeddings.append(entity_emb + type_emb)
        
        # 合并所有实体嵌入
        entity_embedding = torch.stack(entity_embeddings, dim=0)
        
        # 获取关系嵌入
        relation_embedding = None
        if relation_ids is not None:
            # 检查关系ID范围
            valid_relation_ids = []
            for rel_id in relation_ids:
                rel_id_val = rel_id.item()
                if rel_id_val >= self.relation_embedding.num_embeddings:
                    # 如果关系ID超出范围，使用模运算确保在范围内
                    rel_id_val = rel_id_val % self.relation_embedding.num_embeddings
                    logger.warning(f"警告: 关系ID {rel_id.item()} 超出范围，调整为 {rel_id_val}")
                valid_relation_ids.append(rel_id_val)
            
            valid_relation_ids = torch.tensor(valid_relation_ids, device=relation_ids.device)
            relation_embedding = self.relation_embedding(valid_relation_ids)
            
        return entity_embedding, relation_embedding


class TCMGNNInputLayer(nn.Module):
    """
    中药知识图谱模型输入层
    处理症状输入并准备图神经网络的输入
    """
    
    def __init__(self, 
                 entity_type_sizes: Dict[str, int], 
                 relation_type_size: int,
                 embedding_dim: int = 128,
                 symptom_vocab_size: int = 5000):
        """
        初始化输入层
        
        Args:
            entity_type_sizes (Dict[str, int]): 不同实体类型的数量
            relation_type_size (int): 关系类型数量
            embedding_dim (int): 嵌入维度
            symptom_vocab_size (int): 症状词汇表大小
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 症状词汇嵌入（用于处理原始症状文本）
        self.symptom_word_embedding = nn.Embedding(symptom_vocab_size, embedding_dim)
        
        # 知识图谱嵌入层
        self.kg_embedding_layer = TCMKGEmbeddingLayer(
            entity_type_sizes, relation_type_size, embedding_dim
        )
        
        # 症状编码器（将症状文本编码为向量）
        self.symptom_encoder = nn.LSTM(
            embedding_dim, 
            embedding_dim, 
            num_layers=1, 
            batch_first=True,
            bidirectional=True
        )
        
        # 症状-实体注意力层
        self.symptom_entity_attention = nn.MultiheadAttention(
            embedding_dim * 2,  # 双向LSTM输出维度
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 投影层（将双向LSTM输出投影到与实体嵌入相同的维度）
        self.projection = nn.Linear(embedding_dim * 2, embedding_dim)
        
    def forward(self, 
                symptom_tokens: torch.Tensor,
                kg_entity_ids: torch.Tensor,
                kg_entity_types: torch.Tensor,
                kg_relation_ids: Optional[torch.Tensor] = None,
                kg_edge_index: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            symptom_tokens (torch.Tensor): 症状词ID序列, shape: (batch_size, seq_len)
            kg_entity_ids (torch.Tensor): 知识图谱实体ID, shape: (num_entities,)
            kg_entity_types (torch.Tensor): 实体类型, shape: (num_entities,)
            kg_relation_ids (Optional[torch.Tensor]): 关系ID, shape: (num_relations,)
            kg_edge_index (Optional[torch.Tensor]): 边索引, shape: (2, num_edges)
            
        Returns:
            Dict[str, Any]: 包含处理后输入的字典
        """
        batch_size, seq_len = symptom_tokens.shape
        
        # 1. 获取症状词嵌入，添加边界检查
        # 确保所有症状token都在词汇表范围内
        symptom_tokens_clamped = torch.clamp(symptom_tokens, 0, self.symptom_word_embedding.num_embeddings - 1)
        symptom_word_embeddings = self.symptom_word_embedding(symptom_tokens_clamped)  # (batch_size, seq_len, embedding_dim)
        
        # 2. 编码症状序列
        symptom_encoded, _ = self.symptom_encoder(symptom_word_embeddings)  # (batch_size, seq_len, embedding_dim*2)
        
        # 3. 投影到实体嵌入维度
        symptom_projected = self.projection(symptom_encoded)  # (batch_size, seq_len, embedding_dim)
        
        # 4. 获取知识图谱实体和关系嵌入
        entity_embeddings, relation_embeddings = self.kg_embedding_layer(
            kg_entity_ids, kg_entity_types, kg_relation_ids
        )
        
        # 5. 症状-实体注意力
        # 将实体嵌入扩展为batch_size维度以进行注意力计算
        entity_embeddings_expanded = entity_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (batch_size, num_entities, embedding_dim)
        
        # 计算注意力
        attended_entities, attention_weights = self.symptom_entity_attention(
            symptom_projected,  # 查询 (Q)
            entity_embeddings_expanded,  # 键 (K)
            entity_embeddings_expanded   # 值 (V)
        )
        
        # 6. 返回处理后的输入
        return {
            'symptom_embeddings': symptom_projected,  # (batch_size, seq_len, embedding_dim)
            'entity_embeddings': entity_embeddings,   # (num_entities, embedding_dim)
            'relation_embeddings': relation_embeddings,  # (num_relations, embedding_dim) or None
            'attended_entities': attended_entities,   # (batch_size, seq_len, embedding_dim)
            'attention_weights': attention_weights,   # (batch_size, seq_len, num_entities)
            'edge_index': kg_edge_index  # (2, num_edges) or None
        }