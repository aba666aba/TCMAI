import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from torch_geometric.nn import SAGEConv, GATConv, GCNConv
from torch_geometric.data import Data
import torch.nn.functional as F


class TCMGNNLayer(nn.Module):
    """
    中药知识图谱图神经网络层
    使用GraphSAGE进行实体表示学习
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2, dropout: float = 0.1):
        """
        初始化GNN层
        
        Args:
            in_channels (int): 输入特征维度
            out_channels (int): 输出特征维度
            num_layers (int): GNN层数
            dropout (float): Dropout概率
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 创建多层GraphSAGE卷积层
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, out_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(out_channels, out_channels))
        
        # 批量归一化层
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(out_channels))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 节点特征, shape: (num_nodes, in_channels)
            edge_index (torch.Tensor): 边索引, shape: (2, num_edges)
            
        Returns:
            torch.Tensor: 节点嵌入, shape: (num_nodes, out_channels)
        """
        # 多层GNN
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            # 图卷积
            x = conv(x, edge_index)
            
            # 批量归一化
            if x.size(0) > 1:  # 批量大小大于1时才使用批量归一化
                x = bn(x)
            
            # 激活函数和dropout（除了最后一层）
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class TCMGATLayer(nn.Module):
    """
    中药知识图谱图注意力网络层
    使用GAT进行实体表示学习，能够关注重要邻居
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2, 
                 heads: int = 4, dropout: float = 0.1):
        """
        初始化GAT层
        
        Args:
            in_channels (int): 输入特征维度
            out_channels (int): 输出特征维度
            num_layers (int): GAT层数
            heads (int): 注意力头数
            dropout (float): Dropout概率
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 创建多层GAT卷积层
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, out_channels, heads=heads, dropout=dropout))
        
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(out_channels * heads, out_channels, heads=heads, dropout=dropout))
        
        # 输出层（减少头数为1）
        self.convs.append(GATConv(out_channels * heads, out_channels, heads=1, dropout=dropout))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 节点特征, shape: (num_nodes, in_channels)
            edge_index (torch.Tensor): 边索引, shape: (2, num_edges)
            
        Returns:
            torch.Tensor: 节点嵌入, shape: (num_nodes, out_channels)
        """
        # 多层GAT（除了最后一层）
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最后一层
        x = self.convs[-1](x, edge_index)
        
        return x


class TCMKnowledgeGraphEncoder(nn.Module):
    """
    中药知识图谱编码器
    结合多层GNN和注意力机制进行知识图谱编码
    """
    
    def __init__(self, 
                 entity_embedding_dim: int = 128,
                 gnn_hidden_dim: int = 256,
                 gnn_output_dim: int = 128,
                 num_gnn_layers: int = 3,
                 gnn_type: str = 'sage',  # 'sage' or 'gat'
                 dropout: float = 0.1):
        """
        初始化知识图谱编码器
        
        Args:
            entity_embedding_dim (int): 实体嵌入维度
            gnn_hidden_dim (int): GNN隐藏层维度
            gnn_output_dim (int): GNN输出维度
            num_gnn_layers (int): GNN层数
            gnn_type (str): GNN类型 ('sage' or 'gat')
            dropout (float): Dropout概率
        """
        super().__init__()
        self.gnn_type = gnn_type
        
        # GNN编码器
        if gnn_type == 'sage':
            self.gnn_encoder = TCMGNNLayer(
                entity_embedding_dim, 
                gnn_output_dim, 
                num_gnn_layers, 
                dropout
            )
        elif gnn_type == 'gat':
            self.gnn_encoder = TCMGATLayer(
                entity_embedding_dim, 
                gnn_output_dim // 4,  # GAT会将输出维度乘以头数
                num_gnn_layers, 
                heads=4,
                dropout=dropout
            )
        else:
            raise ValueError(f"不支持的GNN类型: {gnn_type}")
        
        # 全局图表示（通过全局平均池化获得）
        self.global_graph_projection = nn.Sequential(
            nn.Linear(gnn_output_dim, gnn_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_output_dim, gnn_output_dim)
        )
    
    def forward(self, entity_embeddings: torch.Tensor, 
                edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            entity_embeddings (torch.Tensor): 实体嵌入, shape: (num_entities, embedding_dim)
            edge_index (torch.Tensor): 边索引, shape: (2, num_edges)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - 更新后的实体嵌入, shape: (num_entities, gnn_output_dim)
                - 全局图表示, shape: (gnn_output_dim,)
        """
        # 通过GNN编码器更新实体表示
        updated_entity_embeddings = self.gnn_encoder(entity_embeddings, edge_index)
        
        # 生成全局图表示（通过平均池化）
        global_graph_representation = torch.mean(updated_entity_embeddings, dim=0)
        global_graph_representation = self.global_graph_projection(global_graph_representation)
        
        return updated_entity_embeddings, global_graph_representation


class TCMAttentionFusion(nn.Module):
    """
    注意力融合层
    融合症状表示和知识图谱表示
    """
    
    def __init__(self, feature_dim: int, dropout: float = 0.1):
        """
        初始化注意力融合层
        
        Args:
            feature_dim (int): 特征维度
            dropout (float): Dropout概率
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        # 多头注意力机制
        self.multihead_attention = nn.MultiheadAttention(
            feature_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, symptom_features: torch.Tensor, 
                kg_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            symptom_features (torch.Tensor): 症状特征, shape: (batch_size, seq_len, feature_dim)
            kg_features (torch.Tensor): 知识图谱特征, shape: (batch_size, num_entities, feature_dim)
            
        Returns:
            torch.Tensor: 融合后的特征, shape: (batch_size, seq_len, feature_dim)
        """
        batch_size, seq_len, _ = symptom_features.shape
        
        # 将知识图谱特征扩展到每个时间步
        kg_features_expanded = kg_features.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
        # 重塑为适合注意力机制的形状
        symptom_features_flat = symptom_features.reshape(-1, self.feature_dim)
        kg_features_flat = kg_features_expanded.reshape(-1, kg_features.shape[2], self.feature_dim)
        
        # 多头注意力
        attended_features, _ = self.multihead_attention(
            symptom_features_flat.unsqueeze(1),  # 查询
            kg_features_flat,                     # 键
            kg_features_flat                      # 值
        )
        attended_features = attended_features.squeeze(1)
        
        # 残差连接和层归一化
        attended_features = self.layer_norm1(
            symptom_features_flat + attended_features
        )
        
        # 前馈网络
        ff_output = self.feed_forward(attended_features)
        
        # 残差连接和层归一化
        output = self.layer_norm2(attended_features + ff_output)
        
        # 恢复原始形状
        output = output.reshape(batch_size, seq_len, self.feature_dim)
        
        return output


# 测试代码
def test_gnn_components():
    """测试GNN组件"""
    # 测试参数
    num_entities = 100
    num_edges = 500
    embedding_dim = 128
    batch_size = 4
    seq_len = 10
    
    # 创建测试数据
    entity_embeddings = torch.randn(num_entities, embedding_dim)
    edge_index = torch.randint(0, num_entities, (2, num_edges))
    symptom_features = torch.randn(batch_size, seq_len, embedding_dim)
    kg_features = torch.randn(batch_size, num_entities, embedding_dim)
    
    # 测试GNN编码器
    print("测试GNN编码器...")
    gnn_encoder = TCMKnowledgeGraphEncoder(
        entity_embedding_dim=embedding_dim,
        gnn_hidden_dim=256,
        gnn_output_dim=128,
        num_gnn_layers=3,
        gnn_type='sage'
    )
    
    updated_entities, global_repr = gnn_encoder(entity_embeddings, edge_index)
    print(f"更新后的实体嵌入形状: {updated_entities.shape}")
    print(f"全局图表示形状: {global_repr.shape}")
    
    # 测试注意力融合层
    print("\n测试注意力融合层...")
    attention_fusion = TCMAttentionFusion(embedding_dim)
    fused_features = attention_fusion(symptom_features, kg_features)
    print(f"融合后的特征形状: {fused_features.shape}")


if __name__ == "__main__":
    test_gnn_components()