import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F


class TCMGNNInferenceModule(nn.Module):
    """
    中药知识图谱推理模块
    用于在生产环境中进行中药处方推荐
    """
    
    def __init__(self, 
                 entity_type_sizes: Dict[str, int],
                 relation_type_size: int,
                 embedding_dim: int = 128,
                 gnn_output_dim: int = 128,
                 herb_vocab_size: int = 1000,
                 symptom_vocab_size: int = 5000,
                 max_prescription_length: int = 15,
                 device: torch.device = torch.device('cpu')):
        """
        初始化推理模块
        
        Args:
            entity_type_sizes (Dict[str, int]): 实体类型和数量
            relation_type_size (int): 关系类型数量
            embedding_dim (int): 嵌入维度
            gnn_output_dim (int): GNN输出维度
            herb_vocab_size (int): 药材词汇表大小
            symptom_vocab_size (int): 症状词汇表大小
            max_prescription_length (int): 最大处方长度
            device (torch.device): 运行设备
        """
        super().__init__()
        self.device = device
        self.herb_vocab_size = herb_vocab_size
        self.symptom_vocab_size = symptom_vocab_size
        self.max_prescription_length = max_prescription_length
        
        # 导入模型组件
        from tcm_gnn_input import TCMGNNInputLayer
        from tcm_gnn_components import TCMKnowledgeGraphEncoder, TCMAttentionFusion
        from tcm_gnn_output import TCMInteractionAwareDecoder, TCMFinalOutputLayer
        
        # 输入层
        self.input_layer = TCMGNNInputLayer(
            entity_type_sizes, 
            relation_type_size, 
            embedding_dim,
            symptom_vocab_size
        )
        
        # 知识图谱编码器
        self.kg_encoder = TCMKnowledgeGraphEncoder(
            entity_embedding_dim=embedding_dim,
            gnn_hidden_dim=256,
            gnn_output_dim=gnn_output_dim,
            num_gnn_layers=3,
            gnn_type='sage'
        )
        
        # 注意力融合层
        self.attention_fusion = TCMAttentionFusion(embedding_dim)
        
        # 解码器
        self.decoder = TCMInteractionAwareDecoder(
            feature_dim=embedding_dim,
            herb_vocab_size=herb_vocab_size,
            max_prescription_length=max_prescription_length
        )
        
        # 最终输出层
        self.final_output = TCMFinalOutputLayer(
            feature_dim=embedding_dim,
            herb_vocab_size=herb_vocab_size,
            max_prescription_length=max_prescription_length
        )
        
        # 移动到指定设备
        self.to(device)
    
    def forward(self, 
                symptom_tokens: torch.Tensor,
                kg_entity_ids: torch.Tensor,
                kg_entity_types: torch.Tensor,
                kg_relation_ids: Optional[torch.Tensor],
                kg_edge_index: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播（推理）
        
        Args:
            symptom_tokens (torch.Tensor): 症状词ID序列
            kg_entity_ids (torch.Tensor): 知识图谱实体ID
            kg_entity_types (torch.Tensor): 实体类型
            kg_relation_ids (Optional[torch.Tensor]): 关系ID
            kg_edge_index (Optional[torch.Tensor]): 边索引
            
        Returns:
            Dict[str, torch.Tensor]: 推理结果
        """
        # 1. 输入层处理
        input_outputs = self.input_layer(
            symptom_tokens=symptom_tokens,
            kg_entity_ids=kg_entity_ids,
            kg_entity_types=kg_entity_types,
            kg_relation_ids=kg_relation_ids,
            kg_edge_index=kg_edge_index
        )
        
        # 2. 知识图谱编码
        entity_embeddings = input_outputs['entity_embeddings']
        updated_entity_embeddings, global_graph_representation = self.kg_encoder(
            entity_embeddings, kg_edge_index
        )
        
        # 3. 注意力融合
        symptom_embeddings = input_outputs['symptom_embeddings']
        attended_symptoms = self.attention_fusion(
            symptom_embeddings, 
            updated_entity_embeddings.unsqueeze(0).expand(
                symptom_embeddings.size(0), -1, -1
            )
        )
        
        # 4. 症状表示聚合
        symptom_representation = attended_symptoms.mean(dim=1)  # (batch_size, embedding_dim)
        
        # 5. 解码器生成药材概率分布
        herb_logits = self.decoder(
            symptom_representation=symptom_representation,
            kg_herb_embeddings=updated_entity_embeddings[:self.herb_vocab_size]  # 假设前herb_vocab_size个实体是药材
        )
        
        # 6. 最终输出层
        prescription_scores, herb_importance = self.final_output(
            symptom_representation=symptom_representation,
            herb_logits=herb_logits,
            kg_herb_embeddings=updated_entity_embeddings[:self.herb_vocab_size]
        )
        
        return {
            'prescription_scores': prescription_scores,
            'herb_logits': herb_logits,
            'herb_importance': herb_importance,
            'symptom_representation': symptom_representation
        }
    
    def recommend_prescription(self, 
                              symptoms: List[str],
                              kg_data: Dict[str, Any],
                              top_k: int = 10) -> List[Dict[str, Any]]:
        """
        根据症状推荐中药处方
        
        Args:
            symptoms (List[str]): 症状列表
            kg_data (Dict[str, Any]): 知识图谱数据
            top_k (int): 返回前k个推荐
            
        Returns:
            List[Dict[str, Any]]: 推荐的处方列表
        """
        # 1. 预处理症状
        symptom_tokens = self._preprocess_symptoms(symptoms)
        
        # 2. 准备知识图谱数据
        kg_tensors = self._prepare_kg_data(kg_data)
        
        # 3. 模型推理
        with torch.no_grad():
            outputs = self.forward(
                symptom_tokens=symptom_tokens,
                kg_entity_ids=kg_tensors['entity_ids'],
                kg_entity_types=kg_tensors['entity_types'],
                kg_relation_ids=kg_tensors['relation_ids'],
                kg_edge_index=kg_tensors['edge_index']
            )
        
        # 4. 后处理生成推荐
        recommendations = self._postprocess_recommendations(
            outputs, 
            kg_data, 
            top_k
        )
        
        return recommendations
    
    def _preprocess_symptoms(self, symptoms: List[str]) -> torch.Tensor:
        """
        预处理症状文本
        
        Args:
            symptoms (List[str]): 症状列表
            
        Returns:
            torch.Tensor: 症状词ID张量
        """
        # TODO: 使用真实的分词器替换简单的哈希方法
        # 实际应用中需要使用专业的中文分词工具
        
        # 简化处理：将症状转换为词ID
        symptom_ids = []
        for symptom in symptoms:
            # 简单哈希处理（实际应用中应使用真实分词器）
            symptom_id = hash(symptom) % self.symptom_vocab_size
            symptom_ids.append(symptom_id)
        
        # 转换为张量并添加批次维度
        symptom_tokens = torch.tensor(symptom_ids, dtype=torch.long).unsqueeze(0)
        return symptom_tokens.to(self.device)
    
    def _prepare_kg_data(self, kg_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        准备知识图谱数据
        
        Args:
            kg_data (Dict[str, Any]): 知识图谱数据
            
        Returns:
            Dict[str, torch.Tensor]: 张量格式的知识图谱数据
        """
        # 提取节点信息
        nodes = kg_data.get('nodes', [])
        edges = kg_data.get('edges', [])
        
        # 创建实体ID和类型张量
        entity_ids = []
        entity_types = []
        
        for node in nodes:
            entity_ids.append(node.get('id', 0))
            entity_type = node.get('type', 'Herb')
            # 将实体类型转换为索引
            type_index = {'Herb': 0, 'Symptom': 1, 'Disease': 2, 'Effect': 3}.get(entity_type, 0)
            entity_types.append(type_index)
        
        # 创建关系ID张量
        relation_ids = [edge.get('relation_type', 0) for edge in edges]
        
        # 创建边索引张量
        edge_index = [[edge.get('source', 0) for edge in edges],
                      [edge.get('target', 0) for edge in edges]]
        
        # 转换为张量
        return {
            'entity_ids': torch.tensor(entity_ids, dtype=torch.long).to(self.device),
            'entity_types': torch.tensor(entity_types, dtype=torch.long).to(self.device),
            'relation_ids': torch.tensor(relation_ids, dtype=torch.long).to(self.device),
            'edge_index': torch.tensor(edge_index, dtype=torch.long).to(self.device)
        }
    
    def _postprocess_recommendations(self, 
                                   outputs: Dict[str, torch.Tensor],
                                   kg_data: Dict[str, Any],
                                   top_k: int) -> List[Dict[str, Any]]:
        """
        后处理推荐结果
        
        Args:
            outputs (Dict[str, torch.Tensor]): 模型输出
            kg_data (Dict[str, Any]): 知识图谱数据
            top_k (int): 返回前k个推荐
            
        Returns:
            List[Dict[str, Any]]: 推荐结果列表
        """
        # 获取处方得分
        prescription_scores = outputs['prescription_scores']  # (batch_size, max_prescription_length)
        herb_logits = outputs['herb_logits']  # (batch_size, max_prescription_length, herb_vocab_size)
        
        # 对于每个处方位置，获取得分最高的药材
        recommendations = []
        
        # 获取所有药材节点
        herb_nodes = [node for node in kg_data.get('nodes', []) if node.get('type') == 'Herb']
        
        for batch_idx in range(prescription_scores.size(0)):
            batch_recommendations = []
            
            # 对每个处方位置，选择得分最高的药材
            for pos in range(min(top_k, prescription_scores.size(1))):
                # 获取当前位置的药材概率分布
                herb_probs = F.softmax(herb_logits[batch_idx, pos, :], dim=-1)
                
                # 获取top-k药材
                top_herbs = torch.topk(herb_probs, min(top_k, len(herb_nodes)), dim=-1)
                
                # 转换为推荐列表
                for i in range(len(top_herbs.indices)):
                    herb_idx = top_herbs.indices[i].item()
                    score = top_herbs.values[i].item()
                    
                    # 获取药材信息
                    if herb_idx < len(herb_nodes):
                        herb_info = herb_nodes[herb_idx]
                        batch_recommendations.append({
                            'position': pos,
                            'herb_id': herb_info.get('id'),
                            'herb_name': herb_info.get('name', f'Herb_{herb_idx}'),
                            'score': score
                        })
            
            recommendations.append(batch_recommendations)
        
        return recommendations[0] if recommendations else []


class TCMKGQueryInterface:
    """
    中药知识图谱查询接口
    用于从Neo4j数据库查询相关信息
    """
    
    def __init__(self, neo4j_uri: str, username: str, password: str):
        """
        初始化查询接口
        
        Args:
            neo4j_uri (str): Neo4j数据库URI
            username (str): 用户名
            password (str): 密码
        """
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(username, password))
    
    def query_kg_for_symptoms(self, symptoms: List[str]) -> Dict[str, Any]:
        """
        根据症状查询相关知识图谱信息
        
        Args:
            symptoms (List[str]): 症状列表
            
        Returns:
            Dict[str, Any]: 知识图谱数据
        """
        # 查询Neo4j数据库
        try:
            with self.driver.session() as session:
                # 查询与症状相关的实体和关系
                query = """
                UNWIND $symptoms AS symptom_name
                MATCH (s:Symptom {name: symptom_name})
                OPTIONAL MATCH (s)-[r]-(e)
                RETURN s, r, e
                """
                
                result = session.run(query, symptoms=symptoms)
                
                # 处理查询结果
                nodes = []
                edges = []
                
                for record in result:
                    # 添加症状节点
                    symptom_node = record['s']
                    if symptom_node:
                        nodes.append({
                            'id': symptom_node.id,
                            'name': symptom_node['name'],
                            'type': 'Symptom'
                        })
                    
                    # 添加相关实体节点
                    entity_node = record['e']
                    if entity_node:
                        nodes.append({
                            'id': entity_node.id,
                            'name': entity_node['name'],
                            'type': list(entity_node.labels)[0] if entity_node.labels else 'Entity'
                        })
                    
                    # 添加关系
                    relation = record['r']
                    if relation and symptom_node and entity_node:
                        edges.append({
                            'source': symptom_node.id,
                            'target': entity_node.id,
                            'relation_type': relation.type
                        })
                
                return {
                    'nodes': nodes,
                    'edges': edges
                }
        except Exception as e:
            print(f"查询知识图谱时出错: {e}")
            # 在生产环境中应该抛出异常而不是返回模拟数据
            raise
    
    def _get_dummy_kg_data(self, symptoms: List[str]) -> Dict[str, Any]:
        """
        获取模拟的知识图谱数据
        
        Args:
            symptoms (List[str]): 症状列表
            
        Returns:
            Dict[str, Any]: 模拟的知识图谱数据
        """
        # 生成模拟数据
        nodes = []
        edges = []
        
        # 添加症状节点
        for i, symptom in enumerate(symptoms):
            nodes.append({
                'id': i,
                'name': symptom,
                'type': 'Symptom'
            })
        
        # 添加药材节点
        herb_names = ['人参', '黄芪', '当归', '川芎', '白芍', '熟地黄', '茯苓', '甘草']
        for i, herb_name in enumerate(herb_names):
            nodes.append({
                'id': len(symptoms) + i,
                'name': herb_name,
                'type': 'Herb'
            })
        
        # 添加关系
        for i, symptom in enumerate(symptoms):
            # 每个症状连接到2-3个药材
            connected_herbs = np.random.choice(
                range(len(symptoms), len(symptoms) + len(herb_names)), 
                size=np.random.randint(2, 4), 
                replace=False
            )
            
            for herb_idx in connected_herbs:
                edges.append({
                    'source': i,
                    'target': herb_idx,
                    'relation_type': 'TREATS'  # 简化的关系类型
                })
        
        return {
            'nodes': nodes,
            'edges': edges
        }
    
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()


# 测试代码
def test_inference_module():
    """测试推理模块"""
    # 测试参数
    entity_type_sizes = {
        'Herb': 1000,
        'Symptom': 500,
        'Disease': 300,
        'Effect': 200
    }
    relation_type_size = 10
    embedding_dim = 128
    gnn_output_dim = 128
    herb_vocab_size = 1000
    symptom_vocab_size = 5000
    
    # 创建推理模块
    print("创建推理模块...")
    inference_module = TCMGNNInferenceModule(
        entity_type_sizes=entity_type_sizes,
        relation_type_size=relation_type_size,
        embedding_dim=embedding_dim,
        gnn_output_dim=gnn_output_dim,
        herb_vocab_size=herb_vocab_size,
        symptom_vocab_size=symptom_vocab_size,
        max_prescription_length=15
    )
    
    # 创建模拟输入数据
    symptom_tokens = torch.randint(0, symptom_vocab_size, (1, 10))  # 1个批次，10个症状词
    kg_entity_ids = torch.randint(0, 1000, (100,))  # 100个实体
    kg_entity_types = torch.randint(0, 4, (100,))   # 4种实体类型
    kg_relation_ids = torch.randint(0, 10, (200,))  # 200个关系
    kg_edge_index = torch.randint(0, 100, (2, 200)) # 200条边
    
    # 测试前向传播
    print("测试前向传播...")
    outputs = inference_module(
        symptom_tokens=symptom_tokens,
        kg_entity_ids=kg_entity_ids,
        kg_entity_types=kg_entity_types,
        kg_relation_ids=kg_relation_ids,
        kg_edge_index=kg_edge_index
    )
    
    print(f"处方得分形状: {outputs['prescription_scores'].shape}")
    print(f"药材logits形状: {outputs['herb_logits'].shape}")
    print(f"症状表示形状: {outputs['symptom_representation'].shape}")
    
    # 测试处方推荐
    print("\n测试处方推荐...")
    symptoms = ["头痛", "发热", "咳嗽"]
    
    # 模拟知识图谱数据
    kg_data = {
        'nodes': [
            {'id': 0, 'name': '头痛', 'type': 'Symptom'},
            {'id': 1, 'name': '发热', 'type': 'Symptom'},
            {'id': 2, 'name': '咳嗽', 'type': 'Symptom'},
            {'id': 3, 'name': '人参', 'type': 'Herb'},
            {'id': 4, 'name': '黄芪', 'type': 'Herb'},
            {'id': 5, 'name': '当归', 'type': 'Herb'}
        ],
        'edges': [
            {'source': 0, 'target': 3, 'relation_type': 'TREATS'},
            {'source': 1, 'target': 4, 'relation_type': 'TREATS'},
            {'source': 2, 'target': 5, 'relation_type': 'TREATS'}
        ]
    }
    
    recommendations = inference_module.recommend_prescription(symptoms, kg_data, top_k=5)
    print(f"推荐处方数量: {len(recommendations)}")
    for i, rec in enumerate(recommendations[:3]):
        print(f"推荐 {i+1}: {rec['herb_name']} (得分: {rec['score']:.4f})")


if __name__ == "__main__":
    test_inference_module()