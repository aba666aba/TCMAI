import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import logging
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

# 创建logger实例
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 导入模型组件
from tcm_complete_gnn_model_fixed import TCMGNNModel
from tcm_gnn_training import TCMRankingLoss, collate_fn

class TCMKnowledgeGraphDataset(Dataset):
    """
    中药知识图谱数据集类
    用于训练基于知识图谱的中药处方推荐模型
    """
    
    def __init__(self, data_file: str, max_symptom_length: int = 20):
        """
        初始化数据集
        
        Args:
            data_file (str): 清洗后的数据文件路径
            max_symptom_length (int): 最大症状序列长度
        """
        self.max_symptom_length = max_symptom_length
        # 加载数据
        self.data = self._load_data(data_file)
        # 创建映射
        self._create_mappings()
        # 处理数据
        self.processed_data = self._process_data()
    
    def _load_data(self, data_file: str) -> Dict[str, Any]:
        """
        加载清洗后的数据
        
        Args:
            data_file (str): 数据文件路径
            
        Returns:
            Dict[str, Any]: 清洗后的数据
        """
        logger.info(f"加载清洗后的数据: {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"成功加载数据，包含 {len(data.get('herbs', []))} 种药材")
        return data
    
    def _create_mappings(self):
        """
        创建词汇表映射
        """
        # 创建药材映射
        self.herb_list = [herb['name'] for herb in self.data['herbs']]
        self.herb_to_id = {herb: idx for idx, herb in enumerate(self.herb_list)}
        self.id_to_herb = {idx: herb for herb, idx in self.herb_to_id.items()}
        
        # 创建症状映射
        self.symptom_list = [symptom['name'] for symptom in self.data['symptoms']]
        self.symptom_to_id = {symptom: idx for idx, symptom in enumerate(self.symptom_list)}
        self.id_to_symptom = {idx: symptom for symptom, idx in self.symptom_to_id.items()}
        
        # 创建疾病映射
        self.disease_list = [disease['name'] for disease in self.data['diseases']]
        self.disease_to_id = {disease: idx for idx, disease in enumerate(self.disease_list)}
        self.id_to_disease = {idx: disease for disease, idx in self.disease_to_id.items()}
        
        logger.info(f"创建词汇表映射完成: {len(self.herb_list)} 种药材, {len(self.symptom_list)} 种症状, {len(self.disease_list)} 种疾病")
    
    def _process_data(self) -> List[Dict[str, Any]]:
        """
        处理数据，生成训练样本
        
        Returns:
            List[Dict[str, Any]]: 处理后的训练数据列表
        """
        processed_data = []
        
        # 为了简化示例，我们基于药材的主治信息生成训练样本
        # 在实际应用中，这应该基于真实的处方数据
        for herb in self.data['herbs']:
            indication = herb.get('indication', '')
            if indication:
                # 从主治信息中提取相关症状
                related_symptoms = []
                for symptom in self.symptom_list:
                    if symptom in indication:
                        related_symptoms.append(symptom)
                
                # 为每个相关症状创建一个训练样本
                for symptom in related_symptoms:
                    # 正样本：该药材
                    positive_prescription = [herb['name']]
                    
                    # 负样本：随机选择其他药材
                    import random
                    other_herbs = [h for h in self.herb_list if h != herb['name']]
                    negative_prescription = random.sample(other_herbs, min(5, len(other_herbs)))
                    
                    # 症状ID
                    symptom_id = self.symptom_to_id.get(symptom, 0)
                    symptoms = [symptom_id]
                    
                    processed_data.append({
                        'symptoms': symptoms,
                        'positive_prescription': positive_prescription,
                        'negative_prescription': negative_prescription,
                        'symptom_text': symptom
                    })
        
        logger.info(f"处理完成，生成 {len(processed_data)} 个训练样本")
        return processed_data
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        """
        获取单个训练样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            Dict[str, Any]: 训练样本
        """
        sample = self.processed_data[idx]
        
        # 填充或截断症状序列
        symptoms = sample['symptoms']
        if len(symptoms) < self.max_symptom_length:
            symptoms = np.pad(symptoms, (0, self.max_symptom_length - len(symptoms)), mode='constant')
        else:
            symptoms = symptoms[:self.max_symptom_length]
        
        # 返回样本数据
        return {
            'symptoms': torch.tensor(symptoms, dtype=torch.long),
            'positive_prescription': sample['positive_prescription'],
            'negative_prescription': sample['negative_prescription'],
            'symptom_text': sample['symptom_text']
        }

class TCMGraphDataBuilder:
    """
    中药知识图谱数据构建器
    从清洗后的数据构建PyG图数据对象
    """
    
    def __init__(self, data: Dict[str, Any]):
        """
        初始化数据构建器
        
        Args:
            data (Dict[str, Any]): 清洗后的数据
        """
        self.data = data
        self._create_mappings()
    
    def _create_mappings(self):
        """
        创建实体类型映射
        """
        # 实体类型映射
        self.entity_types = ['Herb', 'Symptom', 'Disease', 'Effect']
        self.entity_type_to_id = {etype: idx for idx, etype in enumerate(self.entity_types)}
        self.id_to_entity_type = {idx: etype for etype, idx in self.entity_type_to_id.items()}
        
        # 关系类型映射
        self.relation_types = ['治疗', '主治', '功效', '相关']
        self.relation_type_to_id = {rtype: idx for idx, rtype in enumerate(self.relation_types)}
        self.id_to_relation_type = {idx: rtype for rtype, idx in self.relation_type_to_id.items()}
    
    def build_graph_data(self) -> Data:
        """
        构建PyG图数据对象
        
        Returns:
            Data: PyG图数据对象
        """
        logger.info("开始构建知识图谱...")
        
        # 收集所有实体
        entities = []
        entity_types = []
        
        # 添加药材实体
        herb_entities = [(herb['name'], 'Herb') for herb in self.data['herbs']]
        entities.extend(herb_entities)
        
        # 添加症状实体
        symptom_entities = [(symptom['name'], 'Symptom') for symptom in self.data['symptoms']]
        entities.extend(symptom_entities)
        
        # 添加疾病实体
        disease_entities = [(disease['name'], 'Disease') for disease in self.data['diseases']]
        entities.extend(disease_entities)
        
        # 添加功效实体
        effect_entities = [(effect['name'], 'Effect') for effect in self.data['effects']]
        entities.extend(effect_entities)
        
        # 创建实体ID映射
        entity_to_id = {entity[0]: idx for idx, entity in enumerate(entities)}
        id_to_entity = {idx: entity for entity, idx in entity_to_id.items()}
        
        # 创建边列表
        edge_index = []
        edge_type = []
        
        # 添加药材-主治症状关系
        for herb in self.data['herbs']:
            indication = herb.get('indication', '')
            herb_id = entity_to_id[herb['name']]
            for symptom in self.data['symptoms']:
                if symptom['name'] in indication:
                    symptom_id = entity_to_id[symptom['name']]
                    edge_index.append([herb_id, symptom_id])
                    edge_type.append(self.relation_type_to_id['主治'])
        
        # 添加药材-功效关系
        for herb in self.data['herbs']:
            effect_text = herb.get('effect', '')
            herb_id = entity_to_id[herb['name']]
            for effect in self.data['effects']:
                if effect['name'] in effect_text:
                    effect_id = entity_to_id[effect['name']]
                    edge_index.append([herb_id, effect_id])
                    edge_type.append(self.relation_type_to_id['功效'])
        
        # 转换为张量
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long) if edge_type else torch.empty((0,), dtype=torch.long)
        
        # 创建实体类型张量
        entity_type_ids = [self.entity_type_to_id[etype] for _, etype in entities]
        entity_type_tensor = torch.tensor(entity_type_ids, dtype=torch.long)
        
        # 创建实体ID张量（这里简化处理，实际应用中可能需要更复杂的ID系统）
        entity_ids = list(range(len(entities)))
        entity_id_tensor = torch.tensor(entity_ids, dtype=torch.long)
        
        # 创建图数据对象
        graph_data = Data(
            x=torch.randn(len(entities), 128),  # 简化处理，实际应用中应该使用真实的实体嵌入
            edge_index=edge_index,
            edge_type=edge_type,
            node_ids=entity_id_tensor,
            node_types=entity_type_tensor
        )
        
        logger.info(f"知识图谱构建完成: {graph_data.num_nodes} 个节点, {graph_data.num_edges} 条边")
        return graph_data

class TCMGNNTrainer:
    """
    中药知识图谱模型训练器
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 device: torch.device, 
                 graph_data: Data,
                 learning_rate: float = 0.001):
        """
        初始化训练器
        
        Args:
            model (nn.Module): 要训练的模型
            device (torch.device): 训练设备
            graph_data (Data): 图数据
            learning_rate (float): 学习率
        """
        self.model = model
        self.device = device
        self.graph_data = graph_data.to(device)
        self.model.to(device)
        
        # 设置优化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # 设置学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.8)
        
        # 设置损失函数
        self.criterion = TCMRankingLoss(margin=1.0)
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.best_model_path = "best_tcm_gnn_model.pth"
    
    def train_epoch(self, 
                    train_loader: DataLoader) -> float:
        """
        训练一个epoch
        
        Args:
            train_loader (DataLoader): 训练数据加载器
            
        Returns:
            float: 平均损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # 将数据移动到设备
                symptoms = batch['symptoms'].to(self.device)
                positive_prescriptions = batch['positive_prescription']
                negative_prescriptions = batch['negative_prescription']
                
                # 清零梯度
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(
                    symptom_tokens=symptoms,
                    kg_entity_ids=self.graph_data.node_ids,
                    kg_entity_types=self.graph_data.node_types,
                    kg_edge_index=self.graph_data.edge_index
                )
                
                # 计算损失
                batch_loss = 0.0
                valid_samples = 0
                
                for i in range(len(positive_prescriptions)):
                    try:
                        # 转换处方为ID
                        pos_prescription_ids = [self.model.herb_to_id.get(herb, 0) for herb in positive_prescriptions[i]]
                        neg_prescription_ids = [self.model.herb_to_id.get(herb, 0) for herb in negative_prescriptions[i]]
                        
                        # 过滤无效ID
                        pos_prescription_ids = [pid for pid in pos_prescription_ids if pid < self.model.herb_vocab_size]
                        neg_prescription_ids = [nid for nid in neg_prescription_ids if nid < self.model.herb_vocab_size]
                        
                        if len(pos_prescription_ids) == 0:
                            continue
                        
                        # 转换为张量
                        pos_prescription_tensor = torch.tensor(pos_prescription_ids, dtype=torch.long, device=self.device)
                        neg_prescription_tensor = torch.tensor(neg_prescription_ids, dtype=torch.long, device=self.device)
                        
                        # 确保所有ID都在有效范围内
                        pos_prescription_tensor = torch.clamp(pos_prescription_tensor, 0, self.model.herb_vocab_size - 1)
                        neg_prescription_tensor = torch.clamp(neg_prescription_tensor, 0, self.model.herb_vocab_size - 1)
                        
                        # 计算损失
                        sample_outputs = {k: v[i:i+1] if isinstance(v, torch.Tensor) and v.dim() > 0 else v for k, v in outputs.items()}
                        sample_loss = self.model.compute_loss(
                            sample_outputs,
                            pos_prescription_tensor.unsqueeze(0),
                            neg_prescription_tensor.unsqueeze(0)
                        )
                        
                        batch_loss += sample_loss
                        valid_samples += 1
                    except Exception as e:
                        logger.warning(f"处理样本 {i} 时出错: {e}")
                        continue
                
                if valid_samples > 0:
                    batch_loss = batch_loss / valid_samples
                    
                    # 反向传播
                    batch_loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # 更新参数
                    self.optimizer.step()
                    
                    total_loss += batch_loss.item()
                
                # 打印训练进度
                if batch_idx % 10 == 0:
                    logger.info(f'Batch {batch_idx}/{num_batches}, Loss: {batch_loss.item() if valid_samples > 0 else 0.0:.4f}')
                    
            except Exception as e:
                logger.error(f"处理批次 {batch_idx} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 更新学习率
        self.scheduler.step()
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        验证模型
        
        Args:
            val_loader (DataLoader): 验证数据加载器
            
        Returns:
            float: 平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    # 将数据移动到设备
                    symptoms = batch['symptoms'].to(self.device)
                    positive_prescriptions = batch['positive_prescription']
                    negative_prescriptions = batch['negative_prescription']
                    
                    # 前向传播
                    outputs = self.model(
                        symptom_tokens=symptoms,
                        kg_entity_ids=self.graph_data.node_ids,
                        kg_entity_types=self.graph_data.node_types,
                        kg_edge_index=self.graph_data.edge_index
                    )
                    
                    # 计算损失
                    batch_loss = 0.0
                    valid_samples = 0
                    
                    for i in range(len(positive_prescriptions)):
                        try:
                            # 转换处方为ID
                            pos_prescription_ids = [self.model.herb_to_id.get(herb, 0) for herb in positive_prescriptions[i]]
                            neg_prescription_ids = [self.model.herb_to_id.get(herb, 0) for herb in negative_prescriptions[i]]
                            
                            # 过滤无效ID
                            pos_prescription_ids = [pid for pid in pos_prescription_ids if pid < self.model.herb_vocab_size]
                            neg_prescription_ids = [nid for nid in neg_prescription_ids if nid < self.model.herb_vocab_size]
                            
                            if len(pos_prescription_ids) == 0:
                                continue
                            
                            # 转换为张量
                            pos_prescription_tensor = torch.tensor(pos_prescription_ids, dtype=torch.long, device=self.device)
                            neg_prescription_tensor = torch.tensor(neg_prescription_ids, dtype=torch.long, device=self.device)
                            
                            # 确保所有ID都在有效范围内
                            pos_prescription_tensor = torch.clamp(pos_prescription_tensor, 0, self.model.herb_vocab_size - 1)
                            neg_prescription_tensor = torch.clamp(neg_prescription_tensor, 0, self.model.herb_vocab_size - 1)
                            
                            # 计算损失
                            sample_outputs = {k: v[i:i+1] if isinstance(v, torch.Tensor) and v.dim() > 0 else v for k, v in outputs.items()}
                            sample_loss = self.model.compute_loss(
                                sample_outputs,
                                pos_prescription_tensor.unsqueeze(0),
                                neg_prescription_tensor.unsqueeze(0)
                            )
                            
                            batch_loss += sample_loss
                            valid_samples += 1
                        except Exception as e:
                            logger.warning(f"验证时处理样本 {i} 时出错: {e}")
                            continue
                    
                    if valid_samples > 0:
                        batch_loss = batch_loss / valid_samples
                        total_loss += batch_loss.item()
                        
                except Exception as e:
                    logger.error(f"验证时处理批次 {batch_idx} 时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path (str): 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }, path)
        logger.info(f"模型已保存到 {path}")
    
    def load_model(self, path: str):
        """
        加载模型
        
        Args:
            path (str): 加载路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"模型已从 {path} 加载")

def train_model(data_file: str, 
                model_save_path: str = "tcm_gnn_model.pth",
                epochs: int = 50,
                batch_size: int = 32,
                learning_rate: float = 0.001,
                test_split: float = 0.2):
    """
    训练中药知识图谱模型
    
    Args:
        data_file (str): 清洗后的数据文件路径
        model_save_path (str): 模型保存路径
        epochs (int): 训练轮数
        batch_size (int): 批次大小
        learning_rate (float): 学习率
        test_split (float): 测试集比例
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    logger.info("加载数据...")
    dataset = TCMKnowledgeGraphDataset(data_file)
    
    # 划分训练集和验证集
    train_indices, val_indices = train_test_split(
        list(range(len(dataset))), 
        test_size=test_split, 
        random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 构建图数据
    logger.info("构建知识图谱...")
    with open(data_file, 'r', encoding='utf-8') as f:
        clean_data = json.load(f)
    
    graph_builder = TCMGraphDataBuilder(clean_data)
    graph_data = graph_builder.build_graph_data()
    
    # 创建模型
    logger.info("创建模型...")
    entity_type_sizes = {
        'Herb': len(dataset.herb_list),
        'Symptom': len(dataset.symptom_list),
        'Disease': len(dataset.disease_list),
        'Effect': len(dataset.data['effects'])
    }
    
    # 添加 herb_to_id 到模型中
    model = TCMGNNModel(
        entity_type_sizes=entity_type_sizes,
        relation_type_size=len(graph_builder.relation_types),
        embedding_dim=128,
        gnn_hidden_dim=256,
        gnn_output_dim=128,
        herb_vocab_size=len(dataset.herb_list),
        symptom_vocab_size=len(dataset.symptom_list),
        max_prescription_length=15,
        num_gnn_layers=3,
        gnn_type='sage',
        dropout=0.1
    )
    
    # 添加 herb_to_id 到模型中，以便在训练时使用
    model.herb_to_id = dataset.herb_to_id
    
    # 创建训练器
    trainer = TCMGNNTrainer(model, device, graph_data, learning_rate)
    
    # 训练循环
    logger.info("开始训练...")
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # 训练
        train_loss = trainer.train_epoch(train_loader)
        
        # 验证
        val_loss = trainer.validate(val_loader)
        
        logger.info(f"Epoch {epoch+1} - 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            trainer.save_model(model_save_path)
            logger.info(f"保存最佳模型，验证损失: {val_loss:.4f}")
    
    logger.info("训练完成!")

# 主函数
def main():
    """
    主函数
    """
    # 设置参数
    data_file = "tcm_cleaned_data.json"  # 清洗后的数据文件
    model_save_path = "tcm_gnn_model.pth"
    epochs = 50
    batch_size = 32
    learning_rate = 0.001
    
    # 检查数据文件是否存在
    if not os.path.exists(data_file):
        logger.error(f"数据文件 {data_file} 不存在，请先运行数据清洗脚本")
        return
    
    # 训练模型
    train_model(
        data_file=data_file,
        model_save_path=model_save_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

if __name__ == "__main__":
    main()