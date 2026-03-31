# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0

"""
中医药推荐系统 - 训练脚本

训练模型以根据症状推荐药材和方剂
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tcm_recommender.data_loader import TCMDataLoader
from tcm_recommender.model import create_model, SafetyChecker

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TCMDataset(Dataset):
    """中医药推荐数据集"""
    
    def __init__(
        self,
        symptoms_list: List[List[int]],
        herbs_list: List[int],
        formulas_list: List[int],
        max_symptoms: int = 20
    ):
        self.symptoms_list = symptoms_list
        self.herbs_list = herbs_list
        self.formulas_list = formulas_list
        self.max_symptoms = max_symptoms
    
    def __len__(self):
        return len(self.symptoms_list)
    
    def __getitem__(self, idx):
        # 获取症状序列（padding）
        symptoms = self.symptoms_list[idx][:self.max_symptoms]
        symptoms = symptoms + [0] * (self.max_symptoms - len(symptoms))
        
        return {
            'symptoms': torch.tensor(symptoms, dtype=torch.long),
            'herb': torch.tensor(self.herbs_list[idx], dtype=torch.long),
            'formula': torch.tensor(self.formulas_list[idx], dtype=torch.long)
        }


class Trainer:
    """训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.device = device
        
        # 损失函数
        self.herb_criterion = nn.CrossEntropyLoss()
        self.formula_criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'herb_loss': [],
            'formula_loss': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        herb_loss_sum = 0.0
        formula_loss_sum = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc='训练')
        for batch in pbar:
            # 准备数据
            symptoms = batch['symptoms'].to(self.device)
            herbs = batch['herb'].to(self.device)
            formulas = batch['formula'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            herb_scores, formula_scores = self.model(symptoms)
            
            # 计算损失（只计算非零标签）
            herb_mask = herbs != 0
            formula_mask = formulas != 0
            
            loss = 0.0
            if herb_mask.sum() > 0:
                herb_loss = self.herb_criterion(
                    herb_scores[herb_mask],
                    herbs[herb_mask]
                )
                loss += herb_loss
                herb_loss_sum += herb_loss.item()
            
            if formula_mask.sum() > 0:
                formula_loss = self.formula_criterion(
                    formula_scores[formula_mask],
                    formulas[formula_mask]
                )
                loss += formula_loss
                formula_loss_sum += formula_loss.item()
            
            # 反向传播
            if loss > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'herb': f'{herb_loss.item() if herb_mask.sum() > 0 else 0:.4f}',
                    'formula': f'{formula_loss.item() if formula_mask.sum() > 0 else 0:.4f}'
                })
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        herb_loss_sum = 0.0
        formula_loss_sum = 0.0
        num_batches = 0
        correct_herb = 0
        correct_formula = 0
        total = 0
        
        for batch in tqdm(dataloader, desc='验证'):
            symptoms = batch['symptoms'].to(self.device)
            herbs = batch['herb'].to(self.device)
            formulas = batch['formula'].to(self.device)
            
            herb_scores, formula_scores = self.model(symptoms)
            
            # 计算损失
            herb_mask = herbs != 0
            formula_mask = formulas != 0
            
            loss = 0.0
            if herb_mask.sum() > 0:
                herb_loss = self.herb_criterion(
                    herb_scores[herb_mask],
                    herbs[herb_mask]
                )
                loss += herb_loss
                herb_loss_sum += herb_loss.item()
                
                # 计算准确率
                pred_herb = herb_scores.argmax(dim=1)
                correct_herb += (pred_herb == herbs).sum().item()
            
            if formula_mask.sum() > 0:
                formula_loss = self.formula_criterion(
                    formula_scores[formula_mask],
                    formulas[formula_mask]
                )
                loss += formula_loss
                formula_loss_sum += formula_loss.item()
                
                pred_formula = formula_scores.argmax(dim=1)
                correct_formula += (pred_formula == formulas).sum().item()
            
            total += symptoms.size(0)
            
            if loss > 0:
                total_loss += loss.item()
                num_batches += 1
        
        return {
            'loss': total_loss / max(num_batches, 1),
            'herb_loss': herb_loss_sum / max(num_batches, 1),
            'formula_loss': formula_loss_sum / max(num_batches, 1),
            'herb_accuracy': correct_herb / max(total, 1),
            'formula_accuracy': correct_formula / max(total, 1)
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        save_path: str = 'best_model.pth'
    ):
        """完整训练流程"""
        logger.info(f"开始训练，共 {epochs} 个 epoch")
        logger.info(f"训练数据：{len(train_loader.dataset)} 样本")
        logger.info(f"验证数据：{len(val_loader.dataset)} 样本")
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 10
        
        for epoch in range(epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"{'='*50}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            logger.info(f"训练损失：{train_loss:.4f}")
            
            # 验证
            val_metrics = self.evaluate(val_loader)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['herb_loss'].append(val_metrics['herb_loss'])
            self.history['formula_loss'].append(val_metrics['formula_loss'])
            
            logger.info(f"验证损失：{val_metrics['loss']:.4f}")
            logger.info(f"  - 药材损失：{val_metrics['herb_loss']:.4f}")
            logger.info(f"  - 方剂损失：{val_metrics['formula_loss']:.4f}")
            logger.info(f"  - 药材准确率：{val_metrics['herb_accuracy']:.4f}")
            logger.info(f"  - 方剂准确率：{val_metrics['formula_accuracy']:.4f}")
            
            # 学习率调整
            self.scheduler.step(val_metrics['loss'])
            
            # 保存最佳模型
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_model(save_path)
                logger.info(f"✓ 保存最佳模型 (验证损失：{best_val_loss:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"早停计数：{patience_counter}/{early_stop_patience}")
            
            # 早停
            if patience_counter >= early_stop_patience:
                logger.info("触发早停，结束训练")
                break
        
        logger.info(f"\n训练完成！最佳验证损失：{best_val_loss:.4f}")
        
        # 保存训练历史
        history_path = save_path.replace('.pth', '_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"训练历史已保存到：{history_path}")
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        logger.info(f"模型已保存：{path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        logger.info(f"模型已加载：{path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='中医药推荐系统训练')
    parser.add_argument('--db', type=str, default='data/tcm_knowledge.db',
                        help='数据库路径')
    parser.add_argument('--output', type=str, default='tcm_recommender/best_model.pth',
                        help='模型输出路径')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--device', type=str, default='cpu',
                        help='训练设备 (cpu/cuda)')
    
    args = parser.parse_args()
    
    # 数据库路径
    db_path = Path(PROJECT_ROOT) / args.db
    if not db_path.exists():
        logger.error(f"数据库不存在：{db_path}")
        return
    
    logger.info("="*60)
    logger.info("  中医药智能推荐系统 - 训练程序")
    logger.info("="*60)
    
    # 加载数据
    logger.info("\n[1/4] 加载数据...")
    data_loader = TCMDataLoader(str(db_path))
    data_loader.load_all_data()
    
    # 获取训练数据
    symptoms_list, herbs_list, formulas_list = data_loader.get_training_data()
    
    # 划分训练集和验证集
    split_idx = int(len(symptoms_list) * 0.9)
    train_data = TCMDataset(
        symptoms_list[:split_idx],
        herbs_list[:split_idx],
        formulas_list[:split_idx]
    )
    val_data = TCMDataset(
        symptoms_list[split_idx:],
        herbs_list[split_idx:],
        formulas_list[split_idx:]
    )
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    
    # 创建模型
    logger.info("\n[2/4] 创建模型...")
    model = create_model(
        num_symptoms=len(data_loader.symptoms),
        num_herbs=len(data_loader.herbs),
        num_formulas=len(data_loader.formulas)
    )
    
    # 创建训练器
    logger.info("\n[3/4] 准备训练...")
    trainer = Trainer(
        model=model,
        device=args.device,
        learning_rate=args.lr
    )
    
    # 开始训练
    logger.info("\n[4/4] 开始训练...\n")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_path=args.output
    )
    
    # 测试模型
    logger.info("\n" + "="*60)
    logger.info("  测试模型")
    logger.info("="*60)
    
    # 加载最佳模型
    trainer.load_model(args.output)
    
    # 测试推荐
    test_symptoms = torch.tensor([[1, 2, 3]], dtype=torch.long)  # 乏力，食欲不振，失眠
    result = model.predict(test_symptoms, top_k=5)
    
    herb_names = data_loader.get_herb_names()
    formula_names = data_loader.get_formula_names()
    
    print("\n测试症状：乏力，食欲不振，失眠")
    print("\n推荐药材 Top-5:")
    for herb_id, score in result['herbs']:
        print(f"  {herb_names.get(herb_id, 'Unknown')}: {score:.4f}")
    
    print("\n推荐方剂 Top-5:")
    for formula_id, score in result['formulas']:
        print(f"  {formula_names.get(formula_id, 'Unknown')}: {score:.4f}")
    
    # 安全校验
    print("\n安全校验:")
    safety_checker = SafetyChecker(data_loader.incompatible_herbs)
    top_herbs = [herb_names.get(h[0], '') for h in result['herbs'][:5]]
    check_result = safety_checker.check_compatibility(top_herbs)
    
    if check_result['warnings']:
        for w in check_result['warnings']:
            print(f"  {w}")
    else:
        print("  ✓ 无配伍禁忌")
    
    data_loader.close()
    logger.info("\n训练完成！")


if __name__ == '__main__':
    main()
