#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CUDA错误调试脚本
用于诊断和解决CUDA device-side assert triggered错误
"""

import torch
import torch.nn as nn
import os
import sys

# 设置环境变量以便更好地调试CUDA错误
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def check_cuda_setup():
    """检查CUDA设置"""
    print("CUDA设置检查:")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA设备数量: {torch.cuda.device_count()}")
        print(f"  当前设备: {torch.cuda.current_device()}")
        print(f"  设备名称: {torch.cuda.get_device_name()}")
    else:
        print("  CUDA不可用，将使用CPU")
    print()

def test_embedding_lookup():
    """测试嵌入查找"""
    print("测试嵌入查找...")
    
    # 创建一个小的嵌入层
    vocab_size = 100
    embedding_dim = 32
    embedding = nn.Embedding(vocab_size, embedding_dim)
    
    # 测试正常范围内的索引
    indices = torch.tensor([0, 10, 50, 99])
    try:
        result = embedding(indices)
        print(f"  正常索引查找成功: {result.shape}")
    except Exception as e:
        print(f"  正常索引查找失败: {e}")
    
    # 测试超出范围的索引
    bad_indices = torch.tensor([0, 10, 150, 99])  # 150超出范围
    try:
        result = embedding(bad_indices)
        print(f"  超出范围索引查找成功(意外): {result.shape}")
    except Exception as e:
        print(f"  超出范围索引查找失败(预期): {e}")
    
    # 测试使用clamp处理超出范围的索引
    clamped_indices = torch.clamp(bad_indices, 0, vocab_size - 1)
    try:
        result = embedding(clamped_indices)
        print(f"  clamp处理后索引查找成功: {result.shape}")
    except Exception as e:
        print(f"  clamp处理后索引查找失败: {e}")
    
    print()

def test_tensor_operations():
    """测试张量操作"""
    print("测试张量操作...")
    
    # 创建测试张量
    a = torch.randn(5, 3)
    b = torch.randn(3, 4)
    
    try:
        result = torch.matmul(a, b)
        print(f"  矩阵乘法成功: {result.shape}")
    except Exception as e:
        print(f"  矩阵乘法失败: {e}")
    
    # 测试不同设备上的操作
    if torch.cuda.is_available():
        try:
            a_gpu = a.cuda()
            b_gpu = b.cuda()
            result_gpu = torch.matmul(a_gpu, b_gpu)
            print(f"  GPU矩阵乘法成功: {result_gpu.shape}")
        except Exception as e:
            print(f"  GPU矩阵乘法失败: {e}")
    
    print()

def test_attention_mechanism():
    """测试注意力机制"""
    print("测试注意力机制...")
    
    # 创建测试数据
    batch_size = 2
    seq_len = 5
    num_entities = 10
    feature_dim = 32
    
    query = torch.randn(batch_size, seq_len, feature_dim)
    key = torch.randn(batch_size, num_entities, feature_dim)
    value = torch.randn(batch_size, num_entities, feature_dim)
    
    # 创建注意力层
    attention = nn.MultiheadAttention(feature_dim, num_heads=4, batch_first=True)
    
    try:
        output, weights = attention(query, key, value)
        print(f"  注意力机制成功: output={output.shape}, weights={weights.shape}")
    except Exception as e:
        print(f"  注意力机制失败: {e}")
    
    print()

def debug_model_forward_pass():
    """调试模型前向传播"""
    print("调试模型前向传播...")
    
    try:
        # 导入模型组件
        from tcm_complete_gnn_model import TCMGNNModel
        
        # 创建简化版模型参数
        entity_type_sizes = {
            'Herb': 50,
            'Symptom': 30,
            'Disease': 20,
            'Effect': 10
        }
        relation_type_size = 5
        embedding_dim = 32
        gnn_hidden_dim = 64
        gnn_output_dim = 32
        herb_vocab_size = 50
        symptom_vocab_size = 1000
        max_prescription_length = 5
        
        # 创建模型
        print("  创建模型...")
        model = TCMGNNModel(
            entity_type_sizes=entity_type_sizes,
            relation_type_size=relation_type_size,
            embedding_dim=embedding_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim,
            herb_vocab_size=herb_vocab_size,
            symptom_vocab_size=symptom_vocab_size,
            max_prescription_length=max_prescription_length,
            num_gnn_layers=2,
            gnn_type='sage',
            dropout=0.1
        )
        
        print("  模型创建成功")
        
        # 创建测试输入
        print("  创建测试输入...")
        batch_size = 2
        seq_len = 5
        num_entities = 50
        num_edges = 100
        
        symptom_tokens = torch.randint(0, symptom_vocab_size, (batch_size, seq_len))
        kg_entity_ids = torch.randint(0, 100, (num_entities,))
        kg_entity_types = torch.randint(0, 4, (num_entities,))
        kg_edge_index = torch.randint(0, num_entities, (2, num_edges))
        
        print(f"    症状tokens: {symptom_tokens.shape}")
        print(f"    实体IDs: {kg_entity_ids.shape}")
        print(f"    实体类型: {kg_entity_types.shape}")
        print(f"    边索引: {kg_edge_index.shape}")
        
        # 设备配置
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  使用设备: {device}")
        
        # 将模型和数据移动到设备
        model = model.to(device)
        symptom_tokens = symptom_tokens.to(device)
        kg_entity_ids = kg_entity_ids.to(device)
        kg_entity_types = kg_entity_types.to(device)
        kg_edge_index = kg_edge_index.to(device)
        
        # 前向传播
        print("  执行前向传播...")
        with torch.no_grad():
            outputs = model(
                symptom_tokens=symptom_tokens,
                kg_entity_ids=kg_entity_ids,
                kg_entity_types=kg_entity_types,
                kg_edge_index=kg_edge_index
            )
        
        print("  前向传播成功")
        print(f"  输出键: {list(outputs.keys())}")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: {value.shape}")
        
    except Exception as e:
        print(f"  模型前向传播失败: {e}")
        import traceback
        traceback.print_exc()
    
    print()

def main():
    """主函数"""
    print("CUDA错误调试脚本")
    print("=" * 50)
    
    # 检查CUDA设置
    check_cuda_setup()
    
    # 测试嵌入查找
    test_embedding_lookup()
    
    # 测试张量操作
    test_tensor_operations()
    
    # 测试注意力机制
    test_attention_mechanism()
    
    # 调试模型前向传播
    debug_model_forward_pass()
    
    print("调试完成")

if __name__ == "__main__":
    main()