#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CUDA环境和模型组件检查脚本
用于诊断CUDA错误和模型组件问题
"""

import torch
import torch.nn as nn
import os
import sys

# 设置环境变量以便更好地调试CUDA错误
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def check_cuda_environment():
    """检查CUDA环境"""
    print("CUDA环境检查:")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"  CUDA设备数量: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"  当前设备: {torch.cuda.current_device()}")
            print(f"  设备名称: {torch.cuda.get_device_name()}")
            print(f"  设备总内存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("  CUDA不可用，将使用CPU")
    print()

def test_basic_pytorch_operations():
    """测试基本的PyTorch操作"""
    print("基本PyTorch操作测试:")
    
    # 创建张量
    x = torch.randn(3, 4)
    y = torch.randn(4, 5)
    print(f"  创建张量成功: x={x.shape}, y={y.shape}")
    
    # 矩阵乘法
    z = torch.matmul(x, y)
    print(f"  矩阵乘法成功: z={z.shape}")
    
    # 如果CUDA可用，测试GPU操作
    if torch.cuda.is_available():
        try:
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = torch.matmul(x_gpu, y_gpu)
            print(f"  GPU矩阵乘法成功: z_gpu={z_gpu.shape}")
            
            # 测试GPU内存
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"  GPU已分配内存: {gpu_memory_allocated:.2f} GB")
            print(f"  GPU已保留内存: {gpu_memory_reserved:.2f} GB")
        except Exception as e:
            print(f"  GPU操作失败: {e}")
    
    print()

def test_embedding_layers():
    """测试嵌入层"""
    print("嵌入层测试:")
    
    # 创建嵌入层
    vocab_size = 100
    embedding_dim = 32
    embedding = nn.Embedding(vocab_size, embedding_dim)
    print(f"  创建嵌入层成功: vocab_size={vocab_size}, embedding_dim={embedding_dim}")
    
    # 测试正常范围内的索引
    indices = torch.tensor([0, 10, 50, 99])
    try:
        result = embedding(indices)
        print(f"  正常索引查找成功: {result.shape}")
    except Exception as e:
        print(f"  正常索引查找失败: {e}")
    
    # 测试超出范围的索引（应该会失败）
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

def test_model_components():
    """测试模型组件"""
    print("模型组件测试:")
    
    try:
        # 导入模型组件
        from tcm_gnn_input import TCMGNNInputLayer
        from tcm_gnn_components import TCMKnowledgeGraphEncoder
        from tcm_gnn_output import TCMInteractionAwareDecoder
        from tcm_complete_gnn_model import TCMGNNModel
        
        print("  导入模型组件成功")
        
        # 测试输入层
        entity_type_sizes = {
            'Herb': 50,
            'Symptom': 30,
            'Disease': 20,
            'Effect': 10
        }
        relation_type_size = 5
        embedding_dim = 32
        symptom_vocab_size = 1000
        
        input_layer = TCMGNNInputLayer(
            entity_type_sizes, 
            relation_type_size, 
            embedding_dim,
            symptom_vocab_size
        )
        print("  创建输入层成功")
        
        # 测试编码器
        kg_encoder = TCMKnowledgeGraphEncoder(
            entity_embedding_dim=embedding_dim,
            gnn_hidden_dim=64,
            gnn_output_dim=32,
            num_gnn_layers=2,
            gnn_type='sage'
        )
        print("  创建编码器成功")
        
        # 测试解码器
        decoder = TCMInteractionAwareDecoder(
            feature_dim=embedding_dim,
            herb_vocab_size=50,
            max_prescription_length=5
        )
        print("  创建解码器成功")
        
        # 测试完整模型
        model = TCMGNNModel(
            entity_type_sizes=entity_type_sizes,
            relation_type_size=relation_type_size,
            embedding_dim=embedding_dim,
            gnn_hidden_dim=64,
            gnn_output_dim=32,
            herb_vocab_size=50,
            symptom_vocab_size=symptom_vocab_size,
            max_prescription_length=5,
            num_gnn_layers=2,
            gnn_type='sage'
        )
        print("  创建完整模型成功")
        
    except Exception as e:
        print(f"  模型组件测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print()

def main():
    """主函数"""
    print("CUDA环境和模型组件检查脚本")
    print("=" * 50)
    
    # 检查CUDA环境
    check_cuda_environment()
    
    # 测试基本PyTorch操作
    test_basic_pytorch_operations()
    
    # 测试嵌入层
    test_embedding_layers()
    
    # 测试模型组件
    test_model_components()
    
    print("检查完成")

if __name__ == "__main__":
    main()