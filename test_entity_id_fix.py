import torch
import logging

# 配置日志
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_entity_id_fix():
    """测试实体ID修复功能"""
    # 导入修复后的模型
    from tcm_complete_gnn_model_fixed import TCMGNNModel
    from tcm_gnn_input_fixed import TCMGNNInputLayer
    
    # 模拟实体类型大小
    entity_type_sizes = {
        'Herb': 100,
        'Symptom': 50,  # 故意设置较小的Symptom数量，以便测试超出范围的情况
        'Disease': 30,
        'Effect': 20
    }
    
    # 测试参数
    relation_type_size = 10
    embedding_dim = 64
    symptom_vocab_size = 1000
    herb_vocab_size = 100
    max_prescription_length = 10
    
    print("创建模型...")
    # 创建模型
    model = TCMGNNModel(
        entity_type_sizes=entity_type_sizes,
        relation_type_size=relation_type_size,
        embedding_dim=embedding_dim,
        gnn_hidden_dim=128,
        gnn_output_dim=64,
        herb_vocab_size=herb_vocab_size,
        symptom_vocab_size=symptom_vocab_size,
        max_prescription_length=max_prescription_length,
        num_gnn_layers=2,
        gnn_type='sage',
        dropout=0.1
    )
    
    # 创建测试数据
    batch_size = 2
    seq_len = 5
    num_entities = 20
    
    # 创建症状tokens
    symptom_tokens = torch.randint(0, symptom_vocab_size, (batch_size, seq_len))
    
    # 创建实体ID和类型，包含超出范围的ID
    kg_entity_ids = torch.tensor([1340, 45, 200, 15, 300, 25, 180, 18, 250, 20, 
                                  150, 12, 100, 8, 50, 5, 20, 2, 10, 1])
    kg_entity_types = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                                    1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    
    # 创建边索引
    kg_edge_index = torch.randint(0, num_entities, (2, 30))
    
    print("测试实体ID范围检查...")
    print(f"Symptom类型实体数量: {entity_type_sizes['Symptom']}")
    print("实体ID和类型:")
    for i, (eid, etype) in enumerate(zip(kg_entity_ids, kg_entity_types)):
        entity_type_name = ['Herb', 'Symptom', 'Disease', 'Effect'][etype.item()]
        print(f"  实体 {i}: ID={eid.item()}, 类型={entity_type_name}")
    
    # 测试模型前向传播
    print("\n测试模型前向传播...")
    try:
        outputs = model(
            symptom_tokens=symptom_tokens,
            kg_entity_ids=kg_entity_ids,
            kg_entity_types=kg_entity_types,
            kg_edge_index=kg_edge_index
        )
        print("模型前向传播成功!")
        print(f"处方得分形状: {outputs['prescription_scores'].shape}")
        
    except Exception as e:
        print(f"模型前向传播失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_entity_id_fix()