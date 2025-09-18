#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
中药处方推荐系统主应用
基于知识图谱和图神经网络的智能处方生成工具
"""

import torch
import argparse
import sys
import os
from typing import List, Dict, Any
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from tcm_data_pipeline import verify_neo4j_connection
from tcm_gnn_inference import TCMGNNInferenceModule, TCMKGQueryInterface
from tcm_neo4j_importer import TCMNeo4jImporter


class TCMPrecriptionSystem:
    """中药处方推荐系统主类"""
    
    def __init__(self, config_file: str = "config.json"):
        """
        初始化系统
        
        Args:
            config_file (str): 配置文件路径
        """
        self.config = self._load_config(config_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.kg_interface = None
        self._setup_system()
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_file (str): 配置文件路径
            
        Returns:
            Dict[str, Any]: 配置信息
        """
        default_config = {
            "neo4j": {
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": ""  # 应该从环境变量或安全的配置方式获取
            },
            "model": {
                "embedding_dim": 128,
                "gnn_output_dim": 128,
                "herb_vocab_size": 1000,
                "symptom_vocab_size": 5000,
                "max_prescription_length": 15
            }
        }
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            # 合并默认配置和用户配置
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key not in config[key]:
                            config[key][sub_key] = sub_value
            return config
        except FileNotFoundError:
            print(f"配置文件 {config_file} 不存在，使用默认配置")
            return default_config
        except Exception as e:
            print(f"加载配置文件失败: {e}，使用默认配置")
            return default_config
    
    def _setup_system(self):
        """设置系统组件"""
        print("正在初始化中药处方推荐系统...")
        
        # 1. 验证Neo4j连接
        if not verify_neo4j_connection():
            raise RuntimeError("无法连接到Neo4j数据库，请检查配置")
        
        # 2. 初始化知识图谱查询接口
        neo4j_config = self.config["neo4j"]
        self.kg_interface = TCMKGQueryInterface(
            neo4j_uri=neo4j_config["uri"],
            username=neo4j_config["username"],
            password=neo4j_config["password"]
        )
        
        # 3. 初始化模型（如果需要在线推理）
        # 注意：实际应用中，模型应该在训练后保存并在推理时加载
        print("系统初始化完成")
    
    def generate_prescription(self, symptoms: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        根据症状生成处方
        
        Args:
            symptoms (List[str]): 症状列表
            top_k (int): 返回前k个推荐
            
        Returns:
            List[Dict[str, Any]]: 推荐的处方列表
        """
        print(f"正在分析症状: {', '.join(symptoms)}")
        
        # 1. 查询知识图谱获取相关信息
        print("正在查询知识图谱...")
        kg_data = self.kg_interface.query_kg_for_symptoms(symptoms)
        print(f"从知识图谱获取到 {len(kg_data['nodes'])} 个节点和 {len(kg_data['edges'])} 条关系")
        
        # 2. 初始化推理模块
        print("正在加载模型...")
        entity_type_sizes = {
            'Herb': self.config["model"]["herb_vocab_size"],
            'Symptom': 500,  # 根据实际情况调整
            'Disease': 300,  # 根据实际情况调整
            'Effect': 200    # 根据实际情况调整
        }
        
        model = TCMGNNInferenceModule(
            entity_type_sizes=entity_type_sizes,
            relation_type_size=10,  # 根据实际情况调整
            embedding_dim=self.config["model"]["embedding_dim"],
            gnn_output_dim=self.config["model"]["gnn_output_dim"],
            herb_vocab_size=self.config["model"]["herb_vocab_size"],
            symptom_vocab_size=self.config["model"]["symptom_vocab_size"],
            max_prescription_length=self.config["model"]["max_prescription_length"],
            device=self.device
        )
        
        # 3. 生成推荐
        print("正在生成处方...")
        recommendations = model.recommend_prescription(symptoms, kg_data, top_k)
        
        return recommendations
    
    def explain_prescription(self, prescription: List[Dict[str, Any]]) -> str:
        """
        解释处方的组成和原理
        
        Args:
            prescription (List[Dict[str, Any]]): 处方列表
            
        Returns:
            str: 处方解释
        """
        if not prescription:
            return "未生成有效的处方"
        
        explanation = "根据您的症状，推荐以下药材组合：\n\n"
        for i, item in enumerate(prescription[:10], 1):  # 只显示前10个
            explanation += f"{i}. {item['herb_name']} (相关性得分: {item['score']:.3f})\n"
        
        explanation += "\n此处方基于中医理论和知识图谱中的药材相互作用关系生成。"
        return explanation
    
    def interactive_mode(self):
        """交互模式"""
        print("=" * 60)
        print("欢迎使用中药处方推荐系统")
        print("=" * 60)
        print("请输入您的症状，用逗号分隔 (例如: 头痛,发热,咳嗽)")
        print("输入 'quit' 或 'exit' 退出系统")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n请输入症状: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("感谢使用中药处方推荐系统，再见！")
                    break
                
                if not user_input:
                    print("请输入至少一个症状")
                    continue
                
                # 解析症状
                symptoms = [s.strip() for s in user_input.split(',') if s.strip()]
                
                # 生成处方
                recommendations = self.generate_prescription(symptoms, top_k=10)
                
                # 显示结果
                print("\n" + "=" * 60)
                print("处方推荐结果")
                print("=" * 60)
                explanation = self.explain_prescription(recommendations)
                print(explanation)
                print("=" * 60)
                
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except Exception as e:
                print(f"\n处理过程中发生错误: {e}")
                print("请重试")
    
    def close(self):
        """关闭系统资源"""
        if self.kg_interface:
            self.kg_interface.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="中药处方推荐系统")
    parser.add_argument('--symptoms', type=str, help='症状列表，用逗号分隔')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--top_k', type=int, default=10, help='返回前k个推荐')
    
    args = parser.parse_args()
    
    try:
        # 创建系统实例
        system = TCMPrecriptionSystem(config_file=args.config)
        
        if args.symptoms:
            # 命令行模式
            symptoms = [s.strip() for s in args.symptoms.split(',') if s.strip()]
            recommendations = system.generate_prescription(symptoms, args.top_k)
            explanation = system.explain_prescription(recommendations)
            print(explanation)
        else:
            # 交互模式
            system.interactive_mode()
        
        # 关闭系统
        system.close()
        
    except Exception as e:
        print(f"系统运行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()