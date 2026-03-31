# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0

"""
模型配置模块

注意：本系统已移除 AI 大模型支持，使用基于知识库的降级方案。
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    模型配置类（已简化）
    
    系统当前使用基于知识库的快速模式，不需要模型配置。
    """
    model_path: str = None
    hidden_size: int = 0
    num_hidden_layers: int = 0
    use_moe: bool = False
    max_length: int = 0
    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 0
    device: str = "cpu"
    
    # 中医药领域特定配置
    tcm_mode: bool = True
    knowledge_base_enabled: bool = True


def get_default_config() -> ModelConfig:
    """获取默认配置（空配置）"""
    return ModelConfig()


def load_config_from_file(config_path: str) -> ModelConfig:
    """
    从文件加载配置（已禁用）
    
    系统不再使用配置文件。
    """
    import json
    import os

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    
    # 返回空配置
    return ModelConfig()


def save_config_to_file(config: ModelConfig, config_path: str) -> None:
    """
    保存配置到文件（已禁用）
    
    系统不再使用配置文件。
    """
    pass
