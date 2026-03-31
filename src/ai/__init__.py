# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0
# See LICENSE file for details

"""
中医药智能问答系统 - 模型服务模块

本模块基于开源 AI 大模型推理代码修改
修改说明：进行了适配性修改和优化
"""

from .infer import ModelInference
from .config import ModelConfig

__all__ = ['ModelInference', 'ModelConfig']
