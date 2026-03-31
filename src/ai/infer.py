# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0

"""
模型推理模块

注意：本系统已移除 AI 大模型支持，使用基于知识库的降级方案。
如需 AI 能力，请接入外部大模型 API（如 Qwen、ChatGLM 等）。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ModelInference:
    """
    AI 大模型推理类（已禁用）
    
    系统当前使用基于知识库的快速模式，不提供 AI 推理功能。
    """

    def __init__(self, model_path: str = None, config: Optional[dict] = None):
        """
        初始化（空操作）
        
        系统已移除 AI 模型支持，所有推理请求将使用降级方案。
        """
        self.model_path = model_path
        self.config = config or {}
        self.model = None
        self.tokenizer = None
        self.model_available = False  # 始终为 False
        
        logger.info("AI 模型已禁用，使用基于知识库的快速模式")
        print("[模型] AI 模型已禁用")
        print("      [INFO] 系统使用基于知识库的快速模式")
        print("      [INFO] 如需 AI 能力，请接入外部大模型 API")
        print()

    def infer(self, query: str, context: Optional[str] = None) -> None:
        """
        执行推理（已禁用）
        
        Returns:
            None - 始终返回 None，触发降级方案
        """
        logger.info(f"推理请求：{query[:50]}... [AI 模型已禁用]")
        return None  # 返回 None 触发降级方案

    def unload(self):
        """卸载模型（空操作）"""
        pass
