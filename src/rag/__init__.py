# -*- coding: utf-8 -*-
"""RAG 检索增强层（零依赖、纯离线）。

本模块作为现有 TCMAI 训练模型（推荐模型 best_model.pth、NLU 0.5B）的"外脑"：
- corpus : 把 SQLite 知识库 + 知识图谱导出为结构化检索文档
- retriever : 纯 Python BM25 + 同义词扩展的混合检索，返回带出处的证据
后续可由确定性答案合成器（规则/模板）或可选的本地大模型消费。
"""

__all__ = ["tokenizer", "corpus", "retriever"]
