"""NLU 模块 — 自然语言理解层

基于 Qwen2.5 大模型，将用户的口语化症状描述提取为标准症状术语列表。
入口函数 extract_symptoms(text) 接收原始文本，返回 ["失眠", "心悸", ...]。
"""
from .engine import NLUEngine, extract_symptoms, get_engine

__all__ = ["NLUEngine", "extract_symptoms", "get_engine"]
