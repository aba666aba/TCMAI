# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0
# See LICENSE file for details

"""
中医药深度学习推荐子系统

包含基于深度学习的症状编码、药材推荐、方剂推荐和安全检查模块。
"""

from .recommend import TCMRecommender

__all__ = ["TCMRecommender"]
