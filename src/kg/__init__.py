"""TCM 中医药知识图谱模块

提供知识图谱的构建（builder）、查询（query）、增强推荐（recommender）能力。
核心数据结构是 NetworkX MultiDiGraph，节点为实体（药材/方剂/证型等），边为关系。
"""

from .builder import TCMKnowledgeGraph  # noqa: F401
from .query import KnowledgeGraphQuery  # noqa: F401
