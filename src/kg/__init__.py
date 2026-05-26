"""
TCM 中医药知识图谱模块

实体类型:
  Symptom(症状), Syndrome(证型), Herb(药材), Formula(方剂)
  PatentMedicine(中成药), FolkRemedy(验方), Category(药材分类)
  Meridian(归经), Nature(性/气), Flavor(味)

关系类型:
  belongs_to(属于), recommends_herb(推荐药材), recommends_formula(推荐方剂)
  associated_with_herb(症状关联药材), associated_with_formula(症状关联方剂)
  composes(组成方剂), has_nature(性), has_flavor(味), has_meridian(归经)
  belongs_to_category(属于分类), incompatible_with(相反相畏)
  toxic(有毒), co_occurs_with(共现)
"""

from .builder import TCMKnowledgeGraph  # noqa: F401
from .query import KnowledgeGraphQuery  # noqa: F401
