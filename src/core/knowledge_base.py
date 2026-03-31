# Copyright (C) 2026  中医药智能问答系统
# Licensed under Apache License 2.0
# See LICENSE file for details

"""
中医药知识库模块

原创功能：症状 - 方剂匹配、药材说明、知识查询
"""

import logging
import sqlite3
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    中医药知识库管理类

    负责中医药知识的存储、查询和匹配
    """

    def __init__(self, db_path: str = None):
        """
        初始化知识库

        Args:
            db_path: 数据库路径
        """
        if db_path is None:
            # 使用相对于 TCMAI 目录的路径（可移动应用）
            tcmi_root = os.environ.get('TCMAI_ROOT')
            if tcmi_root:
                db_path = os.path.join(tcmi_root, 'data', 'tcm_knowledge.db')
            else:
                # 回退：使用相对路径
                db_path = Path(__file__).parent.parent.parent / "data" / "tcm_knowledge.db"

        self.db_path = str(db_path)
        self.conn = None
        self._connect()

    def _connect(self):
        """连接数据库"""
        try:
            import os
            if not os.path.exists(self.db_path) or os.path.getsize(self.db_path) == 0:
                logger.warning("知识库文件不存在或为空，使用空知识库")
                self.conn = None
                return

            # 尝试连接数据库
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"知识库连接成功：{self.db_path}")
        except Exception as e:
            logger.warning(f"知识库连接失败，使用空知识库：{e}")
            self.conn = None

    def search(self, query: str) -> Dict[str, Any]:
        """
        搜索知识库

        Args:
            query: 查询关键词

        Returns:
            搜索结果
        """
        if not self.conn:
            return {'herbs': [], 'formulas': [], 'patent_medicines': [], 'folk_remedies': [], 'efficacy': []}

        results = {
            'herbs': self._search_herbs(query),
            'formulas': self._search_formulas(query),
            'patent_medicines': self._search_patent_medicines(query),
            'folk_remedies': self._search_folk_remedies(query),
            'efficacy': self._search_efficacy(query)
        }

        return results

    def _search_herbs(self, keyword: str) -> List[Dict]:
        """
        搜索药材

        Args:
            keyword: 关键词

        Returns:
            药材列表
        """
        if not self.conn:
            return []

        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, name, category, nature, flavor, meridian, 
                       efficacy, indications, usage, contraindications
                FROM herbs
                WHERE name LIKE ? OR efficacy LIKE ? OR indications LIKE ? OR category LIKE ?
                LIMIT 10
            """, (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%', f'%{keyword}%'))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row['id'],
                    'name': row['name'],
                    'category': row['category'],
                    'nature': row['nature'],
                    'flavor': row['flavor'],
                    'meridian': row['meridian'],
                    'efficacy': row['efficacy'],
                    'indications': row['indications'],
                    'usage': row['usage'],
                    'contraindications': row['contraindications']
                })

            return results
        except Exception as e:
            logger.error(f"药材搜索失败：{e}")
            return []

    def _search_formulas(self, keyword: str) -> List[Dict]:
        """
        搜索方剂

        Args:
            keyword: 关键词

        Returns:
            方剂列表
        """
        if not self.conn:
            return []

        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, name, category, source_book, composition, dosage,
                       functions, indications, usage, modification, explanation
                FROM formulas
                WHERE name LIKE ? OR functions LIKE ? OR indications LIKE ? OR composition LIKE ?
                LIMIT 10
            """, (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%', f'%{keyword}%'))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row['id'],
                    'name': row['name'],
                    'category': row['category'],
                    'source_book': row['source_book'],
                    'composition': row['composition'],
                    'dosage': row['dosage'],
                    'functions': row['functions'],
                    'indications': row['indications'],
                    'usage': row['usage'],
                    'modification': row['modification'],
                    'explanation': row['explanation']
                })

            return results
        except Exception as e:
            logger.error(f"方剂搜索失败：{e}")
            return []

    def _search_patent_medicines(self, keyword: str) -> List[Dict]:
        """
        搜索中成药

        Args:
            keyword: 关键词

        Returns:
            中成药列表
        """
        if not self.conn:
            return []

        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, name, category, composition, functions, 
                       indications, usage_dosage, contraindications
                FROM patent_medicines
                WHERE name LIKE ? OR functions LIKE ? OR indications LIKE ? OR composition LIKE ?
                LIMIT 10
            """, (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%', f'%{keyword}%'))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row['id'],
                    'name': row['name'],
                    'category': row['category'],
                    'composition': row['composition'],
                    'functions': row['functions'],
                    'indications': row['indications'],
                    'usage_dosage': row['usage_dosage'],
                    'contraindications': row['contraindications']
                })

            return results
        except Exception as e:
            logger.error(f"中成药搜索失败：{e}")
            return []

    def _search_folk_remedies(self, keyword: str) -> List[Dict]:
        """
        搜索民间验方

        Args:
            keyword: 关键词

        Returns:
            民间验方列表
        """
        if not self.conn:
            return []

        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, disease, remedy, ingredients, method, source
                FROM folk_remedies
                WHERE disease LIKE ? OR remedy LIKE ? OR ingredients LIKE ?
                LIMIT 10
            """, (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%'))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row['id'],
                    'disease': row['disease'],
                    'remedy': row['remedy'],
                    'ingredients': row['ingredients'],
                    'method': row['method'],
                    'source': row['source']
                })

            return results
        except Exception as e:
            logger.error(f"民间验方搜索失败：{e}")
            return []

    def _search_efficacy(self, keyword: str) -> List[Dict]:
        """
        搜索功效

        Args:
            keyword: 关键词

        Returns:
            功效相关列表
        """
        if not self.conn:
            return []

        try:
            cursor = self.conn.cursor()
            # 搜索药材功效
            cursor.execute("""
                SELECT 'herb' as type, name, category, efficacy, indications 
                FROM herbs
                WHERE efficacy LIKE ? OR indications LIKE ?
                LIMIT 5
            """, (f'%{keyword}%', f'%{keyword}%'))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'type': row['type'],
                    'name': row['name'],
                    'category': row['category'],
                    'efficacy': row['efficacy'],
                    'indications': row['indications']
                })

            # 搜索方剂功效
            cursor.execute("""
                SELECT 'formula' as type, name, category, functions, indications
                FROM formulas
                WHERE functions LIKE ? OR indications LIKE ?
                LIMIT 5
            """, (f'%{keyword}%', f'%{keyword}%'))

            for row in cursor.fetchall():
                results.append({
                    'type': row['type'],
                    'name': row['name'],
                    'category': row['category'],
                    'efficacy': row['functions'],
                    'indications': row['indications']
                })

            # 搜索中成药功效
            cursor.execute("""
                SELECT 'patent_medicine' as type, name, category, functions, indications
                FROM patent_medicines
                WHERE functions LIKE ? OR indications LIKE ?
                LIMIT 5
            """, (f'%{keyword}%', f'%{keyword}%'))

            for row in cursor.fetchall():
                results.append({
                    'type': row['type'],
                    'name': row['name'],
                    'category': row['category'],
                    'efficacy': row['functions'],
                    'indications': row['indications']
                })

            return results
        except Exception as e:
            logger.error(f"功效搜索失败：{e}")
            return []

    def get_herb_detail(self, herb_id: int) -> Optional[Dict]:
        """
        获取药材详情

        Args:
            herb_id: 药材 ID

        Returns:
            药材详情
        """
        if not self.conn:
            return None

        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, name, category, nature, flavor, meridian, 
                       efficacy, indications, usage, contraindications
                FROM herbs WHERE id = ?
            """, (herb_id,))
            row = cursor.fetchone()

            if row:
                return {
                    'id': row['id'],
                    'name': row['name'],
                    'category': row['category'],
                    'nature': row['nature'],
                    'flavor': row['flavor'],
                    'meridian': row['meridian'],
                    'efficacy': row['efficacy'],
                    'indications': row['indications'],
                    'usage': row['usage'],
                    'contraindications': row['contraindications']
                }
            return None
        except Exception as e:
            logger.error(f"获取药材详情失败：{e}")
            return None

    def get_formula_detail(self, formula_id: int) -> Optional[Dict]:
        """
        获取方剂详情

        Args:
            formula_id: 方剂 ID

        Returns:
            方剂详情
        """
        if not self.conn:
            return None

        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, name, category, source_book, composition, dosage,
                       functions, indications, usage, modification, explanation
                FROM formulas WHERE id = ?
            """, (formula_id,))
            row = cursor.fetchone()

            if row:
                return {
                    'id': row['id'],
                    'name': row['name'],
                    'category': row['category'],
                    'source_book': row['source_book'],
                    'composition': row['composition'],
                    'dosage': row['dosage'],
                    'functions': row['functions'],
                    'indications': row['indications'],
                    'usage': row['usage'],
                    'modification': row['modification'],
                    'explanation': row['explanation']
                }
            return None
        except Exception as e:
            logger.error(f"获取方剂详情失败：{e}")
            return None

    def get_patent_medicine_detail(self, medicine_id: int) -> Optional[Dict]:
        """
        获取中成药详情

        Args:
            medicine_id: 中成药 ID

        Returns:
            中成药详情
        """
        if not self.conn:
            return None

        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, name, category, composition, functions, 
                       indications, usage_dosage, contraindications
                FROM patent_medicines WHERE id = ?
            """, (medicine_id,))
            row = cursor.fetchone()

            if row:
                return {
                    'id': row['id'],
                    'name': row['name'],
                    'category': row['category'],
                    'composition': row['composition'],
                    'functions': row['functions'],
                    'indications': row['indications'],
                    'usage_dosage': row['usage_dosage'],
                    'contraindications': row['contraindications']
                }
            return None
        except Exception as e:
            logger.error(f"获取中成药详情失败：{e}")
            return None

    def match_symptoms(self, symptoms: List[str]) -> List[Dict]:
        """
        根据症状匹配方剂（简化版）
        
        由于新数据库结构移除了症状表，此方法使用简化逻辑，
        通过方剂的 indications 字段进行模糊匹配。

        Args:
            symptoms: 症状列表

        Returns:
            匹配的方剂列表
        """
        if not self.conn or not symptoms:
            return []

        try:
            cursor = self.conn.cursor()
            
            # 构建症状匹配条件
            conditions = []
            params = []
            for symptom in symptoms:
                conditions.append("indications LIKE ?")
                params.append(f'%{symptom}%')
            
            condition_str = ' OR '.join(conditions)
            
            # 查询方剂
            cursor.execute(f"""
                SELECT id, name, category, source_book, composition, 
                       functions, indications
                FROM formulas
                WHERE {condition_str}
                LIMIT 10
            """, params)

            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row['id'],
                    'name': row['name'],
                    'category': row['category'],
                    'source_book': row['source_book'],
                    'composition': row['composition'],
                    'functions': row['functions'],
                    'indications': row['indications']
                })

            logger.info(f"症状匹配完成，找到 {len(results)} 个方剂")
            return results

        except Exception as e:
            logger.error(f"症状 - 方剂匹配失败：{e}")
            return []

    def search_by_category(self, category: str, limit: int = 10) -> Dict[str, List[Dict]]:
        """
        按类别搜索

        Args:
            category: 类别名称
            limit: 返回结果数量限制

        Returns:
            各类别的结果
        """
        if not self.conn:
            return {'herbs': [], 'formulas': [], 'patent_medicines': [], 'folk_remedies': []}

        results = {'herbs': [], 'formulas': [], 'patent_medicines': [], 'folk_remedies': []}
        cursor = self.conn.cursor()

        try:
            # 搜索药材
            cursor.execute("""
                SELECT id, name, category, efficacy, indications
                FROM herbs
                WHERE category LIKE ?
                LIMIT ?
            """, (f'%{category}%', limit))
            for row in cursor.fetchall():
                results['herbs'].append({
                    'id': row['id'],
                    'name': row['name'],
                    'category': row['category'],
                    'efficacy': row['efficacy'],
                    'indications': row['indications']
                })

            # 搜索方剂
            cursor.execute("""
                SELECT id, name, category, functions, indications
                FROM formulas
                WHERE category LIKE ?
                LIMIT ?
            """, (f'%{category}%', limit))
            for row in cursor.fetchall():
                results['formulas'].append({
                    'id': row['id'],
                    'name': row['name'],
                    'category': row['category'],
                    'efficacy': row['functions'],
                    'indications': row['indications']
                })

            # 搜索中成药
            cursor.execute("""
                SELECT id, name, category, functions, indications
                FROM patent_medicines
                WHERE category LIKE ?
                LIMIT ?
            """, (f'%{category}%', limit))
            for row in cursor.fetchall():
                results['patent_medicines'].append({
                    'id': row['id'],
                    'name': row['name'],
                    'category': row['category'],
                    'efficacy': row['functions'],
                    'indications': row['indications']
                })

            # 搜索民间验方（按疾病类别）
            cursor.execute("""
                SELECT id, disease, remedy, ingredients, method
                FROM folk_remedies
                WHERE disease LIKE ?
                LIMIT ?
            """, (f'%{category}%', limit))
            for row in cursor.fetchall():
                results['folk_remedies'].append({
                    'id': row['id'],
                    'disease': row['disease'],
                    'remedy': row['remedy'],
                    'ingredients': row['ingredients'],
                    'method': row['method']
                })

        except Exception as e:
            logger.error(f"按类别搜索失败：{e}")

        return results

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("知识库已关闭")
