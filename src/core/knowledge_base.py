import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class KnowledgeBase:
    def __init__(self, db_path: str = None):
        if db_path is None:
            tcm_root = os.environ.get("TCMAI_ROOT")
            if tcm_root:
                db_path = os.path.join(tcm_root, "data", "tcm_knowledge.db")
            else:
                import sys

                if getattr(sys, "frozen", False):
                    base_path = Path(sys.executable).parent
                else:
                    base_path = Path(__file__).resolve().parents[2]
                db_path = base_path / "data" / "tcm_knowledge.db"

        self.db_path = str(db_path)
        self.conn: sqlite3.Connection | None = None
        self.is_available = False
        self._lock = threading.RLock()
        self.knowledge_config = None
        self._connect()

    def _connect(self):
        try:
            if not os.path.exists(self.db_path):
                logger.warning("知识库文件不存在，已降级为空知识库: %s", self.db_path)
                return
            # logger.debug("连接知识库: %s", self.db_path)
            if os.path.getsize(self.db_path) == 0:
                logger.warning("知识库文件为空，已降级为空知识库: %s", self.db_path)
                return

            self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=5)
            self.conn.row_factory = sqlite3.Row
            self.is_available = True
            logger.info("知识库连接成功: %s", self.db_path)
            # 初始化知识配置管理器
            try:
                from core.knowledge_config import KnowledgeConfig, migrate_hardcoded_to_db

                migrate_hardcoded_to_db(self.conn)
                self.knowledge_config = KnowledgeConfig(self.conn)
            except Exception as cfg_exc:
                logger.warning("知识配置初始化失败（不影响主功能）: %s", cfg_exc)
        except sqlite3.Error as exc:
            logger.error("知识库连接失败(SQLite): %s", exc)
            self.conn = None
            self.is_available = False
        except Exception as exc:
            logger.error("知识库连接失败: %s", exc)
            self.conn = None
            self.is_available = False

    def _empty_result(self):
        return {
            "herbs": [],
            "formulas": [],
            "patent_medicines": [],
            "folk_remedies": [],
            "efficacy": [],
        }

    def search(self, query: str) -> dict[str, Any]:
        with self._lock:
            if not self.conn:
                return self._empty_result()
            if not query or len(query.strip()) == 0:
                logger.warning("收到空查询，跳过搜索")
                return self._empty_result()
            try:
                return {
                    "herbs": self._search_herbs(query),
                    "formulas": self._search_formulas(query),
                    "patent_medicines": self._search_patent_medicines(query),
                    "folk_remedies": self._search_folk_remedies(query),
                    "efficacy": self._search_efficacy(query),
                }
            except Exception as exc:
                logger.error("知识库搜索失败: %s", exc)
                return self._empty_result()

    def _search_herbs(self, keyword):
        # NOTE: 这里的LIKE查询没走索引，数据量大了会慢
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, name, category, nature, flavor, meridian,
                   efficacy, indications, usage, contraindications
            FROM herbs
            WHERE name LIKE ? OR efficacy LIKE ? OR indications LIKE ? OR category LIKE ?
            LIMIT 10
            """,
            (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"),
        )
        return [dict(row) for row in cursor.fetchall()]

    def _search_formulas(self, keyword):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, name, category, source_book, composition, dosage,
                   functions, indications, usage, modification, explanation
            FROM formulas
            WHERE name LIKE ? OR functions LIKE ? OR indications LIKE ? OR composition LIKE ?
            LIMIT 10
            """,
            (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"),
        )
        return [dict(row) for row in cursor.fetchall()]

    def _search_patent_medicines(self, keyword):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, name, category, composition, functions,
                   indications, usage_dosage, contraindications
            FROM patent_medicines
            WHERE name LIKE ? OR functions LIKE ? OR indications LIKE ? OR composition LIKE ?
            LIMIT 10
            """,
            (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"),
        )
        return [dict(row) for row in cursor.fetchall()]

    def _search_folk_remedies(self, keyword):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, disease, remedy, ingredients, method, source
            FROM folk_remedies
            WHERE disease LIKE ? OR remedy LIKE ? OR ingredients LIKE ?
            LIMIT 10
            """,
            (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"),
        )
        return [dict(row) for row in cursor.fetchall()]

    def _search_efficacy(self, keyword):
        results = []
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT 'herb' AS type, name, category, efficacy, indications
            FROM herbs
            WHERE efficacy LIKE ? OR indications LIKE ?
            LIMIT 5
            """,
            (f"%{keyword}%", f"%{keyword}%"),
        )
        results.extend(dict(row) for row in cursor.fetchall())

        cursor.execute(
            """
            SELECT 'formula' AS type, name, category, functions AS efficacy, indications
            FROM formulas
            WHERE functions LIKE ? OR indications LIKE ?
            LIMIT 5
            """,
            (f"%{keyword}%", f"%{keyword}%"),
        )
        results.extend(dict(row) for row in cursor.fetchall())

        cursor.execute(
            """
            SELECT 'patent_medicine' AS type, name, category, functions AS efficacy, indications
            FROM patent_medicines
            WHERE functions LIKE ? OR indications LIKE ?
            LIMIT 5
            """,
            (f"%{keyword}%", f"%{keyword}%"),
        )
        results.extend(dict(row) for row in cursor.fetchall())

        return results

    def get_herb_detail(self, herb_id: int) -> dict | None:
        with self._lock:
            if not self.conn:
                return None
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT id, name, category, nature, flavor, meridian,
                       efficacy, indications, usage, contraindications
                FROM herbs WHERE id = ?
                """,
                (herb_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_formula_detail(self, formula_id: int) -> dict | None:
        with self._lock:
            if not self.conn:
                return None
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT id, name, category, source_book, composition, dosage,
                       functions, indications, usage, modification, explanation
                FROM formulas WHERE id = ?
                """,
                (formula_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_patent_medicine_detail(self, medicine_id: int) -> dict | None:
        with self._lock:
            if not self.conn:
                return None
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT id, name, category, composition, functions,
                       indications, usage_dosage, contraindications
                FROM patent_medicines WHERE id = ?
                """,
                (medicine_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def match_symptoms(self, symptoms: list[str]) -> list[dict]:
        if not symptoms:
            return []

        with self._lock:
            if not self.conn:
                return []

            max_symptoms_count = 20
            max_symptom_length = 100
            params: list[str] = []
            for symptom in symptoms[:max_symptoms_count]:
                if not isinstance(symptom, str):
                    continue
                clean = symptom.strip()[:max_symptom_length]
                if clean:
                    params.append(f"%{clean}%")

            if not params:
                return []

            condition_str = " OR ".join(["indications LIKE ?"] * len(params))
            cursor = self.conn.cursor()
            cursor.execute(
                f"""
                SELECT id, name, category, source_book, composition, functions, indications
                FROM formulas
                WHERE {condition_str}
                LIMIT 10
                """,
                params,
            )
            return [dict(row) for row in cursor.fetchall()]

    def search_by_category(self, category: str, limit: int = 10) -> dict[str, list[dict]]:
        results = {"herbs": [], "formulas": [], "patent_medicines": [], "folk_remedies": []}
        with self._lock:
            if not self.conn:
                return results

            cursor = self.conn.cursor()
            try:
                cursor.execute(
                    """
                    SELECT id, name, category, efficacy, indications
                    FROM herbs
                    WHERE category LIKE ?
                    LIMIT ?
                    """,
                    (f"%{category}%", limit),
                )
                results["herbs"] = [dict(row) for row in cursor.fetchall()]

                cursor.execute(
                    """
                    SELECT id, name, category, functions AS efficacy, indications
                    FROM formulas
                    WHERE category LIKE ?
                    LIMIT ?
                    """,
                    (f"%{category}%", limit),
                )
                results["formulas"] = [dict(row) for row in cursor.fetchall()]

                cursor.execute(
                    """
                    SELECT id, name, category, functions AS efficacy, indications
                    FROM patent_medicines
                    WHERE category LIKE ?
                    LIMIT ?
                    """,
                    (f"%{category}%", limit),
                )
                results["patent_medicines"] = [dict(row) for row in cursor.fetchall()]

                cursor.execute(
                    """
                    SELECT id, disease, remedy, ingredients, method
                    FROM folk_remedies
                    WHERE disease LIKE ?
                    LIMIT ?
                    """,
                    (f"%{category}%", limit),
                )
                results["folk_remedies"] = [dict(row) for row in cursor.fetchall()]
            except Exception as exc:
                logger.error("按类别搜索失败: %s", exc)

        return results

    # ================================================================
    # 智能搜索（多策略融合搜索）
    # ================================================================

    def intelligent_search(self, query: str) -> dict[str, Any]:
        """智能搜索：同义词扩展 + 证型推理 + 多关键词融合

        自动处理口语化表达（"睡不着"→"失眠"、"没力气"→"乏力"），
        并根据症状推断证型，推荐对应方剂。

        Args:
            query: 用户查询文本（口语化或标准化均可）

        Returns:
            包含 expanded_queries, symptoms, syndromes,
            recommended_formulas, herbs, formulas 等字段的完整结果
        """
        try:
            from core.intelligent_search import IntelligentSearch

            searcher = IntelligentSearch(self)
            result = searcher.search(query)
            return result
        except ImportError:
            logger.warning("智能搜索模块不可用，降级到普通搜索")
            kb_result = self.search(query)
            return {
                "query": query,
                "symptoms": [],
                "inferred_syndromes": [],
                "recommended_formulas": [],
                "herbs": kb_result.get("herbs", []),
                "formulas": kb_result.get("formulas", []),
                "patent_medicines": kb_result.get("patent_medicines", []),
                "folk_remedies": kb_result.get("folk_remedies", []),
                "search_summary": "使用普通搜索模式",
            }
        except Exception as exc:
            logger.error("智能搜索异常: %s", exc)
            kb_result = self.search(query)
            return {
                "query": query,
                "symptoms": [],
                "inferred_syndromes": [],
                "recommended_formulas": [],
                "herbs": kb_result.get("herbs", []),
                "formulas": kb_result.get("formulas", []),
                "patent_medicines": kb_result.get("patent_medicines", []),
                "folk_remedies": kb_result.get("folk_remedies", []),
                "search_summary": "搜索异常，降级到普通模式",
            }

    def get_symptom_suggestions(self, partial: str, max_results: int = 8) -> list[str]:
        """根据输入前缀获取症状建议（模糊匹配）"""
        try:
            from core.intelligent_search import IntelligentSearch
            searcher = IntelligentSearch(self)
            return searcher.get_symptom_suggestions(partial, max_results)
        except Exception:
            return []

    def close(self):
        with self._lock:
            if self.conn:
                self.conn.close()
                self.conn = None
                self.is_available = False
                logger.info("知识库已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

