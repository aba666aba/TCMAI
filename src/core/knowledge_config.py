# Copyright (C) 2026 中医药智能问答系统 (TCMAI)
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# 本文件是中医药智能问答系统 (TCMAI) 的一部分。
#
# 本程序是自由软件：你可以依据自由软件基金会发布的 GNU Affero 通用公共
# 许可证条款（第 3 版或你选择的任何更新版本）重新分发和/或修改本程序。
#
# 本程序按“希望有用”的原则分发，但不提供任何担保，甚至不提供适销性或
# 特定用途适用性的默示担保。详见 GNU Affero 通用公共许可证。
#
# 你应已随本程序收到 GNU Affero 通用公共许可证副本。如未收到，
# 请见 <https://www.gnu.org/licenses/>。
#
# 注意：以网络服务方式提供本程序的服务端修改版本，同样必须按 AGPL-3.0
# 第 13 条向所有用户提供对应源代码。

"""知识配置管理 — SQLite优先 + 硬编码回退，支持在线编辑"""

import json
import logging
import sqlite3
from typing import Any

from core import tcm_config

logger = logging.getLogger(__name__)

# 数据库表定义
_CONFIG_TABLES = {
    "symptom_aliases": """
        CREATE TABLE IF NOT EXISTS symptom_aliases (
            alias TEXT PRIMARY KEY,
            standard TEXT NOT NULL
        )
    """,
    "symptom_formula_map": """
        CREATE TABLE IF NOT EXISTS symptom_formula_map (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symptom TEXT NOT NULL,
            syndrome TEXT NOT NULL,
            formula TEXT NOT NULL,
            description TEXT DEFAULT ''
        )
    """,
    "tcm_synonyms": """
        CREATE TABLE IF NOT EXISTS tcm_synonyms (
            keyword TEXT NOT NULL,
            synonym TEXT NOT NULL,
            PRIMARY KEY (keyword, synonym)
        )
    """,
    "formula_aliases": """
        CREATE TABLE IF NOT EXISTS formula_aliases (
            alias TEXT PRIMARY KEY,
            standard TEXT NOT NULL
        )
    """,
    "herb_aliases": """
        CREATE TABLE IF NOT EXISTS herb_aliases (
            alias TEXT PRIMARY KEY,
            standard TEXT NOT NULL
        )
    """,
    "symptom_info": """
        CREATE TABLE IF NOT EXISTS symptom_info (
            name TEXT PRIMARY KEY,
            category TEXT DEFAULT '',
            description TEXT DEFAULT '',
            related_syndromes TEXT DEFAULT ''
        )
    """,
    "syndrome_definitions": """
        CREATE TABLE IF NOT EXISTS syndrome_definitions (
            name TEXT PRIMARY KEY,
            description TEXT DEFAULT '',
            symptoms_json TEXT DEFAULT '[]',
            treatment TEXT DEFAULT '',
            formula TEXT DEFAULT ''
        )
    """,
}


def ensure_config_tables(conn: sqlite3.Connection) -> None:
    """在数据库中创建配置表（如果不存在）。"""
    for ddl in _CONFIG_TABLES.values():
        conn.execute(ddl)
    conn.commit()


def migrate_hardcoded_to_db(conn: sqlite3.Connection) -> int:
    """将 tcm_config.py 中的硬编码数据迁移到数据库。

    仅迁移数据库中尚不存在的条目，已有的不覆盖。
    返回新增条目数。
    """
    ensure_config_tables(conn)
    count = 0

    # 症状别名
    for alias, standard in tcm_config.SYMPTOM_ALIAS_MAP.items():
        try:
            conn.execute(
                "INSERT OR IGNORE INTO symptom_aliases (alias, standard) VALUES (?, ?)",
                (alias, standard),
            )
            count += 1
        except Exception as exc:
            logger.debug("迁移数据跳过: %s", exc)

    # 症状→方剂映射
    for symptom, entries in tcm_config.SYMPTOM_TO_FORMULA_MAP.items():
        for entry in entries:
            try:
                conn.execute(
                    "INSERT INTO symptom_formula_map (symptom, syndrome, formula, description) "
                    "SELECT ?, ?, ?, ? WHERE NOT EXISTS ("
                    "  SELECT 1 FROM symptom_formula_map WHERE symptom=? AND syndrome=? AND formula=?)",
                    (
                        symptom,
                        entry["syndrome"],
                        entry["formula"],
                        entry.get("description", ""),
                        symptom,
                        entry["syndrome"],
                        entry["formula"],
                    ),
                )
                count += 1
            except Exception as exc:
                logger.debug("迁移数据跳过: %s", exc)

    # 中医同义词
    for keyword, synonyms in tcm_config.TCM_SYNONYM_MAP.items():
        for syn in synonyms:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO tcm_synonyms (keyword, synonym) VALUES (?, ?)",
                    (keyword, syn),
                )
                count += 1
            except Exception as exc:
                logger.debug("迁移数据跳过: %s", exc)

    # 方剂别名
    for standard, aliases in tcm_config.FORMULA_ALIAS_MAP.items():
        for alias in aliases:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO formula_aliases (alias, standard) VALUES (?, ?)",
                    (alias, standard),
                )
                count += 1
            except Exception as exc:
                logger.debug("迁移数据跳过: %s", exc)

    # 药材别名
    for standard, aliases in tcm_config.HERB_ALIAS_MAP.items():
        for alias in aliases:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO herb_aliases (alias, standard) VALUES (?, ?)",
                    (alias, standard),
                )
                count += 1
            except Exception as exc:
                logger.debug("迁移数据跳过: %s", exc)

    # 症状信息
    for name, info in tcm_config.SYMPTOM_INFO.items():
        try:
            conn.execute(
                "INSERT OR IGNORE INTO symptom_info (name, category, description, related_syndromes) "
                "VALUES (?, ?, ?, ?)",
                (name, info.get("category", ""), info.get("description", ""),
                 info.get("related_syndromes", "")),
            )
            count += 1
        except Exception as exc:
            logger.debug("迁移数据跳过: %s", exc)

    # 证型定义
    for name, info in tcm_config.SYNDROME_DEFINITIONS.items():
        try:
            conn.execute(
                "INSERT OR IGNORE INTO syndrome_definitions "
                "(name, description, symptoms_json, treatment, formula) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, info.get("description", ""),
                 json.dumps(info.get("symptoms", []), ensure_ascii=False),
                 info.get("treatment", ""), info.get("formula", "")),
            )
            count += 1
        except Exception as exc:
            logger.debug("迁移数据跳过: %s", exc)

    conn.commit()
    logger.info("知识配置迁移完成，新增 %d 条记录", count)
    return count


class KnowledgeConfig:
    """知识配置管理器。

    优先从数据库读取，回退到 tcm_config.py 硬编码值。
    所有读方法均支持在线编辑后的即时生效。
    """

    def __init__(self, conn: sqlite3.Connection | None = None):
        self._conn = conn
        self._has_tables = False
        if conn:
            try:
                ensure_config_tables(conn)
                self._has_tables = True
            except Exception as exc:
                logger.warning("初始化知识配置表失败: %s", exc)

    def set_connection(self, conn: sqlite3.Connection) -> None:
        """设置数据库连接。"""
        self._conn = conn
        try:
            ensure_config_tables(conn)
            self._has_tables = True
        except Exception as exc:
            logger.warning("知识配置表初始化失败，将使用硬编码回退: %s", exc)
            self._has_tables = False

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        if not self._conn or not self._has_tables:
            return []
        try:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
            return [dict(r) if isinstance(r, sqlite3.Row) else r for r in rows]
        except Exception as exc:
            logger.debug("查询失败: %s — %s", sql, exc)
            return []

    def _query_one(self, sql: str, params: tuple = ()) -> Any | None:
        rows = self._query(sql, params)
        return rows[0] if rows else None

    # 症状别名
    def get_symptom_alias_map(self) -> dict[str, str]:
        """获取症状别名映射（口语→标准术语）。"""
        rows = self._query("SELECT alias, standard FROM symptom_aliases")
        if rows:
            return {r["alias"]: r["standard"] for r in rows}
        return dict(tcm_config.SYMPTOM_ALIAS_MAP)

    def get_symptom_list(self) -> list[str]:
        """获取所有标准症状名称。"""
        rows = self._query(
            "SELECT DISTINCT standard FROM symptom_aliases "
            "UNION SELECT name FROM symptom_info"
        )
        if rows:
            return sorted({r[0] if isinstance(r, tuple) else r.get("standard") or r.get("name") for r in rows})
        return list(tcm_config.SYMPTOM_LIST)

    # 症状→方剂映射
    def get_symptom_formula_map(self) -> dict[str, list[dict]]:
        """获取症状→证型→方剂映射。"""
        rows = self._query(
            "SELECT symptom, syndrome, formula, description FROM symptom_formula_map ORDER BY symptom, id"
        )
        if rows:
            result: dict[str, list[dict]] = {}
            for r in rows:
                sym = r["symptom"]
                if sym not in result:
                    result[sym] = []
                result[sym].append({
                    "syndrome": r["syndrome"],
                    "formula": r["formula"],
                    "description": r["description"],
                })
            return result
        return dict(tcm_config.SYMPTOM_TO_FORMULA_MAP)

    def add_symptom_formula(self, symptom: str, syndrome: str, formula: str, description: str = "") -> bool:
        """添加症状→方剂映射。"""
        if not self._conn or not self._has_tables:
            return False
        try:
            self._conn.execute(
                "INSERT INTO symptom_formula_map (symptom, syndrome, formula, description) VALUES (?, ?, ?, ?)",
                (symptom, syndrome, formula, description),
            )
            self._conn.commit()
            return True
        except Exception as exc:
            logger.error("添加映射失败: %s", exc)
            return False

    def delete_symptom_formula(self, row_id: int) -> bool:
        """删除症状→方剂映射条目。"""
        if not self._conn or not self._has_tables:
            return False
        try:
            self._conn.execute("DELETE FROM symptom_formula_map WHERE id = ?", (row_id,))
            self._conn.commit()
            return True
        except Exception as exc:
            logger.error("删除映射失败 id=%d: %s", row_id, exc)
            return False

    # 中医同义词
    def get_tcm_synonym_map(self) -> dict[str, list[str]]:
        """获取中医同义词映射。"""
        rows = self._query("SELECT keyword, synonym FROM tcm_synonyms")
        if rows:
            result: dict[str, list[str]] = {}
            for r in rows:
                kw = r["keyword"]
                if kw not in result:
                    result[kw] = []
                result[kw].append(r["synonym"])
            return result
        return dict(tcm_config.TCM_SYNONYM_MAP)

    # 方剂别名
    def get_formula_alias_map(self) -> dict[str, list[str]]:
        """获取方剂别名映射。"""
        rows = self._query("SELECT alias, standard FROM formula_aliases")
        if rows:
            result: dict[str, list[str]] = {}
            for r in rows:
                std = r["standard"]
                if std not in result:
                    result[std] = []
                result[std].append(r["alias"])
            return result
        return dict(tcm_config.FORMULA_ALIAS_MAP)

    # 药材别名
    def get_herb_alias_map(self) -> dict[str, list[str]]:
        """获取药材别名映射。"""
        rows = self._query("SELECT alias, standard FROM herb_aliases")
        if rows:
            result: dict[str, list[str]] = {}
            for r in rows:
                std = r["standard"]
                if std not in result:
                    result[std] = []
                result[std].append(r["alias"])
            return result
        return dict(tcm_config.HERB_ALIAS_MAP)

    # 症状信息
    def get_symptom_info(self) -> dict[str, dict]:
        """获取症状详细信息。"""
        rows = self._query("SELECT name, category, description, related_syndromes FROM symptom_info")
        if rows:
            return {
                r["name"]: {
                    "category": r["category"],
                    "description": r["description"],
                    "related_syndromes": r["related_syndromes"],
                }
                for r in rows
            }
        return dict(tcm_config.SYMPTOM_INFO)

    # 证型定义
    def get_syndrome_definitions(self) -> dict[str, dict]:
        """获取证型定义。"""
        rows = self._query(
            "SELECT name, description, symptoms_json, treatment, formula FROM syndrome_definitions"
        )
        if rows:
            result = {}
            for r in rows:
                try:
                    symptoms = json.loads(r["symptoms_json"])
                except Exception:
                    symptoms = []
                result[r["name"]] = {
                    "description": r["description"],
                    "symptoms": symptoms,
                    "treatment": r["treatment"],
                    "formula": r["formula"],
                }
            return result
        return dict(tcm_config.SYNDROME_DEFINITIONS)

    def update_symptom_alias(self, alias: str, standard: str) -> bool:
        """添加或更新症状别名。"""
        if not self._conn or not self._has_tables:
            return False
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO symptom_aliases (alias, standard) VALUES (?, ?)",
                (alias, standard),
            )
            self._conn.commit()
            return True
        except Exception as exc:
            logger.error("更新症状别名失败 '%s': %s", alias, exc)
            return False

    def delete_symptom_alias(self, alias: str) -> bool:
        """删除症状别名。"""
        if not self._conn or not self._has_tables:
            return False
        try:
            self._conn.execute("DELETE FROM symptom_aliases WHERE alias = ?", (alias,))
            self._conn.commit()
            return True
        except Exception as exc:
            logger.error("删除症状别名失败 '%s': %s", alias, exc)
            return False

    def get_config_stats(self) -> dict[str, int]:
        """获取各配置表的记录数统计。"""
        tables = [
            "symptom_aliases", "symptom_formula_map", "tcm_synonyms",
            "formula_aliases", "herb_aliases", "symptom_info", "syndrome_definitions",
        ]
        stats = {}
        for table in tables:
            rows = self._query(f"SELECT COUNT(*) AS cnt FROM {table}")
            stats[table] = rows[0]["cnt"] if rows else 0
        return stats
