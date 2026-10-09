# Copyright (C) 2026 aba666aba
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

"""知识管理标签页 — 查看/编辑内置知识数据"""

import logging
import tkinter as tk
from tkinter import ttk

import customtkinter as ctk

logger = logging.getLogger(__name__)


class KnowledgeManagerView(ctk.CTkFrame):
    """知识管理面板：标签页切换 + 表格展示 + 增删改"""

    # 统一字体
    FONT_TITLE = ("Microsoft YaHei", 16, "bold")
    FONT_NORMAL = ("Microsoft YaHei", 13)
    FONT_SMALL = ("Microsoft YaHei", 11)
    FONT_XS = ("Microsoft YaHei", 10)

    # 各配置表的显示定义
    TABLE_DEFS = {
        "symptom_aliases": {
            "label": "症状别名",
            "columns": ("alias", "standard"),
            "headers": ("口语/别名", "标准症状"),
            "description": "将用户的口语化表达映射到标准症状名称",
        },
        "symptom_formula_map": {
            "label": "症状→方剂映射",
            "columns": ("id", "symptom", "syndrome", "formula", "description"),
            "headers": ("ID", "症状", "证型", "方剂", "说明"),
            "description": "根据症状推断证型，推荐对应方剂",
        },
        "tcm_synonyms": {
            "label": "中医同义词",
            "columns": ("keyword", "synonym"),
            "headers": ("关键词", "同义词"),
            "description": "证型、症状的同义表达扩展",
        },
        "formula_aliases": {
            "label": "方剂别名",
            "columns": ("alias", "standard"),
            "headers": ("别名/俗称", "标准方剂名"),
            "description": "方剂的常见别名和简称",
        },
        "herb_aliases": {
            "label": "药材别名",
            "columns": ("alias", "standard"),
            "headers": ("别名", "标准药材名"),
            "description": "药材的常见别名",
        },
    }

    def __init__(self, master, kb=None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.kb = kb
        self._cfg = getattr(kb, "knowledge_config", None) if kb else None
        self._current_table = "symptom_aliases"
        self._all_rows: list[tuple] = []
        self._build_ui()
        self._load_table("symptom_aliases")

    def set_kb(self, kb) -> None:
        """设置知识库实例。"""
        self.kb = kb
        self._cfg = getattr(kb, "knowledge_config", None) if kb else None
        self._load_table(self._current_table)

    def _build_ui(self):
        # 顶部：说明
        info_frame = ctk.CTkFrame(self, fg_color="transparent")
        info_frame.pack(fill="x", padx=16, pady=(10, 4))
        self._info_label = ctk.CTkLabel(
            info_frame, text="", font=self.FONT_SMALL, text_color="#666"
        )
        self._info_label.pack(side="left")

        # 统计信息
        self._stats_label = ctk.CTkLabel(
            info_frame, text="", font=self.FONT_XS, text_color="#999"
        )
        self._stats_label.pack(side="right")

        # 标签栏
        tab_frame = ctk.CTkFrame(self, fg_color="transparent")
        tab_frame.pack(fill="x", padx=16, pady=(4, 8))

        self._tab_buttons: dict[str, ctk.CTkButton] = {}
        for table_key, tdef in self.TABLE_DEFS.items():
            btn = ctk.CTkButton(
                tab_frame,
                text=tdef["label"],
                font=self.FONT_SMALL,
                width=100,
                height=28,
                command=lambda t=table_key: self._load_table(t),
            )
            btn.pack(side="left", padx=(0, 4))
            self._tab_buttons[table_key] = btn

        # 搜索框
        search_frame = ctk.CTkFrame(self, fg_color="transparent")
        search_frame.pack(fill="x", padx=16, pady=(0, 4))
        ctk.CTkLabel(search_frame, text="🔍 搜索:", font=self.FONT_SMALL).pack(
            side="left"
        )
        self._search_var = tk.StringVar()
        search_entry = ctk.CTkEntry(
            search_frame,
            textvariable=self._search_var,
            font=self.FONT_SMALL,
            width=300,
            placeholder_text="输入关键词筛选...",
        )
        search_entry.pack(side="left", padx=(4, 0))
        self._search_var.trace_add("write", lambda *_: self._apply_filter())

        # 操作按钮
        btn_frame = ctk.CTkFrame(search_frame, fg_color="transparent")
        btn_frame.pack(side="right")
        ctk.CTkButton(
            btn_frame,
            text="➕ 添加",
            font=self.FONT_XS,
            width=70,
            height=26,
            command=self._add_entry,
        ).pack(side="left", padx=2)
        ctk.CTkButton(
            btn_frame,
            text="🗑 删除",
            font=self.FONT_XS,
            width=70,
            height=26,
            fg_color="#E53935",
            hover_color="#C62828",
            command=self._delete_entry,
        ).pack(side="left", padx=2)
        ctk.CTkButton(
            btn_frame,
            text="🔄 刷新",
            font=self.FONT_XS,
            width=70,
            height=26,
            fg_color="#607D8B",
            hover_color="#455A64",
            command=lambda: self._load_table(self._current_table),
        ).pack(side="left", padx=2)

        # 表格
        table_frame = ctk.CTkFrame(self)
        table_frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        self._tree = ttk.Treeview(table_frame, show="headings", selectmode="browse")
        self._tree.pack(fill="both", expand=True, side="left")

        scroll_y = ttk.Scrollbar(
            table_frame, orient="vertical", command=self._tree.yview
        )
        scroll_y.pack(fill="y", side="right")
        self._tree.configure(yscrollcommand=scroll_y.set)

        # 表格样式
        style = ttk.Style()
        style.configure(
            "Knowledge.Treeview",
            font=("Microsoft YaHei", 17),
            rowheight=44,
        )
        style.configure(
            "Knowledge.Treeview.Heading",
            font=("Microsoft YaHei", 17, "bold"),
        )
        self._tree.configure(style="Knowledge.Treeview")

    def _load_table(self, table_key: str):
        self._current_table = table_key
        self._search_var.set("")

        # 更新标签栏高亮
        for key, btn in self._tab_buttons.items():
            if key == table_key:
                btn.configure(fg_color="#8B4513")
            else:
                btn.configure(fg_color=("gray75", "gray30"))

        tdef = self.TABLE_DEFS[table_key]
        self._info_label.configure(text=tdef["description"])

        # 更新统计
        if self._cfg:
            stats = self._cfg.get_config_stats()
            total = stats.get(table_key, 0)
            self._stats_label.configure(text=f"共 {total} 条记录")

        # 设置表头
        cols = tdef["columns"]
        headers = tdef["headers"]
        self._tree["columns"] = cols
        for col, header in zip(cols, headers):
            self._tree.heading(col, text=header)
            if col == "id":
                self._tree.column(col, width=50, minwidth=40, anchor="center")
            elif col in ("description", "synonym"):
                self._tree.column(col, width=300, minwidth=100)
            else:
                self._tree.column(col, width=120, minwidth=80)

        # 加载数据
        self._all_rows = self._fetch_table_data(table_key)
        self._refresh_tree()

    def _fetch_table_data(self, table_key: str) -> list[tuple]:
        """从数据库获取表格数据。"""
        if not self._cfg:
            return []
        conn = self._cfg._conn
        if not conn:
            return []
        try:
            tdef = self.TABLE_DEFS[table_key]
            cols = tdef["columns"]
            col_str = ", ".join(cols)
            cur = conn.execute(f"SELECT {col_str} FROM {table_key}")
            return [tuple(row) for row in cur.fetchall()]
        except Exception as exc:
            logger.error("查询 %s 失败: %s", table_key, exc)
            return []

    def _refresh_tree(self):
        for item in self._tree.get_children():
            self._tree.delete(item)
        for row in self._all_rows:
            self._tree.insert("", "end", values=row)

    def _apply_filter(self):
        keyword = self._search_var.get().strip().lower()
        for item in self._tree.get_children():
            self._tree.delete(item)
        for row in self._all_rows:
            if not keyword or keyword in str(row).lower():
                self._tree.insert("", "end", values=row)

    def _add_entry(self):
        """弹出对话框添加新条目。"""
        tdef = self.TABLE_DEFS[self._current_table]
        cols = tdef["columns"]
        headers = tdef["headers"]

        dialog = ctk.CTkToplevel(self)
        dialog.title(f"添加 - {tdef['label']}")
        dialog.geometry("400x300")
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()

        entries: dict[str, ctk.CTkEntry] = {}
        for col, header in zip(cols, headers):
            if col == "id":
                continue
            frame = ctk.CTkFrame(dialog, fg_color="transparent")
            frame.pack(fill="x", padx=16, pady=4)
            ctk.CTkLabel(frame, text=f"{header}:", font=self.FONT_SMALL, width=80).pack(
                side="left"
            )
            entry = ctk.CTkEntry(frame, font=self.FONT_SMALL, width=260)
            entry.pack(side="left", padx=(4, 0), fill="x", expand=True)
            entries[col] = entry

        def on_confirm():
            values = {col: entry.get().strip() for col, entry in entries.items()}
            if not any(values.values()):
                return
            conn = self._cfg._conn if self._cfg else None
            if not conn:
                return
            try:
                if self._current_table == "symptom_formula_map":
                    conn.execute(
                        "INSERT INTO symptom_formula_map (symptom, syndrome, formula, description) "
                        "VALUES (?, ?, ?, ?)",
                        (values.get("symptom", ""), values.get("syndrome", ""),
                         values.get("formula", ""), values.get("description", "")),
                    )
                elif self._current_table == "tcm_synonyms":
                    conn.execute(
                        "INSERT OR IGNORE INTO tcm_synonyms (keyword, synonym) VALUES (?, ?)",
                        (values.get("keyword", ""), values.get("synonym", "")),
                    )
                else:
                    col_list = [c for c in cols if c != "id"]
                    placeholders = ", ".join(["?"] * len(col_list))
                    col_str = ", ".join(col_list)
                    conn.execute(
                        f"INSERT OR IGNORE INTO {self._current_table} ({col_str}) VALUES ({placeholders})",
                        tuple(values.get(c, "") for c in col_list),
                    )
                conn.commit()
                dialog.destroy()
                self._load_table(self._current_table)
            except Exception as exc:
                logger.error("添加失败: %s", exc)

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(fill="x", padx=16, pady=12)
        ctk.CTkButton(
            btn_frame, text="确认添加", font=self.FONT_SMALL,
            command=on_confirm,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            btn_frame, text="取消", font=self.FONT_SMALL,
            fg_color="#757575", hover_color="#616161",
            command=dialog.destroy,
        ).pack(side="left")

    def _delete_entry(self):
        """删除选中的条目。"""
        selection = self._tree.selection()
        if not selection:
            return
        item = self._tree.item(selection[0])
        values = item.get("values", [])
        if not values:
            return

        conn = self._cfg._conn if self._cfg else None
        if not conn:
            return

        tdef = self.TABLE_DEFS[self._current_table]
        cols = tdef["columns"]

        try:
            if self._current_table == "symptom_formula_map":
                conn.execute("DELETE FROM symptom_formula_map WHERE id = ?", (values[0],))
            elif self._current_table == "tcm_synonyms":
                conn.execute(
                    "DELETE FROM tcm_synonyms WHERE keyword = ? AND synonym = ?",
                    (values[0], values[1]),
                )
            else:
                # 使用第一列作为主键删除
                conn.execute(
                    f"DELETE FROM {self._current_table} WHERE {cols[0]} = ?",
                    (values[0],),
                )
            conn.commit()
            self._load_table(self._current_table)
        except Exception as exc:
            logger.error("删除失败: %s", exc)
