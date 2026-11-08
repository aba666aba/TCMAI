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
"""知识图谱可视化组件 — Canvas 渲染，支持搜索/缩放/节点交互"""

import logging
import threading
import tkinter as tk

import customtkinter as ctk
import networkx as nx

logger = logging.getLogger(__name__)

# ── 实体类型颜色方案 ──
NODE_COLORS = {
    "Symptom": "#e74c3c",
    "Syndrome": "#9b59b6",
    "Herb": "#27ae60",
    "Formula": "#2980b9",
    "PatentMedicine": "#f39c12",
    "FolkRemedy": "#95a5a6",
    "Category": "#1abc9c",
    "Meridian": "#e67e22",
    "Nature": "#e91e63",
    "Flavor": "#00bcd4",
}
NODE_SIZE_MAP = {
    "Symptom": 200,
    "Syndrome": 280,
    "Herb": 160,
    "Formula": 220,
    "PatentMedicine": 180,
    "FolkRemedy": 140,
    "Category": 120,
    "Meridian": 120,
    "Nature": 100,
    "Flavor": 100,
}
EDGE_COLORS = {
    "belongs_to": "#e74c3c",
    "recommends_herb": "#9b59b6",
    "recommends_formula": "#2980b9",
    "associated_with_herb": "#27ae60",
    "associated_with_formula": "#2980b9",
    "composes": "#2c3e50",
    "has_nature": "#e91e63",
    "has_flavor": "#00bcd4",
    "has_meridian": "#e67e22",
    "belongs_to_category": "#1abc9c",
    "incompatible_with": "#c0392b",
    "co_occurs_with": "#7f8c8d",
}

KG_INSTANCE = None
_KG_LOCK = threading.Lock()


def get_kg(*, force_rebuild: bool = False):
    global KG_INSTANCE
    if not force_rebuild and KG_INSTANCE is not None:
        return KG_INSTANCE

    with _KG_LOCK:
        if not force_rebuild and KG_INSTANCE is not None:
            return KG_INSTANCE

        import os
        import pickle
        import sys
        from pathlib import Path

        tcm_root = os.environ.get("TCMAI_ROOT")
        if tcm_root:
            project_root = Path(tcm_root)
        elif getattr(sys, "frozen", False):
            project_root = Path(sys.executable).parent
        else:
            project_root = Path(__file__).resolve().parents[2]

        if str(project_root / "src") in sys.path:
            sys.path.remove(str(project_root / "src"))
        sys.path.insert(0, str(project_root / "src"))
        if not os.environ.get("TCMAI_ROOT"):
            os.environ["TCMAI_ROOT"] = str(project_root)

        cache_path = os.path.join(project_root, "data", "kg_cache.pkl")

        if not force_rebuild and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    g = pickle.load(f)
                from kg.query import KnowledgeGraphQuery

                q = KnowledgeGraphQuery(g)
                logger.info(
                    "知识图谱从缓存加载: %s 节点, %s 边",
                    g.number_of_nodes(),
                    g.number_of_edges(),
                )
                KG_INSTANCE = (g, q)
                return KG_INSTANCE
            except Exception as exc:
                logger.warning("缓存加载失败(将重新构建): %s", exc)

        from kg.builder import TCMKnowledgeGraph
        from kg.query import KnowledgeGraphQuery

        logger.info("构建知识图谱（首次较慢，后续将使用缓存）...")
        kg = TCMKnowledgeGraph()
        kg.build()
        g = kg.graph
        q = KnowledgeGraphQuery(g)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(g, f, protocol=4)
            logger.info(
                "知识图谱已缓存到磁盘: %s (%d 节点, %d 边)",
                cache_path,
                g.number_of_nodes(),
                g.number_of_edges(),
            )
        except Exception as exc:
            logger.warning("缓存写入失败: %s", exc)
        KG_INSTANCE = (g, q)
        return KG_INSTANCE


class KnowledgeGraphView(ctk.CTkFrame):
    """知识图谱可视化组件（CTk 版）"""

    def __init__(self, parent, standalone=True, **kwargs):
        super().__init__(parent, **kwargs)
        self.standalone = standalone
        self.graph = None
        self.query = None
        self.fig = None
        self.canvas = None
        self.ax = None
        self._pos_cache = {}
        self._current_subgraph = None
        self._selected_node = None
        self._expand_history = []
        self._all_node_ids = []
        self._tk_canvas = None
        self._canvas_items = {}
        self._edge_items = []
        self._zoom_level = 1.0
        self._canvas_bg = None
        self._drag_start = None
        self.knowledge_text = None
        self._loading = False
        self._pending_callback = None

        # CTk 字体常量
        self.FONT = ("Microsoft YaHei UI", 14)
        self.FONT_SM = ("Microsoft YaHei UI", 12)
        self.FONT_XS = ("Microsoft YaHei UI", 11)

        self._build_ui()

        self.type_vars = {
            "Symptom": tk.BooleanVar(value=True),
            "Syndrome": tk.BooleanVar(value=True),
            "Herb": tk.BooleanVar(value=True),
            "Formula": tk.BooleanVar(value=True),
        }
        self._edge_mode = tk.StringVar(value="all")

    # UI 构建

    def _build_ui(self):
        if self.standalone:
            self._build_search_toolbar()
            self._build_edge_filter_bar()
        else:
            self._build_compact_toolbar()

        if self.standalone:
            legend_frame = ctk.CTkFrame(self, fg_color="transparent")
            legend_frame.pack(fill="x", padx=6, pady=2)
            self._build_legend(legend_frame)

        if self.standalone:
            main_frame = ctk.CTkFrame(self, fg_color="transparent")
            main_frame.pack(fill="both", expand=True)

            left_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
            left_frame.pack(side="left", fill="both", expand=True)

            right_frame = ctk.CTkFrame(
                main_frame,
                width=300,
                corner_radius=8,
            )
            right_frame.pack(side="right", fill="y")
            right_frame.pack_propagate(False)

            ctk.CTkLabel(
                right_frame,
                text="📖 知识详情",
                font=("Microsoft YaHei UI", 14, "bold"),
                anchor="w",
            ).pack(fill="x", padx=14, pady=(12, 4))

            ctk.CTkFrame(
                right_frame, height=1, fg_color=("gray70", "gray30")
            ).pack(fill="x", padx=6)

            self.detail_text = ctk.CTkTextbox(
                right_frame,
                wrap="word",
                font=self.FONT_XS,
                border_width=0,
                activate_scrollbars=True,
            )
            self.detail_text.pack(fill="both", expand=True, padx=6, pady=6)
            self.detail_text.insert(
                "1.0", "💡 点击节点弹出详情窗口\n\n双击展开邻域 | 右键更多操作 | 滚轮缩放"
            )
            self.detail_text.configure(state="disabled")

            canvas_parent = left_frame
        else:
            canvas_parent = self

        self._tk_canvas = tk.Canvas(
            canvas_parent,
            bg="#1e1e1e",
            highlightthickness=0,
            cursor="arrow",
        )
        self._tk_canvas.pack(fill="both", expand=True)

        self._tk_canvas.bind("<ButtonPress-1>", self._cv_on_press)
        self._tk_canvas.bind("<B1-Motion>", self._cv_on_drag)
        self._tk_canvas.bind("<ButtonRelease-1>", self._cv_on_release)
        self._tk_canvas.bind("<Double-Button-1>", self._cv_on_double_click)
        self._tk_canvas.bind("<MouseWheel>", self._cv_on_wheel)
        self._tk_canvas.bind("<Motion>", self._cv_on_motion)
        self._tk_canvas.bind("<Button-3>", self._cv_on_right_click)

        self._context_menu = tk.Menu(self, tearoff=0, bg="#2b2b2b", fg="#ffffff")
        self._context_menu.add_command(
            label="🔍 查看详情", command=self._ctx_show_detail
        )
        self._context_menu.add_command(label="⬆ 展开邻域", command=self._ctx_expand)
        self._context_menu.add_command(
            label="🎯 设为中心", command=self._ctx_set_center
        )
        self._context_menu.add_separator()
        self._context_menu.add_command(label="📋 复制名称", command=self._ctx_copy_name)

        if not self.standalone:
            detail_frame = ctk.CTkFrame(
                self,
                corner_radius=6,
            )
            detail_frame.pack(fill="x", padx=6, pady=4)

            detail_header = ctk.CTkFrame(detail_frame, fg_color="transparent")
            detail_header.pack(fill="x")
            ctk.CTkLabel(
                detail_header,
                text="📖 节点详情",
                font=self.FONT_XS,
            ).pack(side="left")

            self.detail_text = ctk.CTkTextbox(
                detail_frame,
                height=120,
                wrap="word",
                font=self.FONT_XS,
                border_width=0,
                activate_scrollbars=True,
            )
            self.detail_text.pack(fill="x", pady=(4, 0))
            self.detail_text.insert(
                "1.0", "点击节点弹出详情窗口 | 双击展开邻域 | 大图模式"
            )
            self.detail_text.configure(state="disabled")

    def _build_search_toolbar(self):
        toolbar = ctk.CTkFrame(self, corner_radius=0)
        toolbar.pack(fill="x", padx=0, pady=0)

        row1 = ctk.CTkFrame(toolbar, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=6)

        ctk.CTkLabel(
            row1, text="搜索:", font=self.FONT_SM
        ).pack(side="left")
        self.search_var = tk.StringVar()
        self.search_entry = ctk.CTkEntry(
            row1,
            textvariable=self.search_var,
            width=240,
            font=self.FONT_SM,
            placeholder_text="输入节点名称…",
        )
        self.search_entry.pack(side="left", padx=(6, 6))
        self.search_entry.bind("<Return>", lambda e: self._do_search())

        ctk.CTkButton(
            row1, text="搜索", font=self.FONT_XS, width=60, command=self._do_search
        ).pack(side="left", padx=(0, 4))
        ctk.CTkButton(
            row1, text="重置", font=self.FONT_XS, width=60, command=self._reset_view
        ).pack(side="left", padx=(0, 4))
        ctk.CTkButton(
            row1, text="← 后退", font=self.FONT_XS, width=70, command=self._go_back
        ).pack(side="left", padx=(4, 4))

        ctk.CTkLabel(
            row1, text="布局:", font=self.FONT_XS
        ).pack(side="left", padx=(12, 2))
        self._layout_var = tk.StringVar(value="circular")
        self._layout_menu = ctk.CTkOptionMenu(
            row1,
            values=["spring", "circular", "kamada_kawai", "shell"],
            variable=self._layout_var,
            width=120,
            font=self.FONT_XS,
        )
        self._layout_menu.pack(side="left")
        ctk.CTkButton(
            row1,
            text="应用布局",
            font=self.FONT_XS,
            width=80,
            command=self._redraw_current,
        ).pack(side="left", padx=(4, 0))

        ctk.CTkButton(
            row1,
            text="📷 导出PNG",
            font=self.FONT_XS,
            width=90,
            command=self._export_png,
        ).pack(side="right", padx=(0, 4))

    def _build_edge_filter_bar(self):
        bar = ctk.CTkFrame(self, corner_radius=0, fg_color=("gray95", "gray20"))
        bar.pack(fill="x", padx=0, pady=0)

        ctk.CTkLabel(bar, text="显示:", font=self.FONT_XS).pack(
            side="left", padx=(10, 4)
        )

        self._edge_mode_labels = ["全部关系", "证型关联", "药材关联", "方剂关联", "三节点关联", "禁忌"]
        self._edge_mode_map = {
            "全部关系": "all", "证型关联": "syndrome", "药材关联": "herb",
            "方剂关联": "formula", "三节点关联": "triple", "禁忌": "incompatible",
        }
        self._seg_edge = ctk.CTkSegmentedButton(
            bar,
            values=self._edge_mode_labels,
            font=self.FONT_XS,
            command=self._on_edge_seg_change,
        )
        self._seg_edge.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self._seg_edge.set("全部关系")
        self._edge_mode = tk.StringVar(value="all")

    def _on_edge_seg_change(self, selected_label: str):
        self._edge_mode.set(self._edge_mode_map.get(selected_label, "all"))
        self._apply_edge_mode()

    def _apply_edge_mode(self):
        self._edge_mode.get()
        for v in self.type_vars.values():
            v.set(True)
        self._redraw_current()

    def _get_active_edges(self):
        mode = self._edge_mode.get()
        all_edges = {
            "belongs_to",
            "recommends_herb",
            "recommends_formula",
            "composes",
            "associated_with_herb",
            "incompatible_with",
            "co_occurs_with",
        }
        if mode == "all":
            return all_edges
        elif mode == "syndrome":
            return {"belongs_to", "recommends_herb", "recommends_formula"}
        elif mode == "herb":
            return {"associated_with_herb", "composes", "co_occurs_with"}
        elif mode == "formula":
            return {"recommends_formula", "composes", "associated_with_herb"}
        elif mode == "triple":
            return {"belongs_to", "recommends_herb", "composes", "associated_with_herb"}
        elif mode == "incompatible":
            return {"incompatible_with"}
        return all_edges

    def _build_compact_toolbar(self):
        toolbar = ctk.CTkFrame(self, corner_radius=0)
        toolbar.pack(fill="x", padx=0, pady=0)

        ctk.CTkLabel(
            toolbar, text="🔍", font=self.FONT_SM
        ).pack(side="left", padx=(6, 2))
        self.search_var = tk.StringVar()
        self.search_entry = ctk.CTkEntry(
            toolbar,
            textvariable=self.search_var,
            width=180,
            font=self.FONT_XS,
            placeholder_text="搜索节点…",
        )
        self.search_entry.pack(side="left", padx=(4, 4))
        self.search_entry.bind("<Return>", lambda e: self._do_search())
        ctk.CTkButton(
            toolbar, text="搜索", font=self.FONT_XS, width=50, command=self._do_search
        ).pack(side="left", padx=(0, 4))
        ctk.CTkButton(
            toolbar, text="重置", font=self.FONT_XS, width=50, command=self._reset_view
        ).pack(side="left")
        ctk.CTkButton(
            toolbar, text="大图", font=self.FONT_XS, width=60,
            command=self._open_graph_popup,
        ).pack(side="left", padx=(4, 0))

    type_vars: dict = {}

    def _build_legend(self, parent):
        legend_names = {
            "Symptom": "症状",
            "Syndrome": "证型",
            "Herb": "药材",
            "Formula": "方剂",
            "PatentMedicine": "中成药",
            "FolkRemedy": "验方",
        }
        # CTk fg_color 可能是元组 ("light", "dark") 或 "transparent"
        raw = parent.cget("fg_color")
        if isinstance(raw, (tuple, list)):
            bg_color = raw[1]  # 暗色值
        else:
            bg_color = "#2b2b2b"  # 兜底（Canvas 不支持 transparent）
        for tname, color in NODE_COLORS.items():
            if tname in {"Category", "Meridian", "Nature", "Flavor"}:
                continue
            label = legend_names.get(tname, tname)
            dot = tk.Canvas(
                parent, width=12, height=12, bg=bg_color,
                highlightthickness=0,
            )
            dot.create_oval(1, 1, 11, 11, fill=color, outline="")
            dot.pack(side="left", padx=(8, 2))
            tk.Label(
                parent, text=label, bg=bg_color, fg="#e0e0e0",
                font=("Microsoft YaHei UI", 16),
            ).pack(side="left")

    # 图谱加载与绘制

    def _load_graph(self):
        if self.graph is not None:
            return
        g, q = get_kg()
        self.graph = g
        self.query = q

    def _load_graph_async(self, on_done=None):
        if self.graph is not None:
            if on_done:
                on_done()
            return
        if self._loading:
            if on_done:
                self._pending_callback = on_done
            return

        self._loading = True
        self._pending_callback = on_done
        self._show_loading("正在加载知识图谱...")

        def _worker():
            try:
                g, q = get_kg()
                self.after(0, self._on_graph_loaded, g, q)
            except Exception as exc:
                logger.exception("知识图谱异步加载失败")
                self.after(0, self._on_graph_load_error, str(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_graph_loaded(self, g, q):
        self.graph = g
        self.query = q
        self._loading = False
        callback = self._pending_callback
        self._pending_callback = None
        if callback:
            try:
                callback()
            except Exception as exc:
                logger.exception("KG 渲染回调异常")
                self._show_loading(f"⚠ 知识图谱渲染失败: {exc}")

    def _on_graph_load_error(self, err_msg):
        self._loading = False
        self._pending_callback = None
        self._show_loading(f"⚠ 知识图谱加载失败: {err_msg}")

    def _show_loading(self, text="正在加载..."):
        if self._tk_canvas:
            self._tk_canvas.delete("all")
            self._tk_canvas.update_idletasks()
            cw = max(self._tk_canvas.winfo_width(), 800)
            ch = max(self._tk_canvas.winfo_height(), 600)
            self._tk_canvas.create_text(
                cw / 2,
                ch / 2,
                text=text,
                font=("Microsoft YaHei UI", 14),
                fill="#999",
                justify="center",
            )
        else:
            if self.ax:
                self.ax.clear()
                self.ax.text(
                    0.5, 0.5, text, transform=self.ax.transAxes,
                    ha="center", va="center", fontsize=14, color="#999",
                )
                self.ax.set_xlim(0, 1)
                self.ax.set_ylim(0, 1)
                self.ax.axis("off")

    def _show_overview_graph(self):
        if self.graph is None:
            self._load_graph_async(on_done=self._show_overview_graph)
            return
        self._render_overview()

    def _render_overview(self):
        g = self.graph
        formula_nodes = [n for n in g.nodes if g.nodes[n].get("type") == "Formula"]
        formula_nodes.sort(key=lambda n: g.degree(n), reverse=True)
        formula_nodes = formula_nodes[:15]
        if not formula_nodes:
            self._show_placeholder()
            return

        all_nodes = set(formula_nodes)
        for fn in formula_nodes:
            for pred in g.predecessors(fn):
                if g.nodes[pred].get("type") == "Herb":
                    if any(
                        d.get("type") == "composes"
                        for _, v, d in g.edges(pred, data=True)
                        if v == fn
                    ):
                        all_nodes.add(pred)

        for fn in formula_nodes:
            for pred in g.predecessors(fn):
                if g.nodes[pred].get("type") == "Syndrome":
                    if any(
                        d.get("type") == "recommends_formula"
                        for _, v, d in g.edges(pred, data=True)
                        if v == fn
                    ):
                        all_nodes.add(pred)

        all_nodes = list(all_nodes)
        if len(all_nodes) < 3:
            self._show_placeholder()
            return

        center = formula_nodes[0]
        self._show_subgraph_given_nodes(center, all_nodes)

    def _show_placeholder(self):
        if self._tk_canvas:
            self._tk_canvas.delete("all")
            self._tk_canvas.update_idletasks()
            cw = max(self._tk_canvas.winfo_width(), 800)
            ch = max(self._tk_canvas.winfo_height(), 600)
            self._tk_canvas.create_text(
                cw / 2,
                ch / 2,
                text="正在加载知识图谱...\n请稍候",
                font=("Microsoft YaHei UI", 16),
                fill="#999",
                justify="center",
            )
        else:
            if self.ax:
                self.ax.clear()
                self.ax.text(
                    0.5, 0.5, "正在加载知识图谱...\n\n请稍候",
                    transform=self.ax.transAxes, ha="center", va="center",
                    fontsize=14, color="#999",
                )
                self.ax.set_xlim(0, 1)
                self.ax.set_ylim(0, 1)
                self.ax.axis("off")

    def _do_search(self):
        query = self.search_var.get().strip()
        if not query:
            self._reset_view()
            return

        if self.graph is None:
            self._load_graph_async(
                on_done=lambda: (
                    self._do_search() if self.search_var.get().strip() else None
                )
            )
            return

        g = self.graph
        matches = [n for n in g.nodes if query in g.nodes[n].get("name", "")]
        if not matches:
            self._set_detail(f"未找到包含「{query}」的节点")
            return

        center = matches[0]
        self._show_subgraph(center, matches[:10])

    def _show_subgraph(self, center_node: str, extra_nodes: list[str] | None = None):
        g = self.graph
        if g is None:
            self._load_graph()
            g = self.graph

        neighbors = set()
        for _, nbr in g.edges(center_node):
            neighbors.add(nbr)
        for nbr, _ in g.in_edges(center_node):
            neighbors.add(nbr)
        if extra_nodes:
            for n in extra_nodes:
                neighbors.add(n)
        neighbors.add(center_node)

        self._show_subgraph_given_nodes(center_node, list(neighbors))

    def _show_subgraph_all(self, center_node: str):
        g = self.graph
        if g is None:
            self._load_graph()
            g = self.graph

        if self._current_subgraph is not None:
            self._expand_history.append(set(self._current_subgraph.nodes))

        visited = {center_node}
        for _, nbr, ed in g.edges(center_node, data=True):
            etype = ed.get("type", "")
            if etype in {
                "has_nature", "has_flavor", "has_meridian",
                "belongs_to_category", "co_occurs_with",
            }:
                continue
            visited.add(nbr)
        for nbr, _, ed in g.in_edges(center_node, data=True):
            etype = ed.get("type", "")
            if etype in {
                "has_nature", "has_flavor", "has_meridian",
                "belongs_to_category", "co_occurs_with",
            }:
                continue
            visited.add(nbr)

        logger.info(
            "_show_subgraph_all: 中心 %s (%s), 共 %d 节点",
            g.nodes[center_node].get("name", center_node),
            g.nodes[center_node].get("type", "?"),
            len(visited),
        )
        type_counts = {}
        for n in visited:
            t = g.nodes[n].get("type", "?")
            type_counts[t] = type_counts.get(t, 0) + 1
        logger.info("_show_subgraph_all: 节点类型分布: %s", type_counts)

        if hasattr(self, "_layout_var"):
            self._layout_var.set("circular")
        self._show_subgraph_given_nodes(center_node, list(visited))

    def _expand_node(self, node_id: str):
        if self.graph is None or self._current_subgraph is None:
            return
        g = self.graph
        current = set(self._current_subgraph.nodes)
        if node_id not in current:
            return

        self._expand_history.append(set(current))

        new_nodes = set(current)
        for _, nbr in g.edges(node_id):
            new_nodes.add(nbr)
        for nbr, _ in g.in_edges(node_id):
            new_nodes.add(nbr)

        if len(new_nodes) == len(current):
            return

        self._show_subgraph_given_nodes(node_id, list(new_nodes))

    def _go_back(self):
        logger.info("_go_back: 历史深度 %d", len(self._expand_history))
        if not self._expand_history:
            logger.info("_go_back: 历史为空，无法后退")
            return
        prev = self._expand_history.pop()
        logger.info(
            "_go_back: 取出历史状态，剩余 %d，节点数 %d",
            len(self._expand_history),
            len(prev),
        )
        if self._current_subgraph is not None:
            center = None
            for n in prev:
                ntype = self.graph.nodes[n].get("type", "")
                if ntype in ("Syndrome", "Symptom"):
                    center = n
                    break
            if center is None:
                center = next(iter(prev))
            self._show_subgraph_given_nodes(center, list(prev))

    # ── 右键菜单 ──

    def _ctx_show_detail(self):
        if self._selected_node:
            self._show_node_detail(self._selected_node)

    def _ctx_expand(self):
        if self._selected_node:
            self._expand_node(self._selected_node)

    def _ctx_set_center(self):
        if self._selected_node and self._current_subgraph is not None:
            self._show_subgraph_given_nodes(
                self._selected_node, list(self._current_subgraph.nodes)
            )

    def _ctx_copy_name(self):
        if self._selected_node:
            name = self.graph.nodes[self._selected_node].get(
                "name", self._selected_node
            )
            self.clipboard_clear()
            self.clipboard_append(name)

    # ── 布局切换 ──

    def _redraw_current(self):
        if self._current_subgraph is None:
            logger.info("_redraw_current: 无当前子图，跳过")
            return
        logger.info(
            "_redraw_current: 清除布局缓存，重绘 %s 节点",
            self._current_subgraph.number_of_nodes(),
        )
        active_types = {t for t, v in self.type_vars.items() if v.get()}
        if len(active_types) < 2:
            logger.warning("_redraw_current: 至少选择2种节点类型，自动恢复全部")
            for v in self.type_vars.values():
                v.set(True)
        sub_key = frozenset(self._current_subgraph.nodes)
        if sub_key in self._pos_cache:
            del self._pos_cache[sub_key]
        center = self._selected_node or list(self._current_subgraph.nodes)[0]
        self._show_subgraph_given_nodes(center, list(self._current_subgraph.nodes))

    def _get_layout(self, sub, pos=None):
        layout_name = getattr(self, "_layout_var", None)
        layout_type = layout_name.get() if layout_name else "spring"

        if layout_type == "spring":
            return nx.spring_layout(sub, k=0.8, iterations=50, seed=42, pos=pos)
        elif layout_type == "circular":
            return nx.circular_layout(sub)
        elif layout_type == "kamada_kawai":
            return nx.kamada_kawai_layout(sub)
        elif layout_type == "shell":
            if self.graph is None:
                return nx.shell_layout(sub)
            groups = []
            for t in ["Syndrome", "Symptom", "Herb", "Formula"]:
                nodes = [n for n in sub.nodes if self.graph.nodes[n].get("type") == t]
                if nodes:
                    groups.append(nodes)
            return (
                nx.shell_layout(sub, nlist=groups) if groups else nx.shell_layout(sub)
            )
        return nx.spring_layout(sub, k=0.8, iterations=50, seed=42)

    # ── 导出PNG ──

    def _export_png(self):
        from tkinter import filedialog

        path = filedialog.asksaveasfilename(
            defaultextension=".eps",
            filetypes=[("EPS 图片", "*.eps")],
            title="导出知识图谱",
        )
        if path:
            self._tk_canvas.postscript(file=path)
            tk.messagebox.showinfo("导出", f"已保存到:\n{path}")

    # ── 子图绘制 ──

    def _show_subgraph_given_nodes(self, center: str, node_list: list):
        g = self.graph
        if g is None:
            self._load_graph()
            g = self.graph

        logger.info(
            "_show_subgraph_given_nodes: 输入 %d 节点, 中心=%s",
            len(node_list),
            center,
        )

        active_types = {
            t for t, v in getattr(self, "type_vars", {}).items() if v.get()
        }
        logger.info("_show_subgraph_given_nodes: 活跃节点类型: %s", active_types)
        if active_types:
            filtered = [
                n for n in node_list if g.nodes[n].get("type", "") in active_types
            ]
            filtered_out = len(node_list) - len(filtered)
            if filtered_out > 0:
                logger.info(
                    "_show_subgraph_given_nodes: 过滤掉 %d 节点（类型不匹配）",
                    filtered_out,
                )
            if center not in filtered:
                filtered.append(center)
            node_list = filtered

        sub = g.subgraph(node_list).copy()
        logger.info(
            "_show_subgraph_given_nodes: 子图节点=%d, 边=%d",
            sub.number_of_nodes(),
            sub.number_of_edges(),
        )

        sub_key = frozenset(sub.nodes)
        if sub_key in self._pos_cache:
            pos = self._pos_cache[sub_key]
        else:
            pos = self._get_layout(sub)
            self._pos_cache[sub_key] = pos

        self._current_subgraph = sub
        self._all_node_ids = list(sub.nodes)

        logger.info(
            "渲染子图: %s 节点, %s 边",
            sub.number_of_nodes(),
            sub.number_of_edges(),
        )
        try:
            self._draw_canvas(sub, pos, center)
        except Exception as exc:
            logger.exception("Canvas 渲染失败: %s", exc)
            self._set_detail(f"❌ 渲染失败: {exc}")
        self._show_node_detail(center)

    # ── Canvas 渲染 ──

    def _draw_canvas(self, sub, pos, center_node=None):
        logger.info("_draw_canvas: 开始渲染 %s 节点", sub.number_of_nodes())
        c = self._tk_canvas
        c.delete("all")
        self._canvas_items = {}
        self._edge_items = []
        self._zoom_level = 1.0
        g = self.graph

        xs = [pos[n][0] for n in sub.nodes]
        ys = [pos[n][1] for n in sub.nodes]
        if not xs:
            logger.warning("_draw_canvas: 无节点可渲染")
            return
        pad_x = max((max(xs) - min(xs)) * 0.3, 1)
        pad_y = max((max(ys) - min(ys)) * 0.3, 1)
        x_min, x_max = min(xs) - pad_x, max(xs) + pad_x
        y_min, y_max = min(ys) - pad_y, max(ys) + pad_y
        c.update_idletasks()
        cw = max(c.winfo_width(), 800)
        ch = max(c.winfo_height(), 600)
        scale_x = cw / (x_max - x_min) if x_max > x_min else 1
        scale_y = ch / (y_max - y_min) if y_max > y_min else 1
        s = min(scale_x, scale_y) * 0.85
        cx, cy = cw / 2, ch / 2
        mx, my = (x_min + x_max) / 2, (y_min + y_max) / 2

        def to_canvas(x, y):
            return (cx + (x - mx) * s, cy + (y - my) * s)

        self._cv_s = s
        self._cv_cx, self._cv_cy = cx, cy
        self._cv_mx, self._cv_my = mx, my
        self._cv_pos = pos

        active_edges = self._get_active_edges()
        drawn = 0
        filtered_edges = 0
        edge_type_count = {}
        for u, v, data in sub.edges(data=True):
            etype = data.get("type", "")
            if etype not in active_edges:
                filtered_edges += 1
                continue
            edge_type_count[etype] = edge_type_count.get(etype, 0) + 1
            drawn += 1
            ecolor = EDGE_COLORS.get(etype, "#ccc")
            x1, y1 = to_canvas(pos[u][0], pos[u][1])
            x2, y2 = to_canvas(pos[v][0], pos[v][1])
            lid = c.create_line(x1, y1, x2, y2, fill=ecolor, width=1, tags=("edge",))
            self._edge_items.append(lid)

        logger.info(
            "_draw_canvas: 边绘制: 总%d, 已画%d(按类型: %s), 过滤%d(模式外)",
            len(list(sub.edges())),
            drawn,
            edge_type_count,
            filtered_edges,
        )

        NODE_COLORS_HEX = {
            "Symptom": "#e74c3c",
            "Syndrome": "#9b59b6",
            "Herb": "#27ae60",
            "Formula": "#2980b9",
            "PatentMedicine": "#f39c12",
            "FolkRemedy": "#95a5a6",
            "Category": "#1abc9c",
            "Meridian": "#e67e22",
            "Nature": "#e91e63",
            "Flavor": "#00bcd4",
        }
        NODE_RADIUS = {"Syndrome": 20, "Formula": 17, "Symptom": 14, "Herb": 12}
        for n in sub.nodes:
            ntype = g.nodes[n].get("type", "Unknown")
            name = g.nodes[n].get("name", "")
            x, y = to_canvas(pos[n][0], pos[n][1])
            r = NODE_RADIUS.get(ntype, 10)
            color = NODE_COLORS_HEX.get(ntype, "#bdc3c7")

            c.create_oval(
                x - r + 1, y - r + 1, x + r + 1, y + r + 1,
                fill="#404040", outline="", tags=("shadow",),
            )

            oid = c.create_oval(
                x - r, y - r, x + r, y + r,
                fill=color, outline="#ffffff", width=2,
                tags=(f"node_{n}", "node"),
            )

            label = name if len(name) <= 4 else name[:3] + "…"
            fs = max(8, min(12, int(r * 0.75)))
            tid = c.create_text(
                x, y, text=label,
                font=("Microsoft YaHei UI", fs, "bold"),
                fill="#ffffff", anchor="center",
                tags=(f"label_{n}", "label"),
            )
            self._canvas_items[n] = {
                "oval": oid, "text": tid, "r": r, "color": color,
                "font_size": fs,
            }

        if center_node and center_node in self._canvas_items:
            ci = self._canvas_items[center_node]
            c.itemconfig(ci["oval"], width=3, outline="#2c3e50")

        self._cv_node_tags = {f"node_{n}": n for n in sub.nodes}

        self._set_detail(
            f"📊 共 {sub.number_of_nodes()} 节点 / {sub.number_of_edges()} 条边\n"
            f"💡 拖拽平移 · 滚轮缩放 · 点击节点查看详情"
        )

    def _cv_get_node(self, x, y):
        overlap = self._tk_canvas.find_overlapping(x - 5, y - 5, x + 5, y + 5)
        for item in overlap:
            tags = self._tk_canvas.gettags(item)
            for t in tags:
                if t.startswith("node_"):
                    return t[5:]
        return None

    def _cv_on_press(self, e):
        logger.info("_cv_on_press: (%d, %d)", e.x, e.y)
        self._cv_drag_start = (e.x, e.y)
        node = self._cv_get_node(e.x, e.y)
        self._cv_press_node = node
        if node:
            logger.info("_cv_on_press: 点击节点 %s", node)

    def _cv_on_double_click(self, e):
        """双击展开邻域"""
        node = self._cv_get_node(e.x, e.y)
        if node:
            logger.info("_cv_on_double_click: 双击节点 %s", node)
            self._selected_node = node
            self._show_subgraph_all(node)

    def _cv_on_drag(self, e):
        if self._cv_drag_start:
            dx = e.x - self._cv_drag_start[0]
            dy = e.y - self._cv_drag_start[1]
            self._tk_canvas.move("all", dx, dy)
            self._cv_drag_start = (e.x, e.y)

    def _cv_on_release(self, e):
        # 如果鼠标没怎么移动（不是拖拽），视为单击 → 显示节点详情
        if self._cv_drag_start is not None:
            dx = abs(e.x - self._cv_drag_start[0])
            dy = abs(e.y - self._cv_drag_start[1])
            if dx < 5 and dy < 5 and self._cv_press_node:
                self._selected_node = self._cv_press_node
                self._show_node_detail(self._cv_press_node)
        self._cv_drag_start = None
        self._cv_press_node = None

    def _cv_on_wheel(self, e):
        """滚轮缩放。

        return "break" 至关重要：嵌入首页时外层 CTkScrollableFrame 用
        bind_all(add="+") 全局监听滚轮，若不中断传播，缩放同时会滚动
        页面（即"滚动条和缩放冲突"的根因）。
        """
        factor = 1.1 if e.delta > 0 else 0.9
        new_zoom = self._zoom_level * factor
        if not (0.3 <= new_zoom <= 4.0):
            return "break"
        self._zoom_level = new_zoom

        self._tk_canvas.scale("all", e.x, e.y, factor, factor)
        # 字号 = 基准字号 × 当前缩放（基准存于 _canvas_items，避免
        # 逐次解析 itemcget(font) 造成的累积漂移）
        for _n, ci in self._canvas_items.items():
            tid = ci.get("text")
            if tid is None:
                continue
            new_sz = max(5, min(26, round(ci["font_size"] * self._zoom_level)))
            try:
                self._tk_canvas.itemconfig(
                    tid, font=("Microsoft YaHei UI", new_sz, "bold")
                )
            except Exception as exc:
                logger.warning("Canvas 文字缩放失败: %s", exc)
        return "break"

    def _cv_on_motion(self, e):
        node = self._cv_get_node(e.x, e.y)
        if node and node != getattr(self, "_cv_hover", None):
            self._cv_hover = node
            self._tk_canvas.config(cursor="hand2")
        elif not node and getattr(self, "_cv_hover", None):
            self._cv_hover = None
            self._tk_canvas.config(cursor="")

    def _cv_on_right_click(self, e):
        node = self._cv_get_node(e.x, e.y)
        if node:
            self._selected_node = node
            # 详情改由菜单"🔍 查看详情"触发（避免弹窗与菜单同时出现）
            self._context_menu.tk_popup(e.x_root, e.y_root)

    def _show_node_detail(self, node_id: str):
        """节点详情展示。

        嵌入首页模式：弹出独立详情窗口（原 120px 底部小条太局促）；
        独立图谱模式：写入右侧 300px 详情面板。
        """
        if not self.standalone:
            self._open_detail_popup(node_id)
            return
        text = self._node_detail_text(node_id)
        self._set_detail(text)

    def _node_detail_text(self, node_id: str) -> str:
        g = self.graph
        data = g.nodes[node_id]
        name = data.get("name", node_id)
        ntype = data.get("type", "Unknown")

        lines = []
        type_label = {
            "Symptom": "症状", "Syndrome": "证型", "Herb": "药材",
            "Formula": "方剂", "PatentMedicine": "中成药", "FolkRemedy": "验方",
        }
        lines.append(f"【{type_label.get(ntype, ntype)}】{name}")
        lines.append("─" * 40)

        if ntype == "Herb":
            cat = data.get("category", "")
            if cat:
                lines.append(f"  分类: {cat}")
            eff = data.get("efficacy", "")
            if eff:
                lines.append(f"  功效: {eff}")
            ind = data.get("indications", "")
            if ind:
                lines.append(f"  主治: {ind}")
            usage = data.get("usage", "")
            if usage:
                lines.append(f"  用法: {usage}")
            cont = data.get("contraindications", "")
            if cont:
                lines.append(f"  禁忌: {cont}")
            source = data.get("source", "")
            if source:
                lines.append(f"  来源: {source}")
            for _, nbr, ed in g.edges(node_id, data=True):
                et = ed.get("type", "")
                nb_name = g.nodes[nbr].get("name", "")
                if et == "has_nature":
                    lines.append(f"  性: {nb_name}")
                elif et == "has_flavor":
                    lines.append(f"  味: {nb_name}")
                elif et == "has_meridian":
                    lines.append(f"  归经: {nb_name}")
            if data.get("toxic"):
                lines.append("  ⚠️ 有毒药材")
            if data.get("pregnancy_forbidden"):
                lines.append("  ⚠️ 孕妇禁用")
            incomp_set = set()
            for _, nbr, ed in g.edges(node_id, data=True):
                if ed.get("type") == "incompatible_with":
                    incomp_set.add(g.nodes[nbr].get("name", ""))
            if incomp_set:
                lines.append(f"  相反相畏: {'、'.join(sorted(incomp_set)[:6])}")
        elif ntype == "Formula":
            sb = data.get("source_book", "")
            if sb:
                lines.append(f"  来源: {sb}")
            comp = data.get("composition", "")
            if comp:
                lines.append(f"  组成: {comp[:80]}{'…' if len(comp) > 80 else ''}")
            func = data.get("functions", "")
            if func:
                lines.append(f"  功能: {func}")
            ind = data.get("indications", "")
            if ind:
                lines.append(f"  主治: {ind}")
            usage = data.get("usage", "")
            if usage:
                lines.append(f"  用法: {usage}")
            dos = data.get("dosage", "")
            if dos:
                lines.append(f"  用量: {dos}")
        elif ntype == "Syndrome":
            desc = data.get("description", "")
            if desc:
                lines.append(f"  证候: {desc}")
            treat = data.get("treatment", "")
            if treat:
                lines.append(f"  治则: {treat}")
            herbs = []
            for _, nbr, ed in g.edges(node_id, data=True):
                if ed.get("type") == "recommends_herb":
                    herbs.append(g.nodes[nbr].get("name", ""))
            if herbs:
                lines.append(f"  推荐药材: {'、'.join(herbs[:6])}")
            formulas = []
            for _, nbr, ed in g.edges(node_id, data=True):
                if ed.get("type") == "recommends_formula":
                    formulas.append(g.nodes[nbr].get("name", ""))
            if formulas:
                lines.append(f"  推荐方剂: {'、'.join(formulas[:4])}")
        elif ntype == "Symptom":
            lines.append(f"  标准症状名: {name}")
            syndromes = []
            for _, nbr, ed in g.edges(node_id, data=True):
                if ed.get("type") == "belongs_to":
                    syndromes.append(g.nodes[nbr].get("name", ""))
            if syndromes:
                lines.append(f"  所属证型: {'、'.join(syndromes[:5])}")

        degree = g.degree(node_id)
        lines.append(f"  关联数: {degree}")

        neighbor_types = {}
        for _, nbr in g.edges(node_id):
            nt = g.nodes[nbr].get("type", "?")
            neighbor_types[nt] = neighbor_types.get(nt, 0) + 1
        if neighbor_types:
            parts = [f"{nt}×{cnt}" for nt, cnt in sorted(neighbor_types.items())]
            lines.append(f"  相邻: {', '.join(parts)}")

        text = "\n".join(lines)
        return text

    def _open_detail_popup(self, node_id: str):
        """节点详情弹窗：大窗展示完整知识，支持展开邻域/设为中心联动主图。"""
        g = self.graph
        data = g.nodes[node_id]
        name = data.get("name", node_id)
        ntype = data.get("type", "Unknown")
        text = self._node_detail_text(node_id)

        win = ctk.CTkToplevel(self)
        win.title(f"知识详情 · {name}")
        win.geometry("560x480")
        win.transient(self.winfo_toplevel())
        win.attributes("-topmost", True)
        win.after(180, lambda: win.attributes("-topmost", False))
        win.lift()
        win.focus_force()

        type_label = {
            "Symptom": "症状", "Syndrome": "证型", "Herb": "药材",
            "Formula": "方剂", "PatentMedicine": "中成药", "FolkRemedy": "验方",
        }
        header = ctk.CTkFrame(win, fg_color=(" #f3f7f5", "#22352b"))
        header.pack(fill="x", padx=10, pady=(10, 0))
        ctk.CTkLabel(
            header,
            text=f"{name}（{type_label.get(ntype, ntype)}）",
            font=("Microsoft YaHei UI", 15, "bold"),
            anchor="w",
        ).pack(fill="x", padx=12, pady=8)

        body = ctk.CTkTextbox(win, wrap="word", font=("Microsoft YaHei UI", 12))
        body.pack(fill="both", expand=True, padx=10, pady=8)
        body.insert("1.0", text)
        body.configure(state="disabled")

        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(0, 10))

        def _expand():
            win.destroy()
            self._selected_node = node_id
            self._show_subgraph_all(node_id)

        def _center():
            win.destroy()
            self._selected_node = node_id
            self._show_subgraph(node_id)

        ctk.CTkButton(
            btn_row, text="⬆ 展开邻域", width=110, command=_expand
        ).pack(side="left", padx=(0, 6))
        ctk.CTkButton(
            btn_row, text="🎯 设为中心", width=110, command=_center
        ).pack(side="left", padx=(0, 6))
        ctk.CTkButton(
            btn_row, text="关闭", width=80,
            fg_color="gray30", hover_color="gray20",
            command=win.destroy,
        ).pack(side="right")

    def _open_graph_popup(self):
        """图谱大图弹窗：全屏画布浏览，拖拽平移 + 滚轮缩放（无滚动条冲突）。"""
        win = ctk.CTkToplevel(self)
        win.title("知识图谱 · 大图浏览")
        win.geometry("1000x700")
        win.transient(self.winfo_toplevel())
        win.after(120, win.lift)

        big = KnowledgeGraphView(win, standalone=True)
        big.graph = self.graph
        big.query = self.query
        big.pack(fill="both", expand=True)
        big._load_graph_async(
            on_done=lambda: self._restore_popup_state(big)
        )

    def _restore_popup_state(self, big_view):
        """弹窗图谱恢复主视图当前状态（搜索词 + 当前子图）。"""
        q = self.search_var.get().strip()
        if q:
            big_view.search_var.set(q)
            big_view._do_search()
        elif self._current_subgraph is not None and self._selected_node:
            big_view._show_subgraph(self._selected_node)
        else:
            big_view._show_overview_graph()

    def _set_detail(self, text: str):
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", text)
        self.detail_text.configure(state="disabled")

    def _update_filters(self):
        if self.search_var.get().strip():
            self._do_search()

    def _reset_view(self):
        self.search_var.set("")
        self._set_detail("点击图谱中的节点查看详情")
        self._current_subgraph = None
        self._expand_history = []
        self._show_overview_graph()

    def reload_graph(self, force_rebuild=False):
        global KG_INSTANCE
        if force_rebuild:
            KG_INSTANCE = None
            self.graph = None
            self.query = None
            self._load_graph_async(on_done=self._reset_view)
        else:
            self.graph = None
            self.query = None
            self._reset_view()

    # 外部接口：展示查询结果

    def show_query_result(
        self,
        symptoms: list[str],
        syndrome_name: str | None = None,
        herbs: list[str] | None = None,
        formulas: list[str] | None = None,
    ):
        if self.graph is None:
            self._load_graph_async(
                on_done=lambda: self.show_query_result(
                    symptoms, syndrome_name, herbs, formulas
                )
            )
            return
        g = self.graph

        target_nodes: list[str] = []
        if syndrome_name:
            nid = f"syndrome:{syndrome_name}"
            if g.has_node(nid):
                target_nodes.append(nid)

        for sym in symptoms:
            nid = f"symptom:{sym}"
            if g.has_node(nid):
                target_nodes.append(nid)

        if herbs:
            for h in herbs[:5]:
                nid = f"herb:{h}"
                if g.has_node(nid):
                    target_nodes.append(nid)

        if formulas:
            for f in formulas[:3]:
                nid = f"formula:{f}"
                if g.has_node(nid):
                    target_nodes.append(nid)

        if not target_nodes:
            self._set_detail(f"未在图谱中找到相关节点: {symptoms}")
            return

        # ── 按节点类型分组，构建聚焦查询子图 ──
        symptom_nodes = [n for n in target_nodes if g.nodes[n].get("type") == "Symptom"]
        syndrome_node = next(
            (n for n in target_nodes if g.nodes[n].get("type") == "Syndrome"), None
        )
        herb_nodes = [n for n in target_nodes if g.nodes[n].get("type") == "Herb"]
        formula_nodes = [n for n in target_nodes if g.nodes[n].get("type") == "Formula"]
        grouped = set(symptom_nodes + herb_nodes + formula_nodes)
        if syndrome_node:
            grouped.add(syndrome_node)
        other_nodes = [n for n in target_nodes if n not in grouped]

        center, node_list = self._build_query_subgraph(
            symptom_nodes + other_nodes, syndrome_node, herb_nodes, formula_nodes
        )
        if not node_list:
            self._set_detail(f"未在图谱中找到与 {symptoms} 相关的连通节点")
            return

        self.search_var.set(g.nodes[center].get("name", "") if center else "")
        self._show_subgraph_given_nodes(center, node_list)

    def _build_query_subgraph(
        self,
        symptom_nodes: list,
        syndrome_node: str | None,
        herb_nodes: list,
        formula_nodes: list,
    ) -> tuple[str | None, list]:
        """构建"查询关联子图"：只保留目标节点之间的边 + 必要的桥接邻居。

        修复原 _show_subgraph 路径的三个显示问题：
          1. 以症状为中心拉取全部邻居 → 节点爆炸、画面杂乱
          2. 推荐节点与中心无边时成为孤立漂浮点
          3. 中心选择不符合"辨证论治"语义（证型应居中）
        """
        g = self.graph
        core = [n for n in [syndrome_node] + symptom_nodes if n and g.has_node(n)]
        recommended = [
            n for n in herb_nodes + formula_nodes if n and g.has_node(n)
        ]
        if not core:
            # 症状/证型都不在图中：退化为中心取第一个有节点的推荐项
            core = recommended[:1]
            recommended = recommended[1:]
        if not core:
            return None, []

        nodes = set(core) | set(recommended)

        # 桥接：推荐节点若与 core 无边，找一个它与任一 core 节点的共同邻居
        # 加入子图作为"桥"，保证连通与语义可读（如：症状—证型—药材 的中介）
        for r in recommended:
            r_nb = set(g.successors(r)) | set(g.predecessors(r))
            if any(r_nb & (set(g.successors(c)) | set(g.predecessors(c))) for c in core):
                continue  # 已连通
            for c in core:
                c_nb = set(g.successors(c)) | set(g.predecessors(c))
                common = r_nb & c_nb
                if common:
                    nodes.add(sorted(common)[0])
                    break

        # .copy() 必须：subgraph 默认返回只读视图，下方 remove_nodes_from 会抛
        # "Frozen graph can't be modified"
        sub = g.subgraph(nodes).copy()

        # 剔除孤立点（core 节点保留，避免中心消失）
        isolated = [
            n for n, deg in sub.degree()
            if deg == 0 and n not in core
        ]
        if isolated:
            sub.remove_nodes_from(isolated)

        # 中心：证型优先（辨证论治语义），其次第一个症状
        center = None
        if syndrome_node and syndrome_node in sub.nodes:
            center = syndrome_node
        elif symptom_nodes:
            center = next((n for n in symptom_nodes if n in sub.nodes), None)
        if center is None:
            center = next(iter(sub.nodes)) if sub.number_of_nodes() else None
        return center, list(sub.nodes)


def open_kg_result_popup(parent, knowledge=None):
    """查询关联知识图谱弹窗：只画与本次查询症状/证型/方药相关的子图。

    knowledge 为桌面端 query() 返回的 knowledge 字段（含 symptoms/syndrome/
    herbs/formulas）。未查询时给出引导提示，不展示默认全图。
    """
    win = ctk.CTkToplevel(parent)
    win.title("知识图谱 · 查询关联视图")
    win.geometry("1080x720")
    try:
        win.transient(parent.winfo_toplevel())
    except Exception:
        pass

    view = KnowledgeGraphView(win, standalone=True)
    view.pack(fill="both", expand=True)

    def _after_load():
        k = knowledge or {}
        symptoms = k.get("symptoms") or []
        syn = k.get("syndrome")
        syn_name = syn.get("name") if isinstance(syn, dict) else None
        herbs = [h["name"] for h in k.get("herbs") or [] if isinstance(h, dict)]
        forms = [f["name"] for f in k.get("formulas") or [] if isinstance(f, dict)]
        if symptoms or herbs or forms:
            view.show_query_result(
                symptoms=symptoms,
                syndrome_name=syn_name,
                herbs=herbs,
                formulas=forms,
            )
        else:
            view._set_detail(
                "请先在首页进行症状查询。\n\n"
                "图谱将只展示与本次症状相关的证型、药材、方剂关联子图，"
                "而非默认全图。"
            )

    view._load_graph_async(on_done=_after_load)
    return win
