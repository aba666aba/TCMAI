"""
知识图谱可视化组件

嵌入 Tkinter 的交互式知识图谱查看器。
使用 matplotlib 渲染，支持搜索、缩放和节点点击查看详情。
"""
import logging
import math
import threading
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import networkx as nx

logger = logging.getLogger(__name__)

# ── 实体类型颜色方案 ──
NODE_COLORS = {
    "Symptom": "#e74c3c",       # 红 — 症状
    "Syndrome": "#9b59b6",      # 紫 — 证型
    "Herb": "#27ae60",          # 绿 — 药材
    "Formula": "#2980b9",       # 蓝 — 方剂
    "PatentMedicine": "#f39c12",# 橙 — 中成药
    "FolkRemedy": "#95a5a6",    # 灰 — 验方
    "Category": "#1abc9c",      # 青 — 分类
    "Meridian": "#e67e22",      # 橙黄 — 归经
    "Nature": "#e91e63",        # 粉 — 性
    "Flavor": "#00bcd4",        # 青蓝 — 味
}
NODE_SIZE_MAP = {
    "Symptom": 200, "Syndrome": 280, "Herb": 160,
    "Formula": 220, "PatentMedicine": 180, "FolkRemedy": 140,
    "Category": 120, "Meridian": 120, "Nature": 100, "Flavor": 100,
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

# ── 中文字体 ──
_CN_FONT = None
import os as _os
_win_fonts = [
    _os.path.expandvars(r"%WINDIR%\Fonts\msyh.ttc"),
    _os.path.expandvars(r"%WINDIR%\Fonts\simhei.ttf"),
    _os.path.expandvars(r"%WINDIR%\Fonts\msyhl.ttc"),
]
for _fp in _win_fonts:
    if _os.path.exists(_fp):
        from matplotlib.font_manager import FontProperties
        _CN_FONT = FontProperties(fname=_fp, size=11)
        # 同时设全局字体，消除 matplotlib 后端警告
        import matplotlib as _mpl
        _mpl.rcParams["font.sans-serif"] = [_CN_FONT.get_name()] + _mpl.rcParams["font.sans-serif"]
        _mpl.rcParams["axes.unicode_minus"] = False
        break


KG_INSTANCE = None       # 全局缓存图谱实例
_KG_LOCK = threading.Lock()  # 构建锁，防止并发构建


def get_kg(*, force_rebuild: bool = False):
    """懒加载知识图谱（磁盘缓存加速）

    Args:
        force_rebuild: True 时忽略缓存，强制重新构建。
    """
    global KG_INSTANCE
    if not force_rebuild and KG_INSTANCE is not None:
        return KG_INSTANCE

    with _KG_LOCK:
        # 双重检查
        if not force_rebuild and KG_INSTANCE is not None:
            return KG_INSTANCE

        import os, sys, pickle
        from pathlib import Path
        project_root = Path(__file__).resolve().parents[2]
        if str(project_root / "src") not in sys.path:
            sys.path.insert(0, str(project_root / "src"))
        os.environ["TCMAI_ROOT"] = str(project_root)

        cache_path = os.path.join(project_root, "data", "kg_cache.pkl")

        # 先从缓存加载（只要 pickle 可反序列化就用，不依赖 mtime）
        if not force_rebuild and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    g = pickle.load(f)
                from kg.query import KnowledgeGraphQuery
                q = KnowledgeGraphQuery(g)
                logger.info("知识图谱从缓存加载: %s 节点, %s 边",
                             g.number_of_nodes(), g.number_of_edges())
                KG_INSTANCE = (g, q)
                return KG_INSTANCE
            except Exception as exc:
                logger.warning("缓存加载失败(将重新构建): %s", exc)

        # 缓存不存在或不可用 → 重新构建
        from kg.builder import TCMKnowledgeGraph
        logger.info("构建知识图谱（首次较慢，后续将使用缓存）...")
        kg = TCMKnowledgeGraph()
        kg.build()
        g = kg.graph
        q = KnowledgeGraphQuery(g)
        # 缓存到磁盘供下次使用
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(g, f, protocol=4)
            logger.info("知识图谱已缓存到磁盘: %s (%d 节点, %d 边)",
                         cache_path, g.number_of_nodes(), g.number_of_edges())
        except Exception as exc:
            logger.warning("缓存写入失败: %s", exc)
        KG_INSTANCE = (g, q)
        return KG_INSTANCE


class KnowledgeGraphView(ttk.Frame):
    """知识图谱可视化组件"""

    def __init__(self, parent, standalone=True, **kwargs):
        """
        Args:
            standalone: True=独立标签页（完整UI），False=嵌入首页（紧凑UI+自动展示结果）
        """
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
        self._expand_history = []  # 展开历史栈
        self._all_node_ids = []     # 当前子图节点列表
        # 缩放相关
        self._base_x_range = None      # 基准 x 范围（用于计算缩放因子）
        self._current_zoom = 1.0       # 当前缩放因子
        # 快速 Canvas 渲染（替代 matplotlib，极致性能）
        self._tk_canvas = None          # tkinter Canvas 控件
        self._canvas_items = {}         # {node_id: {"oval": id, "text": id}}
        self._edge_items = []           # [line_id, ...]
        self._canvas_bg = None
        # Canvas 交互状态
        self._drag_start = None         # 拖拽平移起始点
        # 右侧知识面板
        self.knowledge_text = None
        # 异步加载相关
        self._loading = False
        self._pending_callback = None  # KG 加载完毕后要执行的回调

        self._build_ui()

    # ═══════════════════════════════════════════
    # UI 构建
    # ═══════════════════════════════════════════

    def _build_ui(self):
        """构建界面布局"""
        # 工具栏（独立模式显示完整搜索栏，嵌入模式显示简洁搜索栏）
        if self.standalone:
            self._build_search_toolbar()
            self._build_edge_filter_bar()
            self._build_filter_bar()
        else:
            self._build_compact_toolbar()

        # 图例（嵌入模式不显示图例，节省空间）
        if self.standalone:
            legend_frame = tk.Frame(self, bg="#fcfcfc", padx=10, pady=2)
            legend_frame.pack(fill="x")
            self._build_legend(legend_frame)

        # ── 主内容区 ──
        if self.standalone:
            # 左右布局：左侧画布 + 右侧知识面板
            main_frame = tk.Frame(self, bg="#ffffff")
            main_frame.pack(fill="both", expand=True)

            left_frame = tk.Frame(main_frame, bg="#ffffff")
            left_frame.pack(side="left", fill="both", expand=True)

            # 右侧知识面板
            right_frame = tk.Frame(main_frame, bg="#fcfcfc", width=300,
                                    highlightthickness=1, highlightbackground="#ddd")
            right_frame.pack(side="right", fill="y")
            right_frame.pack_propagate(False)

            # 知识面板标题
            tk.Label(right_frame, text="📖 知识详情",
                     bg="#fcfcfc", fg="#256a4b",
                     font=("Microsoft YaHei UI", 12, "bold"),
                     anchor="w", padx=12, pady=10).pack(fill="x")
            tk.Frame(right_frame, bg="#ddd", height=1).pack(fill="x")

            self.detail_text = ScrolledText(
                right_frame, wrap="word",
                font=("Microsoft YaHei UI", 10),
                relief="flat", bd=0,
                bg="#fcfcfc", fg="#333",
                padx=12, pady=10,
                highlightthickness=0,
            )
            self.detail_text.pack(fill="both", expand=True)
            self.detail_text.insert("1.0", "💡 点击节点查看详情\n\n双击展开邻域 | 右键更多操作")
            self.detail_text.config(state="disabled")

            canvas_parent = left_frame
        else:
            canvas_parent = self

        # ── 快速 Tkinter Canvas 渲染 ──
        self._tk_canvas = tk.Canvas(
            canvas_parent, bg="#fcfcfc", highlightthickness=0, cursor="arrow"
        )
        self._tk_canvas.pack(fill="both", expand=True)

        # 交互事件
        self._tk_canvas.bind("<ButtonPress-1>", self._cv_on_press)
        self._tk_canvas.bind("<B1-Motion>", self._cv_on_drag)
        self._tk_canvas.bind("<ButtonRelease-1>", self._cv_on_release)
        self._tk_canvas.bind("<MouseWheel>", self._cv_on_wheel)
        self._tk_canvas.bind("<Motion>", self._cv_on_motion)
        self._tk_canvas.bind("<Button-3>", self._cv_on_right_click)

        # 右键菜单
        self._context_menu = tk.Menu(self, tearoff=0)
        self._context_menu.add_command(label="🔍 查看详情", command=self._ctx_show_detail)
        self._context_menu.add_command(label="⬆ 展开邻域", command=self._ctx_expand)
        self._context_menu.add_command(label="🎯 设为中心", command=self._ctx_set_center)
        self._context_menu.add_separator()
        self._context_menu.add_command(label="📋 复制名称", command=self._ctx_copy_name)

        if not self.standalone:
            # 嵌入模式：底部节点详情（用 ScrolledText 支持滚动）
            detail_frame = tk.Frame(self, bg="#fcfcfc", padx=10, pady=6,
                                    highlightthickness=1, highlightbackground="#ddd")
            detail_frame.pack(fill="x")
            
            # 标题行
            detail_header = tk.Frame(detail_frame, bg="#fcfcfc")
            detail_header.pack(fill="x")
            tk.Label(detail_header, text="📖 节点详情",
                     bg="#fcfcfc", fg="#256a4b",
                     font=("Microsoft YaHei UI", 10, "bold")).pack(side="left")
            
            self.detail_text = ScrolledText(detail_frame, height=8, wrap="word",
                                        font=("Microsoft YaHei UI", 9),
                                        relief="flat", bd=0,
                                        bg="#fcfcfc", fg="#555",
                                        highlightthickness=0)
            self.detail_text.pack(fill="x", pady=(4, 0))
            self.detail_text.insert("1.0", "💡 点击节点查看详情 | 双击展开邻域 | 右键更多操作")
            self.detail_text.config(state="disabled")

    def _build_search_toolbar(self):
        """独立模式：完整搜索工具栏"""
        toolbar = tk.Frame(self, bg="#f0f0f0", padx=10, pady=6)
        toolbar.pack(fill="x")
        
        # 第一行：搜索
        row1 = tk.Frame(toolbar, bg="#f0f0f0")
        row1.pack(fill="x")
        tk.Label(row1, text="搜索:", bg="#f0f0f0",
                 font=("Microsoft YaHei UI", 10)).pack(side="left")
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(row1, textvariable=self.search_var,
                                     width=30, font=("Microsoft YaHei UI", 10))
        self.search_entry.pack(side="left", padx=(6, 6))
        self.search_entry.bind("<Return>", lambda e: self._do_search())
        ttk.Button(row1, text="搜索", command=self._do_search).pack(side="left", padx=(0, 4))
        ttk.Button(row1, text="重置", command=self._reset_view).pack(side="left", padx=(0, 4))
        ttk.Button(row1, text="←后退", command=self._go_back).pack(side="left", padx=(4, 4))
        
        # 布局切换
        tk.Label(row1, text="布局:", bg="#f0f0f0",
                 font=("Microsoft YaHei UI", 9)).pack(side="left", padx=(12, 2))
        self._layout_var = tk.StringVar(value="spring")
        layout_menu = ttk.Combobox(row1, textvariable=self._layout_var,
                                    values=["spring", "circular", "kamada_kawai", "shell"],
                                    width=10, state="readonly", font=("Microsoft YaHei UI", 9))
        layout_menu.pack(side="left")
        # 仅通过"应用布局"按钮触发重绘
        ttk.Button(row1, text="应用布局", command=self._redraw_current, width=8).pack(side="left", padx=(4, 0))
        
        # 导出
        ttk.Button(row1, text="📷 导出PNG", command=self._export_png, width=10).pack(side="right", padx=(0, 4))

    def _build_edge_filter_bar(self):
        """边类型过滤栏（单选模式）"""
        bar = tk.Frame(self, bg="#f5f5f5", padx=10, pady=3)
        bar.pack(fill="x")
        tk.Label(bar, text="显示:", bg="#f5f5f5",
                 font=("Microsoft YaHei UI", 9)).pack(side="left")

        self._edge_mode = tk.StringVar(value="all")
        edge_modes = [
            ("all", "全部关系"),
            ("syndrome", "仅证型关联"),
            ("herb", "仅药材关联"),
            ("formula", "仅方剂关联"),
            ("incompatible", "仅禁忌"),
        ]
        for mode_val, mode_label in edge_modes:
            rb = tk.Radiobutton(bar, text=mode_label, variable=self._edge_mode,
                                value=mode_val, bg="#f5f5f5",
                                font=("Microsoft YaHei UI", 8))
            rb.pack(side="left", padx=2)

        # 初始化 edge_vars（供 _show_subgraph_given_nodes 使用）
        self._edge_vars = {
            "belongs_to": tk.BooleanVar(value=True),
            "recommends_herb": tk.BooleanVar(value=True),
            "recommends_formula": tk.BooleanVar(value=True),
            "composes": tk.BooleanVar(value=True),
            "associated_with_herb": tk.BooleanVar(value=True),
            "incompatible_with": tk.BooleanVar(value=True),
            "co_occurs_with": tk.BooleanVar(value=True),
        }

    def _apply_edge_mode(self):
        """根据 _edge_mode 设置 _edge_vars"""
        mode = self._edge_mode.get()
        # 先全部关闭
        for v in self._edge_vars.values():
            v.set(False)
        if mode == "all":
            for v in self._edge_vars.values():
                v.set(True)
        elif mode == "syndrome":
            self._edge_vars["belongs_to"].set(True)
            self._edge_vars["recommends_herb"].set(True)
            self._edge_vars["recommends_formula"].set(True)
        elif mode == "herb":
            self._edge_vars["associated_with_herb"].set(True)
            self._edge_vars["composes"].set(True)
            self._edge_vars["co_occurs_with"].set(True)
        elif mode == "formula":
            self._edge_vars["recommends_formula"].set(True)
            self._edge_vars["composes"].set(True)
            self._edge_vars["associated_with_herb"].set(True)
        elif mode == "incompatible":
            self._edge_vars["incompatible_with"].set(True)

    def _build_compact_toolbar(self):
        """嵌入模式：紧凑搜索栏"""
        toolbar = tk.Frame(self, bg="#f5f5f5", padx=6, pady=4)
        toolbar.pack(fill="x")
        tk.Label(toolbar, text="🔍", bg="#f5f5f5",
                 font=("Microsoft YaHei UI", 10)).pack(side="left")
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(toolbar, textvariable=self.search_var,
                                     width=20, font=("Microsoft YaHei UI", 9))
        self.search_entry.pack(side="left", padx=(4, 4))
        self.search_entry.bind("<Return>", lambda e: self._do_search())
        ttk.Button(toolbar, text="搜索", command=self._do_search, width=4).pack(side="left", padx=(0, 4))
        ttk.Button(toolbar, text="重置", command=self._reset_view, width=4).pack(side="left")

    def _build_filter_bar(self):
        """节点类型过滤（单选模式）"""
        filter_frame = tk.Frame(self, bg="#f0f0f0", padx=10, pady=4)
        filter_frame.pack(fill="x")
        tk.Label(filter_frame, text="节点:", bg="#f0f0f0",
                 font=("Microsoft YaHei UI", 9)).pack(side="left")

        self._node_mode = tk.StringVar(value="all")
        node_modes = [
            ("all", "全部"),
            ("syndrome", "仅证型"),
            ("herb_medicine", "仅药材"),
            ("formula", "仅方剂"),
            ("symptom_only", "仅症状"),
        ]
        for mode_val, mode_label in node_modes:
            rb = tk.Radiobutton(filter_frame, text=mode_label, variable=self._node_mode,
                                value=mode_val, bg="#f0f0f0",
                                font=("Microsoft YaHei UI", 9))
            rb.pack(side="left", padx=(4, 0))

        self.type_vars = {
            "Symptom": tk.BooleanVar(value=True),
            "Syndrome": tk.BooleanVar(value=True),
            "Herb": tk.BooleanVar(value=True),
            "Formula": tk.BooleanVar(value=True),
        }

    def _apply_node_mode(self):
        """根据 _node_mode 设置 type_vars"""
        mode = self._node_mode.get()
        for v in self.type_vars.values():
            v.set(False)
        if mode == "all":
            for v in self.type_vars.values():
                v.set(True)
        elif mode == "syndrome":
            self.type_vars["Syndrome"].set(True)
        elif mode == "herb_medicine":
            self.type_vars["Herb"].set(True)
        elif mode == "formula":
            self.type_vars["Formula"].set(True)
        elif mode == "symptom_only":
            self.type_vars["Symptom"].set(True)
            self.type_vars["Syndrome"].set(True)

    def _build_legend(self, parent):
        """图例"""
        for tname, color in NODE_COLORS.items():
            if tname in {"Category", "Meridian", "Nature", "Flavor"}:
                continue  # 辅助节点隐藏在图例中
            dot = tk.Canvas(parent, width=12, height=12, bg=parent["bg"],
                            highlightthickness=0)
            dot.create_oval(1, 1, 11, 11, fill=color, outline="")
            dot.pack(side="left", padx=(8, 2))
            tk.Label(parent, text=tname, bg=parent["bg"],
                     font=("Microsoft YaHei UI", 8)).pack(side="left")

    # ═══════════════════════════════════════════
    # 图谱加载与绘制
    # ═══════════════════════════════════════════

    def _load_graph(self):
        """同步加载知识图谱（可能阻塞，尽量用异步版本）"""
        if self.graph is not None:
            return
        g, q = get_kg()
        self.graph = g
        self.query = q

    def _load_graph_async(self, on_done=None):
        """异步加载知识图谱，GUI 不阻塞"""
        if self.graph is not None:
            if on_done:
                on_done()
            return
        if self._loading:
            # 已经在加载中，缓存回调
            if on_done:
                self._pending_callback = on_done
            return

        self._loading = True
        self._pending_callback = on_done
        self._show_loading("正在加载知识图谱...")

        def _worker():
            try:
                g, q = get_kg()
                # 回到主线程更新
                self.after(0, self._on_graph_loaded, g, q)
            except Exception as exc:
                logger.exception("知识图谱异步加载失败")
                self.after(0, self._on_graph_load_error, str(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_graph_loaded(self, g, q):
        """KG 加载完毕后的回调（主线程执行）"""
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
        """KG 加载失败的回调（主线程执行）"""
        self._loading = False
        self._pending_callback = None
        self._show_loading(f"⚠ 知识图谱加载失败: {err_msg}")
        logger.error("知识图谱加载失败: %s", err_msg)

    def _show_loading(self, text="正在加载..."):
        """显示加载提示"""
        if self._tk_canvas:
            self._tk_canvas.delete("all")
            cw = max(self._tk_canvas.winfo_width(), 100)
            ch = max(self._tk_canvas.winfo_height(), 100)
            self._tk_canvas.create_text(cw/2, ch/2, text=text,
                font=("Microsoft YaHei UI", 14), fill="#999", justify="center")
        else:
            self.ax.clear()
            self.ax.text(0.5, 0.5, text,
                         transform=self.ax.transAxes, ha="center", va="center",
                         fontsize=14, color="#999")
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.ax.axis("off")

    def _show_overview_graph(self):
        """启动时展示图谱概览（仅显示证型+部分症状，轻量快速）"""
        # 如果图还没加载，异步加载后再执行
        if self.graph is None:
            self._load_graph_async(on_done=self._show_overview_graph)
            return
        self._render_overview()

    def _render_overview(self):
        """渲染概览图：以方剂为核心，展示方剂→药材→证型关系"""
        g = self.graph

        # 取度数最高的前 15 个方剂（按组成药材数排序）
        formula_nodes = [
            n for n in g.nodes
            if g.nodes[n].get("type") == "Formula"
        ]
        formula_nodes.sort(key=lambda n: g.degree(n), reverse=True)
        formula_nodes = formula_nodes[:15]
        if not formula_nodes:
            self._show_placeholder()
            return

        all_nodes = set(formula_nodes)

        # 收集每个方剂的组成药材（composes 是药材→方剂，所以用入边）
        for fn in formula_nodes:
            for pred in g.predecessors(fn):
                if g.nodes[pred].get("type") == "Herb":
                    if any(d.get("type") == "composes" for _, v, d in g.edges(pred, data=True) if v == fn):
                        all_nodes.add(pred)

        # 收集推荐这些方剂的证型
        for fn in formula_nodes:
            for pred in g.predecessors(fn):
                if g.nodes[pred].get("type") == "Syndrome":
                    if any(d.get("type") == "recommends_formula" for _, v, d in g.edges(pred, data=True) if v == fn):
                        all_nodes.add(pred)

        all_nodes = list(all_nodes)
        if len(all_nodes) < 3:
            self._show_placeholder()
            return

        # 以第一个方剂为中心
        center = formula_nodes[0]
        self._show_subgraph_given_nodes(center, all_nodes)

    def _show_placeholder(self):
        """显示初始提示"""
        if self._tk_canvas:
            self._tk_canvas.delete("all")
            cw = self._tk_canvas.winfo_width() or 600
            ch = self._tk_canvas.winfo_height() or 400
            self._tk_canvas.create_text(cw/2, ch/2-20,
                text="正在加载知识图谱...\n请稍候",
                font=("Microsoft YaHei UI", 16), fill="#999", justify="center")
        else:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "正在加载知识图谱...\n\n请稍候",
                         transform=self.ax.transAxes, ha="center", va="center",
                         fontsize=14, color="#999")
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.ax.axis("off")
            self.canvas.draw_idle()

    def _do_search(self):
        """执行图谱搜索"""
        query = self.search_var.get().strip()
        if not query:
            self._reset_view()
            return

        if self.graph is None:
            self._load_graph_async(
                on_done=lambda: self._do_search() if self.search_var.get().strip() else None
            )
            return

        # 查找匹配节点
        g = self.graph
        matches = [n for n in g.nodes if query in g.nodes[n].get("name", "")]
        if not matches:
            self._set_detail(f"未找到包含「{query}」的节点")
            return

        # 取第一个匹配节点，显示其1-hop子图
        center = matches[0]
        self._show_subgraph(center, matches[:10])

    def _show_subgraph(self, center_node: str, extra_nodes: Optional[List[str]] = None):
        """显示以 center_node 为中心的1跳子图（使用新绘制引擎）"""
        g = self.graph
        if g is None:
            self._load_graph()
            g = self.graph

        # 收集1跳邻居
        neighbors = set()
        for _, nbr in g.edges(center_node):
            neighbors.add(nbr)
        for nbr, _ in g.edges(center_node):
            neighbors.add(nbr)
        if extra_nodes:
            for n in extra_nodes:
                neighbors.add(n)
        neighbors.add(center_node)

        self._show_subgraph_given_nodes(center_node, list(neighbors))

    def _on_pick(self, event):
        """节点点击事件：聚焦到该节点（显示1跳邻域 + 自动缩放 + 更新详情）"""
        node_id = self._get_picked_node(event)
        if node_id:
            self._selected_node = node_id
            # 保存当前视图到展开历史（用于后退）
            if self._current_subgraph is not None:
                self._expand_history.append(set(self._current_subgraph.nodes))
            # 以该节点为中心显示1跳子图
            self._show_subgraph(node_id)
            # 自动缩放聚焦到节点邻域
            self._auto_zoom_to_fit()

    def _on_click(self, event):
        """鼠标事件：双击展开 + 右键菜单"""
        if event.inaxes != self.ax:
            return
        # 右键菜单
        if event.button == 3:
            node_id = self._hit_test_node(event.xdata, event.ydata)
            if node_id:
                self._selected_node = node_id
                self._context_menu.tk_popup(event.guiEvent.x_root if hasattr(event, 'guiEvent') else 
                                            self.winfo_pointerx(),
                                            event.guiEvent.y_root if hasattr(event, 'guiEvent') else
                                            self.winfo_pointery())
            return
        # 双击展开
        if event.button == 1 and event.dblclick:
            node_id = self._hit_test_node(event.xdata, event.ydata)
            if node_id:
                self._expand_node(node_id)

    def _on_mouse_wheel(self, event):
        """鼠标滚轮缩放，以鼠标位置为中心，节点/字体同步缩放"""
        if event.inaxes != self.ax:
            return
        # 缩放系数：上滚放大，下滚缩小
        scale = 0.85 if event.step > 0 else 1.15
        x_lo, x_hi = self.ax.get_xlim()
        y_lo, y_hi = self.ax.get_ylim()
        cx, cy = event.xdata, event.ydata  # 鼠标在数据坐标中的位置
        # 以鼠标位置为中心缩放
        self.ax.set_xlim(cx + (x_lo - cx) * scale, cx + (x_hi - cx) * scale)
        self.ax.set_ylim(cy + (y_lo - cy) * scale, cy + (y_hi - cy) * scale)

        # 增量更新 + 防抖：50ms 内多次滚轮只触发一次重绘
        if hasattr(self, '_zoom_timer'):
            try:
                self.winfo_toplevel().after_cancel(self._zoom_timer)
            except Exception:
                pass
        self._zoom_timer = self.winfo_toplevel().after(50, self._zoom_update)

    def _on_motion(self, event):
        """鼠标移动：悬停节点时改变光标形状"""
        if event.inaxes != self.ax:
            if self._motion_node_id is not None:
                self._motion_node_id = None
                self.config(cursor="")
            return
        node = self._hit_test_node(event.xdata, event.ydata)
        if node is not None and node != self._motion_node_id:
            self._motion_node_id = node
            self.config(cursor="hand2")
        elif node is None and self._motion_node_id is not None:
            self._motion_node_id = None
            self.config(cursor="")

    def _hit_test_node(self, x, y):
        """命中测试：给定坐标返回节点ID"""
        if self._current_subgraph is None:
            return None
        sub = self._current_subgraph
        sub_key = frozenset(sub.nodes)
        pos = self._pos_cache.get(sub_key)
        if pos is None:
            return None
        # 找最近节点（阈值30像素）
        import math
        best_dist = 30.0 / self.fig.dpi  # 屏幕像素转数据坐标
        best_node = None
        for n, (nx, ny) in pos.items():
            dist = math.hypot(x - nx, y - ny)
            if dist < best_dist:
                best_dist = dist
                best_node = n
        return best_node

    def _get_picked_node(self, event):
        """从pick event获取节点ID（支持 artist→node 映射和旧式索引两种方式）"""
        if not hasattr(event, "artist"):
            return None
        aid = id(event.artist)
        if aid in self._artist_to_node:
            return self._artist_to_node[aid]
        # 旧式索引回退（分组绘制时用）
        if not hasattr(event.artist, "get_offsets"):
            return None
        if event.ind is None or len(event.ind) == 0:
            return None
        idx = event.ind[0]
        if self._current_subgraph is not None:
            nodes = list(self._current_subgraph.nodes)
            if idx < len(nodes):
                return nodes[idx]
        return None

    def _expand_node(self, node_id: str):
        """双击展开：将节点的1跳邻居加入子图"""
        if self.graph is None or self._current_subgraph is None:
            return
        g = self.graph
        current = set(self._current_subgraph.nodes)
        if node_id not in current:
            return

        # 保存历史
        self._expand_history.append(set(current))

        # 添加新邻居
        new_nodes = set(current)
        for _, nbr in g.edges(node_id):
            new_nodes.add(nbr)
        for nbr, _ in g.edges(node_id):
            new_nodes.add(nbr)

        if len(new_nodes) == len(current):
            return  # 没有新节点

        self._show_subgraph_given_nodes(node_id, list(new_nodes))

    def _go_back(self):
        """后退到上一个展开状态"""
        logger.info("_go_back: 历史深度 %d", len(self._expand_history))
        if not self._expand_history:
            logger.info("_go_back: 历史为空，无法后退")
            return
        prev = self._expand_history.pop()
        logger.info("_go_back: 取出历史状态，剩余 %d，节点数 %d",
                     len(self._expand_history), len(prev))
        if self._current_subgraph is not None:
            current_nodes = set(self._current_subgraph.nodes)
            # 找上一个状态中不在当前子图中的节点作为中心
            center = None
            for n in prev:
                ntype = self.graph.nodes[n].get("type", "")
                if ntype in ("Syndrome", "Symptom"):
                    center = n
                    break
            if center is None:
                center = next(iter(prev))
            logger.info("_go_back: 以 %s 为中心重绘", center)
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
            name = self.graph.nodes[self._selected_node].get("name", self._selected_node)
            self.clipboard_clear()
            self.clipboard_append(name)

    # ── 布局切换 ──

    def _redraw_current(self):
        """使用当前设置重绘子图"""
        if self._current_subgraph is None:
            logger.info("_redraw_current: 无当前子图，跳过")
            return
        logger.info("_redraw_current: 清除布局缓存，重绘 %s 个节点",
                     self._current_subgraph.number_of_nodes())
        # 应用模式设置
        self._apply_edge_mode()
        self._apply_node_mode()
        # 清除布局缓存强制重新计算
        sub_key = frozenset(self._current_subgraph.nodes)
        if sub_key in self._pos_cache:
            del self._pos_cache[sub_key]
        center = self._selected_node or list(self._current_subgraph.nodes)[0]
        self._show_subgraph_given_nodes(center, list(self._current_subgraph.nodes))

    def _get_layout(self, sub, pos=None):
        """根据当前布局选择计算布局"""
        import networkx as nx
        layout_name = getattr(self, '_layout_var', None)
        layout_type = layout_name.get() if layout_name else "spring"
        
        if layout_type == "spring":
            return nx.spring_layout(sub, k=0.8, iterations=50, seed=42, pos=pos)
        elif layout_type == "circular":
            return nx.circular_layout(sub)
        elif layout_type == "kamada_kawai":
            return nx.kamada_kawai_layout(sub)
        elif layout_type == "shell":
            # 按类型分层
            if self.graph is None:
                return nx.shell_layout(sub)
            groups = []
            for t in ["Syndrome", "Symptom", "Herb", "Formula"]:
                nodes = [n for n in sub.nodes if self.graph.nodes[n].get("type") == t]
                if nodes:
                    groups.append(nodes)
            return nx.shell_layout(sub, nlist=groups) if groups else nx.shell_layout(sub)
        return nx.spring_layout(sub, k=0.8, iterations=50, seed=42)

    # ── 导出PNG ──

    def _export_png(self):
        """导出当前图谱为PNG"""
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png")],
            title="导出知识图谱"
        )
        if path:
            self._tk_canvas.postscript(file=path.replace(".png", ".eps"))
            tk.messagebox.showinfo("导出", f"已保存到:\n{path.replace('.png', '.eps')}")

    # ── 子图绘制（改进版） ──

    def _show_subgraph_given_nodes(self, center: str, node_list: list):
        """给定节点列表绘制子图"""
        g = self.graph
        if g is None:
            self._load_graph()
            g = self.graph

        # 按类型过滤
        active_types = {t for t, v in getattr(self, 'type_vars', {}).items() if v.get()}
        if active_types:
            filtered = [n for n in node_list if g.nodes[n].get("type", "") in active_types]
            if center not in filtered:
                filtered.append(center)
            node_list = filtered

        sub = g.subgraph(node_list).copy()
        sub_key = frozenset(sub.nodes)

        if sub_key in self._pos_cache:
            pos = self._pos_cache[sub_key]
        else:
            pos = self._get_layout(sub)
            self._pos_cache[sub_key] = pos

        self._current_subgraph = sub
        self._all_node_ids = list(sub.nodes)

        # 使用快速 Canvas 渲染（如果有 tk_canvas）
        if self._tk_canvas is not None:
            logger.info("渲染子图: %s 节点, %s 边", sub.number_of_nodes(), sub.number_of_edges())
            try:
                self._draw_canvas(sub, pos, center)
            except Exception as exc:
                logger.exception("Canvas 渲染失败: %s", exc)
                self._set_detail(f"❌ 渲染失败: {exc}")
        else:
            self._draw_subgraph(sub, pos, center)
        self._current_zoom = 1.0
        self._show_node_detail(center)

    # ── 快速 Canvas 渲染（Tkinter Canvas，硬件加速，丝滑缩放平移） ──

    def _draw_canvas(self, sub, pos, center_node=None):
        """用 Tkinter Canvas 绘制图谱（替代 matplotlib）"""
        logger.info("_draw_canvas: 开始渲染 %s 个节点", sub.number_of_nodes())
        c = self._tk_canvas
        c.delete("all")
        self._canvas_items = {}
        self._edge_items = []
        g = self.graph

        # 计算数据→画布映射
        xs = [pos[n][0] for n in sub.nodes]
        ys = [pos[n][1] for n in sub.nodes]
        if not xs:
            logger.warning("_draw_canvas: 无节点可渲染")
            return
        pad_x = max((max(xs) - min(xs)) * 0.3, 1)
        pad_y = max((max(ys) - min(ys)) * 0.3, 1)
        x_min, x_max = min(xs) - pad_x, max(xs) + pad_x
        y_min, y_max = min(ys) - pad_y, max(ys) + pad_y
        # 确保画布已布局再取尺寸，否则默认用 800×600
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

        # 保存变换参数供 zoom/pan 使用
        self._cv_s = s
        self._cv_cx, self._cv_cy = cx, cy
        self._cv_mx, self._cv_my = mx, my
        self._cv_pos = pos  # 原始数据坐标
        self._cv_zoom = 1.0

        # 画边
        for u, v, data in sub.edges(data=True):
            etype = data.get("type", "")
            ecolor = EDGE_COLORS.get(etype, "#ccc")
            x1, y1 = to_canvas(pos[u][0], pos[u][1])
            x2, y2 = to_canvas(pos[v][0], pos[v][1])
            lid = c.create_line(x1, y1, x2, y2, fill=ecolor, width=1, tags=("edge",))
            self._edge_items.append(lid)

        # 画节点
        NODE_COLORS_HEX = {
            "Symptom": "#e74c3c", "Syndrome": "#9b59b6", "Herb": "#27ae60",
            "Formula": "#2980b9", "PatentMedicine": "#f39c12", "FolkRemedy": "#95a5a6",
            "Category": "#1abc9c", "Meridian": "#e67e22", "Nature": "#e91e63", "Flavor": "#00bcd4",
        }
        NODE_RADIUS = {"Syndrome": 20, "Formula": 17, "Symptom": 14, "Herb": 12}
        for n in sub.nodes:
            ntype = g.nodes[n].get("type", "Unknown")
            name = g.nodes[n].get("name", "")
            x, y = to_canvas(pos[n][0], pos[n][1])
            r = NODE_RADIUS.get(ntype, 10)
            color = NODE_COLORS_HEX.get(ntype, "#bdc3c7")

            # 阴影（偏移 1px）
            c.create_oval(x - r + 1, y - r + 1, x + r + 1, y + r + 1,
                          fill="#a0a0a0", outline="", tags=("shadow",))

            # 底色圆
            c.create_oval(x - r, y - r, x + r, y + r,
                          fill=color, outline="#ffffff", width=2,
                          tags=(f"node_{n}", "node"))

            # 标签背景
            label = name if len(name) <= 6 else name[:5] + "…"
            fs = max(7, min(13, int(11 * r / 12)))
            tw = len(label) * fs * 0.55
            th = fs + 4
            lx, ly = x, y + r + th / 2 + 1
            c.create_rectangle(lx - tw / 2 - 3, ly - th / 2,
                               lx + tw / 2 + 3, ly + th / 2,
                               fill="#f5f5f5", outline="#e0e0e0", width=1,
                               tags=(f"label_{n}", "label_bg"))
            # 标签文字
            c.create_text(lx, ly, text=label,
                          font=("Microsoft YaHei UI", fs),
                          fill="#333", anchor="center",
                          tags=(f"label_{n}", "label"))
            self._canvas_items[n] = {"oval": n, "text": n, "r": r, "color": color}

        # 高亮中心
        if center_node and center_node in self._canvas_items:
            ci = self._canvas_items[center_node]
            c.itemconfig(ci["oval"], width=3, outline="#2c3e50")

        # 保存 node_id → tag 映射（用于命中测试）
        self._cv_node_tags = {f"node_{n}": n for n in sub.nodes}

        # 更新详情面板
        self._set_detail(f"📊 共 {sub.number_of_nodes()} 节点 / {sub.number_of_edges()} 条边\n"
                         f"💡 拖拽平移 · 滚轮缩放 · 点击节点查看详情")

    def _cv_get_node(self, x, y):
        """命中测试：Canvas 坐标 → 节点 ID"""
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
        if node:
            logger.info("_cv_on_press: 点击节点 %s", node)
            self._selected_node = node
            # 保存当前视图到展开历史（用于后退）
            if self._current_subgraph is not None:
                self._expand_history.append(set(self._current_subgraph.nodes))
                logger.info("_cv_on_press: 保存历史，当前深度 %d", len(self._expand_history))
            # 显示1跳子图
            self._show_subgraph(node)

    def _cv_on_drag(self, e):
        if self._cv_drag_start:
            dx = e.x - self._cv_drag_start[0]
            dy = e.y - self._cv_drag_start[1]
            self._tk_canvas.move("all", dx, dy)
            self._cv_drag_start = (e.x, e.y)

    def _cv_on_release(self, e):
        self._cv_drag_start = None

    def _cv_on_wheel(self, e):
        """滚轮缩放（基于 Canvas.scale，单操作 O(1)）"""
        factor = 1.1 if e.delta > 0 else 0.9
        self._tk_canvas.scale("all", e.x, e.y, factor, factor)
        # 同步调整标签字体（scale 不会缩放字体）
        for n, ci in self._canvas_items.items():
            old_fs = self._tk_canvas.itemcget(ci["text"], "font")
            try:
                sz = int(old_fs.split()[-1])
                new_sz = max(6, min(18, int(sz * factor)))
                self._tk_canvas.itemconfig(ci["text"],
                    font=("Microsoft YaHei UI", new_sz))
            except Exception:
                pass

    def _cv_on_motion(self, e):
        """悬停高亮：放大节点"""
        node = self._cv_get_node(e.x, e.y)
        if node and node != getattr(self, '_cv_hover', None):
            self._cv_hover = node
            self._tk_canvas.config(cursor="hand2")
        elif not node and getattr(self, '_cv_hover', None):
            self._cv_hover = None
            self._tk_canvas.config(cursor="")

    def _cv_on_right_click(self, e):
        node = self._cv_get_node(e.x, e.y)
        if node:
            self._selected_node = node
            self._show_node_detail(node)
            self._context_menu.tk_popup(e.x_root, e.y_root)

    def _draw_subgraph(self, sub, pos, center_node=None, zoom_factor=1.0):
        """绘制子图（包含边过滤）

        Args:
            zoom_factor: 当前缩放比例（基准为1.0），用于同步缩放节点和字体。
        """
        g = self.graph
        self._cached_scatters = []
        self._cached_texts = []
        self._cached_edges = []
        self.ax.clear()
        self.ax.set_facecolor("#fcfcfc")

        # 边过滤
        edge_vars = getattr(self, '_edge_vars', {})
        active_edges = {t for t, v in edge_vars.items() if v.get()}

        # 画边（边的宽度也随缩放微调）
        edge_groups = {}
        for u, v, data in sub.edges(data=True):
            etype = data.get("type", "")
            if active_edges and etype not in active_edges:
                continue
            edge_groups.setdefault(etype, []).append((u, v))

        edge_width = max(0.5, 1.0 * zoom_factor ** 0.5)
        for etype, edges in edge_groups.items():
            ecolor = EDGE_COLORS.get(etype, "#ccc")
            nx.draw_networkx_edges(
                sub, pos, ax=self.ax,
                edgelist=edges, edge_color=ecolor,
                alpha=0.3, width=edge_width, style="-",
            )

        # 按类型分组画节点
        type_groups = {}
        for n in sub.nodes:
            ntype = g.nodes[n].get("type", "Unknown")
            type_groups.setdefault(ntype, []).append(n)

        for ntype, nodes in type_groups.items():
            color = NODE_COLORS.get(ntype, "#bdc3c7")
            base_size = NODE_SIZE_MAP.get(ntype, 80)
            size = max(20, base_size * zoom_factor)
            nx.draw_networkx_nodes(
                sub, pos, ax=self.ax,
                nodelist=nodes, node_color=color,
                node_size=size, alpha=0.9, edgecolors="#fff",
                linewidths=min(3.0, 1.2 * zoom_factor ** 0.5),
            )

        # 标签（截断长名，使用中文字体，大小随缩放同步）
        labels = {}
        for n in sub.nodes:
            name = g.nodes[n].get("name", "")
            labels[n] = name if len(name) <= 8 else name[:7] + "…"
        fontsize = max(5, min(20, 9 * zoom_factor))
        for n, (x, y) in pos.items():
            label = labels.get(n, "")
            if not label:
                continue
            kwargs = dict(fontsize=fontsize, color="#333",
                          ha="center", va="center")
            if _CN_FONT:
                kwargs["fontproperties"] = _CN_FONT
            self.ax.text(x, y, label, **kwargs)

        # 高亮中心
        if center_node and center_node in sub.nodes:
            ctype = g.nodes[center_node].get("type", "")
            c_base = NODE_SIZE_MAP.get(ctype, 80)
            c_size = max(30, (c_base + 40) * zoom_factor)
            nx.draw_networkx_nodes(
                sub, pos, ax=self.ax,
                nodelist=[center_node],
                node_color=NODE_COLORS.get(ctype, "#ccc"),
                node_size=c_size,
                alpha=1.0, edgecolors="#2c3e50", linewidths=min(4, 3 * zoom_factor ** 0.5),
            )

        self.ax.axis("off")
        name = g.nodes[center_node].get("name", "") if center_node else ""
        n_nodes = sub.number_of_nodes()
        n_edges = sub.number_of_edges()
        title_font = max(8, min(18, 11 * zoom_factor ** 0.6))
        self.ax.set_title(f"「{name}」的关联网络 ({n_nodes}节点, {n_edges}边)",
                          fontsize=title_font, color="#333", pad=12)
        self.canvas.draw_idle()
        # 保存 blit 背景（供后续增量缩放使用）
        try:
            self.canvas.blit(self.ax.bbox)
        except Exception:
            pass

    # ── 缩放增量更新（防抖 + blit） ──
    def _zoom_update(self):
        """缩放增量更新：调整已有节点/字体大小，不全量重绘"""
        if self._current_subgraph is None:
            self.canvas.draw_idle()
            return

        x_lo, x_hi = self.ax.get_xlim()
        y_lo, y_hi = self.ax.get_ylim()
        current_x_range = x_hi - x_lo

        zoom_factor = 1.0
        if self._base_x_range and current_x_range > 0:
            zoom_factor = self._base_x_range / current_x_range
        self._current_zoom = zoom_factor

        # 更新已有 scatter 节点大小
        for coll in self.ax.collections:
            if hasattr(coll, 'get_sizes') and hasattr(coll, 'set_sizes'):
                old_sizes = coll.get_sizes()
                if len(old_sizes) > 0:
                    # 从基准大小按比例缩放，保持相对尺寸
                    base = old_sizes[0] / (self._current_zoom or 1)
                    new_sizes = max(20, base * zoom_factor)
                    coll.set_sizes([new_sizes] * len(old_sizes))

        # 更新已有文本大小
        for txt in self.ax.texts:
            try:
                old_size = txt.get_fontsize()
                new_size = max(5, min(20, old_size / (self._current_zoom / max(zoom_factor, 0.1))))
                txt.set_fontsize(new_size)
            except Exception:
                pass

        # 更新边宽
        for line in self.ax.lines:
            try:
                old_w = line.get_linewidth()
                new_w = max(0.5, old_w * (zoom_factor / max(self._current_zoom, 0.1)))
                line.set_linewidth(new_w)
            except Exception:
                pass

        # blit 渲染：只更新 axes 区域，比 draw_idle() 全画布重绘快得多
        try:
            self.canvas.blit(self.ax.bbox)
        except Exception:
            self.canvas.draw_idle()

    def _auto_zoom_to_fit(self):
        """自动缩放以适配当前子图的所有节点（聚焦视图 + 同步缩放节点/字体）"""
        if self._current_subgraph is None:
            return
        sub_key = frozenset(self._current_subgraph.nodes)
        pos = self._pos_cache.get(sub_key)
        if pos is None:
            return

        nodes = list(self._current_subgraph.nodes)
        xs = [pos[n][0] for n in nodes if n in pos]
        ys = [pos[n][1] for n in nodes if n in pos]
        if not xs or not ys:
            return

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        pad_x = max((x_max - x_min) * 0.4, 0.5)
        pad_y = max((y_max - y_min) * 0.4, 0.5)

        self.ax.set_xlim(x_min - pad_x, x_max + pad_x)
        self.ax.set_ylim(y_min - pad_y, y_max + pad_y)

        # 同步缩放节点/字体大小
        self._zoom_update()

    def _highlight_node(self, node_id: str):
        """高亮特定节点（支持缩放同步）"""
        if self._current_subgraph is None or node_id not in self._current_subgraph:
            return
        g = self.graph
        sub = self._current_subgraph
        sub_key = frozenset(sub.nodes)
        pos = self._pos_cache.get(sub_key)
        if pos is None:
            return
        z = self._current_zoom  # 当前缩放因子

        # 移除旧的节点绘制
        for artist in self.ax.collections[:]:
            try:
                artist.remove()
            except Exception:
                pass

        type_groups = {}
        for n in sub.nodes:
            ntype = g.nodes[n].get("type", "Unknown")
            type_groups.setdefault(ntype, []).append(n)

        for ntype, nodes in type_groups.items():
            color = NODE_COLORS.get(ntype, "#bdc3c7")
            base = NODE_SIZE_MAP.get(ntype, 80)
            size = max(20, base * z)
            ec, lw = ("#fff", min(3.0, 1.2 * z ** 0.5))
            if node_id in nodes:
                size = max(30, (base + 40) * z)
                ec, lw = "#2c3e50", min(4, 3 * z ** 0.5)
            nx.draw_networkx_nodes(
                sub, pos, ax=self.ax,
                nodelist=[n for n in nodes if n != node_id] if node_id in nodes else nodes,
                node_color=color, node_size=max(20, base * z),
                alpha=0.9, edgecolors="#fff", linewidths=min(3.0, 1.2 * z ** 0.5),
            )
            if node_id in nodes:
                nx.draw_networkx_nodes(
                    sub, pos, ax=self.ax,
                    nodelist=[node_id],
                    node_color=color, node_size=size,
                    alpha=1.0, edgecolors=ec, linewidths=lw,
                )

        self.canvas.draw_idle()

    def _show_node_detail(self, node_id: str):
        """增强详情面板"""
        g = self.graph
        data = g.nodes[node_id]
        name = data.get("name", node_id)
        ntype = data.get("type", "Unknown")

        lines = []
        # 标题行
        type_icon = {"Symptom": "🟡", "Syndrome": "🟣", "Herb": "🟢", 
                     "Formula": "🔵", "PatentMedicine": "🟠", "FolkRemedy": "⚪"}
        icon = type_icon.get(ntype, "⚪")
        lines.append(f"{icon} 【{ntype}】{name}")
        lines.append("─" * 40)

        # 根据类型显示信息
        if ntype == "Herb":
            cat = data.get("category", "")
            if cat: lines.append(f"  分类: {cat}")
            eff = data.get("efficacy", "")
            if eff: lines.append(f"  功效: {eff}")
            ind = data.get("indications", "")
            if ind: lines.append(f"  主治: {ind}")
            usage = data.get("usage", "")
            if usage: lines.append(f"  用法: {usage}")
            cont = data.get("contraindications", "")
            if cont: lines.append(f"  禁忌: {cont}")
            source = data.get("source", "")
            if source: lines.append(f"  来源: {source}")
            # 获取性味归经详情
            for _, nbr, ed in g.edges(node_id, data=True):
                et = ed.get("type", "")
                nb_name = g.nodes[nbr].get("name", "")
                if et == "has_nature":
                    lines.append(f"  性: {nb_name}")
                elif et == "has_flavor":
                    lines.append(f"  味: {nb_name}")
                elif et == "has_meridian":
                    lines.append(f"  归经: {nb_name}")
            if data.get("toxic"): lines.append("  ⚠️ 有毒药材")
            if data.get("pregnancy_forbidden"): lines.append("  ⚠️ 孕妇禁用")
            # 相反相畏
            incomp_set = set()
            for _, nbr, ed in g.edges(node_id, data=True):
                if ed.get("type") == "incompatible_with":
                    incomp_set.add(g.nodes[nbr].get("name", ""))
            if incomp_set:
                lines.append(f"  相反相畏: {'、'.join(sorted(incomp_set)[:6])}")
        elif ntype == "Formula":
            sb = data.get("source_book", "")
            if sb: lines.append(f"  来源: {sb}")
            comp = data.get("composition", "")
            if comp: lines.append(f"  组成: {comp[:80]}{'…' if len(comp)>80 else ''}")
            func = data.get("functions", "")
            if func: lines.append(f"  功能: {func}")
            ind = data.get("indications", "")
            if ind: lines.append(f"  主治: {ind}")
            usage = data.get("usage", "")
            if usage: lines.append(f"  用法: {usage}")
            dos = data.get("dosage", "")
            if dos: lines.append(f"  用量: {dos}")
        elif ntype == "Syndrome":
            desc = data.get("description", "")
            if desc: lines.append(f"  证候: {desc}")
            treat = data.get("treatment", "")
            if treat: lines.append(f"  治则: {treat}")
            # 推荐药材
            herbs = []
            for _, nbr, ed in g.edges(node_id, data=True):
                if ed.get("type") == "recommends_herb":
                    herbs.append(g.nodes[nbr].get("name", ""))
            if herbs:
                lines.append(f"  推荐药材: {'、'.join(herbs[:6])}")
            # 推荐方剂
            formulas = []
            for _, nbr, ed in g.edges(node_id, data=True):
                if ed.get("type") == "recommends_formula":
                    formulas.append(g.nodes[nbr].get("name", ""))
            if formulas:
                lines.append(f"  推荐方剂: {'、'.join(formulas[:4])}")
        elif ntype == "Symptom":
            lines.append(f"  标准症状名: {name}")
            # 所属证型
            syndromes = []
            for _, nbr, ed in g.edges(node_id, data=True):
                if ed.get("type") == "belongs_to":
                    syndromes.append(g.nodes[nbr].get("name", ""))
            if syndromes:
                lines.append(f"  所属证型: {'、'.join(syndromes[:5])}")

        # 度数
        degree = g.degree(node_id)
        lines.append(f"  关联数: {degree}")

        # 相邻节点摘要
        neighbor_types = {}
        for _, nbr in g.edges(node_id):
            nt = g.nodes[nbr].get("type", "?")
            neighbor_types[nt] = neighbor_types.get(nt, 0) + 1
        for nbr, _ in g.edges(node_id):
            nt = g.nodes[nbr].get("type", "?")
            neighbor_types[nt] = neighbor_types.get(nt, 0) + 1
        if neighbor_types:
            parts = [f"{nt}×{cnt}" for nt, cnt in sorted(neighbor_types.items())]
            lines.append(f"  相邻: {', '.join(parts)}")

        text = "\n".join(lines)
        self.detail_text.config(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", text)
        self.detail_text.config(state="disabled")

    def _set_detail(self, text: str):
        """设置详情文本"""
        self.detail_text.config(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", text)
        self.detail_text.config(state="disabled")

    def _update_filters(self):
        """更新类型过滤"""
        if self.search_var.get().strip():
            self._do_search()

    def _reset_view(self):
        """重置视图"""
        self.search_var.set("")
        self._set_detail("点击图谱中的节点查看详情")
        self._current_subgraph = None
        self._expand_history = []
        self._show_overview_graph()

    def reload_graph(self, force_rebuild=False):
        """重新加载图谱（在数据更新后调用）"""
        global KG_INSTANCE
        if force_rebuild:
            KG_INSTANCE = None
            self.graph = None
            self.query = None
            self._load_graph_async(on_done=self._reset_view)
        else:
            # 只清除本地引用，下次异步重新加载
            self.graph = None
            self.query = None
            self._reset_view()

    # ═══════════════════════════════════════════
    # 外部接口：展示查询结果
    # ═══════════════════════════════════════════

    def show_query_result(self, symptoms: List[str],
                          syndrome_name: Optional[str] = None,
                          herbs: Optional[List[str]] = None,
                          formulas: Optional[List[str]] = None):
        """
        根据查询结果展示相关子图（搜索后首页调用）

        收集症状/证型/药材/方剂相关节点，显示以证型为中心的1跳子图。
        """
        if self.graph is None:
            self._load_graph_async(
                on_done=lambda: self.show_query_result(
                    symptoms, syndrome_name, herbs, formulas
                )
            )
            return
        g = self.graph

        # 收集所有相关节点ID
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

        # 优先用首个症状作为中心，让用户看到自己输入的症状关联了什么
        primary = target_nodes[0]
        for n in target_nodes:
            if g.nodes[n].get("type") == "Symptom":
                primary = n
                break

        self.search_var.set(g.nodes[primary].get("name", ""))
        self._show_subgraph(primary, target_nodes)
