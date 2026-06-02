"""
中医药智能问答系统 — 桌面主窗口（CustomTkinter 版）
"""

import json
import logging
import os
import queue
import threading
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import customtkinter as ctk

from desktop.service import TCMDesktopService

logger = logging.getLogger(__name__)

# 知识图谱可视化（可选依赖）
try:
    from desktop.kg_view import KnowledgeGraphView  # noqa: F401

    HAS_KG = True
except ImportError as exc:
    logger.warning("知识图谱模块加载失败: %s", exc)
    HAS_KG = False


class DesktopApp:
    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        self.service = TCMDesktopService()
        self.result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.selected_history_id: int | None = None
        self.query_in_flight = False

        from core.tcm_config import UI_QUICK_SYMPTOMS

        self.symptom_options = list(UI_QUICK_SYMPTOMS)[:5]
        self._configure_root()
        self._build_layout()
        self._load_history()
        self._poll_queue()
        self._start_nlu_preload()

    # ── 字体常量（整体放大） ──
    FONT = ("Microsoft YaHei UI", 17)
    FONT_SMALL = ("Microsoft YaHei UI", 15)
    FONT_XS = ("Microsoft YaHei UI", 14)
    FONT_TITLE = ("Microsoft YaHei UI", 24, "bold")
    FONT_SECTION = ("Microsoft YaHei UI", 19, "bold")
    FONT_HERO_SUB = ("Microsoft YaHei UI", 17)

    # 窗口配置
    def _configure_root(self):
        self.root.title("中医药智能问答系统")
        self.root.minsize(1080, 720)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # 整体布局
    def _build_layout(self):
        # 外层容器
        container = ctk.CTkFrame(self.root, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=20, pady=(16, 16))

        # ── Hero 区域 ──
        hero = ctk.CTkFrame(container, fg_color=("#1a3a2b", "#1e2e24"), corner_radius=8)
        hero.pack(fill="x", pady=(0, 0))

        # 装饰顶线
        ctk.CTkFrame(hero, fg_color="#c8953e", height=3, corner_radius=0).pack(
            fill="x", pady=(0, 12)
        )
        ctk.CTkLabel(
            hero,
            text="中医药智能问答系统",
            font=self.FONT_TITLE,
            text_color="#ffffff",
        ).pack(anchor="w", padx=20, pady=(8, 0))
        ctk.CTkLabel(
            hero,
            text="基于深度学习的中医药知识推荐 · 支持症状辨证与方剂推荐",
            font=self.FONT_HERO_SUB,
            text_color=("#c8ddd2", "#8ab09a"),
        ).pack(anchor="w", padx=20, pady=(4, 16))

        # ── 标签页 ──
        self._notebook = ctk.CTkTabview(container, corner_radius=6)
        self._notebook.pack(fill="both", expand=True, pady=(14, 0))

        self._notebook.add("首页")
        self._notebook.add("知识图谱")
        self._notebook.add("知识管理")
        self._notebook.add("历史记录")
        self._notebook.add("关于")

        self.home_tab = self._notebook.tab("首页")
        self.kg_tab = self._notebook.tab("知识图谱")
        self.km_tab = self._notebook.tab("知识管理")
        self.history_tab = self._notebook.tab("历史记录")
        self.about_tab = self._notebook.tab("关于")

        self._build_home_tab()
        self._build_kg_tab()
        self._build_km_tab()
        self._build_history_tab()
        self._build_about_tab()

    # 首页
    def _build_home_tab(self) -> None:
        # FIXME: 低分辨率下标签页显示不全，需要响应式布局
        container = ctk.CTkScrollableFrame(self.home_tab, fg_color="transparent")
        container.pack(fill="both", expand=True)

        # ── 搜索卡片 ──
        search_area = ctk.CTkFrame(container, corner_radius=8)
        search_area.pack(fill="x", pady=(0, 6))

        ctk.CTkFrame(search_area, fg_color="#c8953e", height=2, corner_radius=0).pack(
            fill="x", pady=(0, 10)
        )

        input_row = ctk.CTkFrame(search_area, fg_color="transparent")
        input_row.pack(fill="x", padx=16)

        self.query_input = ctk.CTkTextbox(
            input_row,
            height=36,
            font=self.FONT,
            wrap="word",
            border_width=1,
        )
        self.query_input.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.query_input.bind(
            "<Return>", lambda e: self._submit_query() or "break"
        )
        self.query_input.bind(
            "<Control-Return>", lambda e: self._submit_query() or "break"
        )
        self.query_input.bind(
            "<Escape>", lambda e: self._clear_query() or "break"
        )

        self.clear_button = ctk.CTkButton(
            input_row,
            text="清空",
            width=80,
            font=self.FONT_SMALL,
            command=self._clear_query,
        )
        self.clear_button.pack(side="right")

        self.submit_button = ctk.CTkButton(
            input_row,
            text="搜索",
            width=80,
            font=self.FONT_SMALL,
            command=self._submit_query,
        )
        self.submit_button.pack(side="right", padx=(6, 0))

        # ── 快捷标签 ──
        tags_row = ctk.CTkFrame(search_area, fg_color="transparent")
        tags_row.pack(fill="x", padx=16, pady=(6, 10))
        ctk.CTkLabel(
            tags_row,
            text="快捷:",
            font=self.FONT_XS,
            text_color=("gray50", "gray60"),
        ).pack(side="left", padx=(0, 4))
        for symptom in self.symptom_options:
            btn = ctk.CTkButton(
                tags_row,
                text=symptom,
                font=self.FONT_XS,
                width=60,
                height=26,
                fg_color=("#e9f2ed", "#2a3d32"),
                text_color=("#256a4b", "#6fbe92"),
                hover_color=("#256a4b", "#1d573c"),
                command=lambda v=symptom: self._append_symptom(v),
            )
            btn.pack(side="left", padx=2)

        # ── 状态 + 进度 ──
        status_line = ctk.CTkFrame(search_area, fg_color="transparent")
        status_line.pack(fill="x", padx=16, pady=(0, 10))
        self.status_var = ctk.StringVar(value="输入症状，搜索中医药知识")
        ctk.CTkLabel(
            status_line,
            textvariable=self.status_var,
            font=self.FONT_SMALL,
            text_color=("#256a4b", "#5ca882"),
        ).pack(side="left")
        self.progress = ctk.CTkProgressBar(
            status_line,
            width=140,
            mode="indeterminate",
        )
        self.progress.pack(side="right")
        self.progress.set(0)

        # ── 子标签页：推荐结果 + 知识图谱 ──
        self._home_notebook = ctk.CTkTabview(container, corner_radius=6)
        self._home_notebook.pack(fill="both", expand=True, pady=(4, 0))
        # 缩小子标签页字体
        self._home_notebook._segmented_button.configure(
            font=self.FONT_SMALL
        )

        self._home_notebook.add("推荐结果")
        self._home_notebook.add("知识图谱")

        # ── 推荐结果 ──
        result_tab = self._home_notebook.tab("推荐结果")
        top_panel = ctk.CTkFrame(result_tab, corner_radius=8)
        top_panel.pack(fill="both", expand=True, padx=4, pady=4)

        ctk.CTkFrame(top_panel, fg_color=("#256a4b", "#2d7a4f"), height=2, corner_radius=0).pack(
            fill="x", pady=(0, 8)
        )
        ctk.CTkLabel(
            top_panel,
            text="推荐结果",
            font=self.FONT_SECTION,
            text_color=("#256a4b", "#5ca882"),
        ).pack(anchor="w", padx=6)

        self.answer_text = ctk.CTkTextbox(
            top_panel,
            wrap="word",
            font=self.FONT,
            border_width=0,
            activate_scrollbars=True,
        )
        self.answer_text.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.answer_text.configure(state="disabled")

        # ── 知识图谱子标签 ──
        kg_tab_inner = self._home_notebook.tab("知识图谱")
        self._home_kg_frame = ctk.CTkFrame(kg_tab_inner, fg_color="transparent")
        self._home_kg_frame.pack(fill="both", expand=True)

        self._home_kg_view = None
        if HAS_KG:
            try:
                from desktop.kg_view import KnowledgeGraphView

                self._home_kg_view = KnowledgeGraphView(
                    self._home_kg_frame, standalone=False
                )
                self._home_kg_view.pack(fill="both", expand=True)
                self._home_kg_view._show_placeholder()
            except Exception as exc:
                import traceback

                traceback.print_exc()
                ctk.CTkLabel(
                    self._home_kg_frame,
                    text=f"知识图谱加载失败: {exc}",
                    text_color="#e74c3c",
                ).pack(expand=True)

        # ── 免责声明 ──
        disclaimer_frame = ctk.CTkFrame(
            container,
            fg_color=("#fef6e9", "#2a2520"),
            border_width=1,
            border_color=("#f0dca0", "#4a3f2a"),
            corner_radius=6,
        )
        disclaimer_frame.pack(fill="x", pady=(4, 10))
        ctk.CTkLabel(
            disclaimer_frame,
            text="⚕ 免责声明",
            font=("Microsoft YaHei UI", 11, "bold"),
            text_color=("#7a5a16", "#c8a050"),
        ).pack(anchor="w", padx=14, pady=(6, 1))
        self.disclaimer_var = ctk.StringVar(
            value="本系统仅提供中医药知识科普和参考信息，不构成医疗诊断或治疗建议。如有健康问题，请咨询专业医疗机构或执业医师。"
        )
        ctk.CTkLabel(
            disclaimer_frame,
            textvariable=self.disclaimer_var,
            justify="left",
            font=self.FONT_XS,
            text_color=("#7a5a16", "#c8a050"),
        ).pack(anchor="w", padx=14, pady=(0, 6))

    # 知识图谱标签页
    def _build_kg_tab(self) -> None:
        if not HAS_KG:
            ctk.CTkLabel(
                self.kg_tab,
                text="知识图谱模块不可用",
                text_color=("gray50", "gray60"),
                font=self.FONT,
            ).pack(expand=True)
            return

        try:
            from desktop.kg_view import KnowledgeGraphView

            self._kg_tab_view = KnowledgeGraphView(self.kg_tab, standalone=True)
            self._kg_tab_view.pack(fill="both", expand=True)
            self.root.after(300, self._delayed_kg_overview)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            ctk.CTkLabel(
                self.kg_tab,
                text=f"知识图谱加载失败: {exc}",
                text_color="#e74c3c",
                font=self.FONT,
            ).pack(expand=True)

    def _delayed_kg_overview(self):
        try:
            if hasattr(self, "_kg_tab_view") and self._kg_tab_view is not None:
                self._kg_tab_view._show_overview_graph()
        except Exception as exc:
            logger.exception("知识图谱概览加载失败")
            ctk.CTkLabel(
                self.kg_tab,
                text=f"知识图谱概览生成失败: {exc}",
                text_color="#e74c3c",
            ).pack(expand=True)

    # 知识管理标签页
    def _build_km_tab(self) -> None:
        try:
            from desktop.knowledge_manager import KnowledgeManagerView

            kb = getattr(self.service, "kb", None)
            self._km_view = KnowledgeManagerView(self.km_tab, kb=kb)
            self._km_view.pack(fill="both", expand=True)
        except Exception as exc:
            logger.warning("知识管理标签页加载失败: %s", exc)
            ctk.CTkLabel(
                self.km_tab,
                text=f"知识管理模块加载失败: {exc}",
                text_color="#e74c3c",
            ).pack(expand=True)

    # 历史记录标签页
    def _build_history_tab(self) -> None:
        wrapper = ctk.CTkFrame(self.history_tab, fg_color="transparent")
        wrapper.pack(fill="both", expand=True)

        # ── 工具栏 ──
        toolbar = ctk.CTkFrame(wrapper, corner_radius=8)
        toolbar.pack(fill="x", padx=4, pady=(4, 0))

        ctk.CTkFrame(toolbar, fg_color=("#256a4b", "#2d7a4f"), height=2, corner_radius=0).pack(
            fill="x", pady=(0, 12)
        )

        title_label = ctk.CTkLabel(
            toolbar, text="查询历史", font=self.FONT_SECTION
        )
        title_label.pack(side="left", padx=12)

        # 搜索框 + 标签
        self.history_search_var = ctk.StringVar()
        search_entry = ctk.CTkEntry(
            toolbar,
            textvariable=self.history_search_var,
            width=200,
            font=self.FONT_XS,
            placeholder_text="搜索历史记录…",
        )
        search_entry.pack(side="right", padx=(4, 6))
        search_entry.bind("<KeyRelease>", lambda _event: self._load_history())
        ctk.CTkLabel(toolbar, text="搜索：", font=self.FONT_XS).pack(side="right")

        ctk.CTkButton(
            toolbar,
            text="导出全部",
            font=self.FONT_XS,
            command=self._export_history,
        ).pack(side="right", padx=(4, 0))
        ctk.CTkButton(
            toolbar,
            text="清空历史",
            font=self.FONT_XS,
            command=self._clear_history,
        ).pack(side="right", padx=(4, 0))
        ctk.CTkButton(
            toolbar,
            text="刷新",
            font=self.FONT_XS,
            command=self._load_history,
        ).pack(side="right", padx=(4, 0))

        # ── 内容区：左右分栏（手动布局替代 Panedwindow） ──
        content = ctk.CTkFrame(wrapper, fg_color="transparent")
        content.pack(fill="both", expand=True, pady=(10, 4), padx=4)

        # 左侧：列表
        list_frame = ctk.CTkFrame(content, corner_radius=8)
        list_frame.pack(side="left", fill="both", expand=True, padx=(0, 4))

        ctk.CTkFrame(list_frame, fg_color=("#256a4b", "#2d7a4f"), height=2, corner_radius=0).pack(
            fill="x", pady=(0, 10)
        )
        ctk.CTkLabel(
            list_frame, text="记录列表", font=self.FONT_SECTION
        ).pack(anchor="w", padx=10)

        tree_frame = ctk.CTkFrame(list_frame, fg_color="transparent")
        tree_frame.pack(fill="both", expand=True, padx=6, pady=(6, 6))

        tree_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical")
        tree_scrollbar.pack(side="right", fill="y")

        style = ttk.Style(tree_frame)
        style.configure("History.Treeview", font=("Microsoft YaHei UI", 16), rowheight=38)
        style.configure("History.Treeview.Heading", font=("Microsoft YaHei UI", 15, "bold"))
        self.history_tree = ttk.Treeview(
            tree_frame,
            columns=("timestamp", "query"),
            show="headings",
            selectmode="browse",
            yscrollcommand=tree_scrollbar.set,
            style="History.Treeview",
        )
        tree_scrollbar.config(command=self.history_tree.yview)
        self.history_tree.heading("timestamp", text="时间")
        self.history_tree.heading("query", text="问题")
        self.history_tree.column("timestamp", width=170, anchor="w")
        self.history_tree.column("query", width=350, anchor="w")
        self.history_tree.pack(fill="both", expand=True)
        self.history_tree.bind("<<TreeviewSelect>>", self._on_history_select)

        # 右侧：详情
        detail_frame = ctk.CTkFrame(content, corner_radius=8)
        detail_frame.pack(side="right", fill="both", expand=False, padx=(4, 0))
        # 设定固定宽度比例（约40%）
        detail_frame.configure(width=360)
        detail_frame.pack_propagate(False)

        ctk.CTkFrame(detail_frame, fg_color=("#256a4b", "#2d7a4f"), height=2, corner_radius=0).pack(
            fill="x", pady=(0, 10)
        )

        actions = ctk.CTkFrame(detail_frame, fg_color="transparent")
        actions.pack(fill="x", padx=8)
        ctk.CTkLabel(
            actions, text="记录详情", font=self.FONT_SECTION
        ).pack(side="left")
        ctk.CTkButton(
            actions,
            text="删除",
            font=self.FONT_XS,
            width=60,
            command=self._delete_selected_history,
        ).pack(side="right")

        self.history_detail = ctk.CTkTextbox(
            detail_frame,
            wrap="word",
            font=self.FONT_SMALL,
            border_width=1,
        )
        self.history_detail.pack(fill="both", expand=True, padx=8, pady=(8, 8))
        self.history_detail.configure(state="normal")
        self.history_detail.insert(
            "1.0",
            "选择左侧一条记录后，这里会显示完整问题、回答、症状和时间。",
        )
        self.history_detail.configure(state="disabled")

    # 关于标签页
    def _build_about_tab(self) -> None:
        # 外层滚动区
        about_frame = ctk.CTkScrollableFrame(self.about_tab, corner_radius=8)
        about_frame.pack(fill="both", expand=True, padx=8, pady=8)

        ctk.CTkFrame(about_frame, fg_color=("#256a4b", "#2d7a4f"), height=2, corner_radius=0).pack(
            fill="x", pady=(0, 16)
        )
        ctk.CTkLabel(
            about_frame,
            text="关于系统",
            font=self.FONT_SECTION,
        ).pack(anchor="w")
        ctk.CTkLabel(
            about_frame,
            text="中医药智能问答系统",
            font=("Microsoft YaHei UI", 11),
            text_color=("gray50", "gray60"),
        ).pack(anchor="w", pady=(2, 18))

        sections = [
            (
                "📋 项目简介",
                "中医药智能问答系统是一款基于深度学习与知识图谱的中医辅助诊疗参考工具。\n"
                "系统搭载 TCMNet Transformer 深度学习模型，融合中医药知识图谱进行智能辨证推理，\n"
                "覆盖 30 种证型分类体系（肺系、心系、肝胆系、脾胃系、肾系、气血津液等），\n"
                "支持从症状描述到证型判定、方剂推荐、药材与用量建议的完整推理管线。",
            ),
            (
                "🔧 主要功能",
                "1. 智能辨证与方剂推荐 — 输入症状描述，自动进行证型分析并推荐方剂与药材（含用量）\n"
                "2. 智能搜索引擎 — 支持口语化症状识别与同义词扩展（272 条口语映射），精准匹配知识库\n"
                "3. 配伍安全筛查 — 自动过滤十八反十九畏、孕妇禁忌及毒性药材配伍\n"
                "4. 知识图谱可视化 — 以交互式图谱展示症状、证型、药材、方剂之间的关联关系\n"
                "5. 知识库管理 — 支持对症状、证型、药材、方剂等知识的增删改查与导入导出\n"
                "6. 历史记录管理 — 查看、搜索、导出和清理历史问诊记录",
            ),
            (
                "⚕ 使用提醒",
                "本系统仅提供中医药知识参考，不构成医疗诊断或治疗建议。\n"
                "系统生成的方剂仅供参考，实际用药请以执业医师诊断为准。\n"
                "如有持续不适，请及时咨询专业医疗机构或执业医师。",
            ),
            (
                "👨‍💻 开发团队",
                "李俊贤 · 王子璇 · 周洪宇 · 刘丁夺",
            ),
        ]
        for title, body in sections:
            card = ctk.CTkFrame(
                about_frame,
                corner_radius=8,
                border_width=1,
            )
            card.pack(fill="x", pady=(0, 12))
            ctk.CTkLabel(
                card,
                text=title,
                font=("Microsoft YaHei UI", 14, "bold"),
                text_color=("#256a4b", "#5ca882"),
            ).pack(anchor="w", padx=18, pady=(14, 6))
            ctk.CTkLabel(
                card,
                text=body,
                justify="left",
                font=self.FONT_SMALL,
            ).pack(anchor="w", padx=18, pady=(0, 14))

    # 工具方法
    def _set_text(self, widget, value):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value.strip() if value else "")
        widget.configure(state="disabled")

    def _append_symptom(self, symptom: str) -> None:
        current = self.query_input.get("1.0", "end").strip()
        updated = f"{current}，{symptom}" if current else symptom
        self.query_input.delete("1.0", "end")
        self.query_input.insert("1.0", updated)
        self.query_input.focus_set()

    def _clear_query(self):
        self.query_input.delete("1.0", "end")
        self.status_var.set("输入已清空")

    # NLU 模型预加载
    def _start_nlu_preload(self) -> None:
        from desktop.service import preload_nlu_async

        self.status_var.set("⏳ AI 模型加载中，请稍候…")
        self._nlu_load_start = __import__("time").time()
        preload_nlu_async(callback=self._on_nlu_loaded)
        self._poll_nlu_status()

    def _on_nlu_loaded(self, success: bool, message: str) -> None:
        elapsed = __import__("time").time() - self._nlu_load_start
        if success:
            logger.info("NLU 模型就绪（耗时 %.1f 秒）", elapsed)
        else:
            logger.error("NLU 模型加载失败: %s", message)

    def _poll_nlu_status(self) -> None:
        from desktop.service import get_nlu_status

        loading, ready, error = get_nlu_status()
        if ready:
            self.status_var.set("✅ AI 模型已就绪，输入症状开始查询")
            self.submit_button.configure(state="normal")
            return
        if error:
            self.status_var.set(f"⚠ 模型加载失败: {error[:60]}")
            self.submit_button.configure(state="normal")
            return
        if loading:
            elapsed = __import__("time").time() - self._nlu_load_start
            self.status_var.set(f"⏳ AI 模型加载中…（已等待 {elapsed:.0f} 秒）")
        self.root.after(1000, self._poll_nlu_status)

    # 查询逻辑
    def _submit_query(self) -> None:
        from desktop.service import is_nlu_loading, is_nlu_ready, get_nlu_status

        query = self.query_input.get("1.0", "end").strip()
        if self.query_in_flight:
            messagebox.showinfo("提示", "上一条查询仍在处理中，请稍候。")
            return
        if not query:
            messagebox.showwarning("提示", "请输入您的问题或症状。")
            return
        if len(query) > 1000:
            messagebox.showwarning("提示", "输入内容过长，请控制在 1000 字以内。")
            return
        if sum(1 for char in query if not char.isspace()) < 2:
            messagebox.showwarning("提示", "输入内容过短，请补充更完整的问题描述。")
            return

        # 模型状态检查
        if not is_nlu_ready():
            _, _, error = get_nlu_status()
            if error:
                messagebox.showerror(
                    "模型加载失败",
                    f"AI 模型未能成功加载，无法查询。\n\n错误信息：{error}\n\n请检查模型文件是否完整。",
                )
                return
            if is_nlu_loading():
                messagebox.showinfo(
                    "模型加载中",
                    "AI 模型正在后台加载，请稍候再试。\n\n加载完成后状态栏会显示「AI 模型已就绪」。",
                )
                return
            messagebox.showinfo("提示", "AI 模型尚未开始加载，请稍候。")
            return

        self.query_in_flight = True
        self.submit_button.configure(state="disabled")
        self.progress.start()
        self.status_var.set("正在分析您的问题，请稍候...")
        self._set_text(self.answer_text, "")

        thread = threading.Thread(target=self._run_query, args=(query,), daemon=True)
        thread.start()

    def _run_query(self, query: str) -> None:
        try:
            from desktop.service import detect_symptoms_nlu

            detected = detect_symptoms_nlu(query)
            logger.info("检测到症状: %s", detected)
            result = self.service.query(query, detected)
            self.result_queue.put(("success", result))
        except ValueError as exc:
            logger.warning("输入校验失败: %s", exc)
            self.result_queue.put(("error", f"输入有误：{exc}"))
        except RuntimeError as exc:
            logger.exception("NLU 或 AI 模型错误")
            self.result_queue.put(("error", f"模型错误：{exc}"))
        except Exception as exc:
            logger.exception("桌面端查询失败")
            self.result_queue.put(("error", str(exc)))

    def _poll_queue(self) -> None:
        try:
            while True:
                status, payload = self.result_queue.get_nowait()
                self._finish_query(status, payload)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_queue)

    def _finish_query(self, status: str, payload: Any) -> None:
        self.progress.stop()
        self.query_in_flight = False
        self.submit_button.configure(state="normal")
        if status == "error":
            err_msg = str(payload)
            user_msg = "查询过程中出现异常，请稍后重试。"
            if "timeout" in err_msg.lower() or "超时" in err_msg:
                user_msg = "AI 模型响应超时，请稍后重试。"
            elif "not available" in err_msg.lower() or "不可用" in err_msg:
                user_msg = "AI 模型暂未加载完成，请等待状态栏显示「就绪」后再试。"
            elif "CUDA" in err_msg or "显存" in err_msg:
                user_msg = "GPU 资源不足，系统将自动切换到 CPU 模式。"
            elif "内存" in err_msg or "memory" in err_msg.lower():
                user_msg = "系统内存不足，请关闭其他程序后重试。"
            self.status_var.set("⚠ 查询失败")
            messagebox.showerror("查询失败", user_msg)
            return

        data: dict[str, Any] = payload
        answer = data.get("answer", "")

        self.status_var.set("✅ AI 模型回答完成")
        self.disclaimer_var.set(
            data.get("disclaimer")
            or "本系统仅提供中医药知识科普和参考信息，不构成医疗诊断或治疗建议。"
        )
        self._set_text(self.answer_text, answer)
        self._load_history()

        # 切换到推荐结果标签页
        if hasattr(self, "_home_notebook"):
            self._home_notebook.set("推荐结果")

        # 搜索后更新知识图谱
        if self._home_kg_view is not None:
            try:
                knowledge = data.get("knowledge", {})
                symptoms = knowledge.get("symptoms", [])
                syndrome_info = knowledge.get("syndrome", {})
                syndrome_name = (
                    syndrome_info.get("name")
                    if isinstance(syndrome_info, dict)
                    else None
                )
                herb_names = [h["name"] for h in knowledge.get("herbs", [])]
                formula_names = [f["name"] for f in knowledge.get("formulas", [])]

                self._home_kg_view.show_query_result(
                    symptoms=symptoms,
                    syndrome_name=syndrome_name,
                    herbs=herb_names,
                    formulas=formula_names,
                )
            except Exception as exc:
                logger.exception("首页知识图谱展示失败")
                self.status_var.set(f"⚠ 知识图谱加载失败: {exc}")

    # 历史记录管理
    def _load_history(self) -> None:
        search = (
            self.history_search_var.get().strip().lower()
            if hasattr(self, "history_search_var")
            else ""
        )
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)

        import difflib

        for record in self.service.list_history():
            haystack = f"{record.get('query', '')} {record.get('answer', '')}".lower()
            if search:
                if search in haystack:
                    pass
                elif all(c in haystack for c in search):
                    score = difflib.SequenceMatcher(None, search, haystack).ratio()
                    if score < 0.2:
                        continue
                else:
                    continue
            preview = record.get("query", "").replace("\n", " ")
            if len(preview) > 34:
                preview = preview[:34] + "..."
            self.history_tree.insert(
                "",
                "end",
                iid=str(record["id"]),
                values=(record.get("timestamp", ""), preview),
            )

    def _on_history_select(self, _event: Any) -> None:
        selection = self.history_tree.selection()
        if not selection:
            return
        self.selected_history_id = int(selection[0])
        record = next(
            (
                item
                for item in self.service.list_history()
                if item.get("id") == self.selected_history_id
            ),
            None,
        )
        if not record:
            return
        symptoms = "、".join(record.get("symptoms", [])) or "未识别"
        detail = (
            f"时间：{record.get('timestamp', '')}\n"
            f"症状：{symptoms}\n\n"
            f"问题：\n{record.get('query', '')}\n\n"
            f"回答：\n{record.get('answer', '')}"
        )
        self._set_text(self.history_detail, detail)

    def _delete_selected_history(self) -> None:
        if self.selected_history_id is None:
            messagebox.showinfo("提示", "请先选择一条历史记录。")
            return
        if not messagebox.askyesno("确认删除", "确定删除当前历史记录吗？"):
            return
        if self.service.delete_history_item(self.selected_history_id):
            self.selected_history_id = None
            self._set_text(self.history_detail, "记录已删除。")
            self._load_history()

    def _clear_history(self) -> None:
        if not self.service.history_data:
            messagebox.showinfo("提示", "当前没有可清空的历史记录。")
            return
        if not messagebox.askyesno("确认清空", "确定清空全部历史记录吗？"):
            return
        self.service.clear_history()
        self.selected_history_id = None
        self._set_text(self.history_detail, "历史记录已清空。")
        self._load_history()

    def _export_history(self) -> None:
        records = self.service.export_history()
        if not records:
            messagebox.showinfo("提示", "当前没有可导出的历史记录。")
            return
        if not messagebox.askyesno(
            "确认导出",
            "即将导出最近 100 条历史记录。是否继续？",
        ):
            return

        file_path = filedialog.asksaveasfilename(
            title="导出历史记录",
            defaultextension=".json",
            initialfile="tcm_history_export.json",
            filetypes=[
                ("JSON 文件", "*.json"),
                ("文本文件", "*.txt"),
                ("所有文件", "*.*"),
            ],
        )
        if not file_path:
            return

        path = Path(file_path)
        try:
            if path.suffix.lower() == ".txt":
                lines: list[str] = []
                for item in records:
                    lines.append(
                        f"时间：{item.get('timestamp', '')}\n"
                        f"问题：{item.get('query', '')}\n"
                        f"回答：{item.get('answer', '')}\n" + "-" * 50
                    )
                path.write_text("\n".join(lines), encoding="utf-8")
            else:
                path.write_text(
                    json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            messagebox.showinfo("导出成功", f"历史记录已导出到：\n{path}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))

    def _on_close(self):
        self.service.close()
        self.root.destroy()


def run_desktop_app() -> None:
    """启动桌面应用（含 Splash 动画）"""
    # 全局 CTk 配置 — 在创建任何窗口前设置
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("green")
    ctk.set_widget_scaling(1.25)  # 整体缩放 25%

    root = ctk.CTk()
    root.state("zoomed")

    # 先显示启动画面，再构建主界面（覆盖初始化耗时）
    from desktop.splash import SplashScreen
    splash = SplashScreen(parent=root)
    root.update()

    DesktopApp(root)
    root.update()

    splash.animate_and_close(duration=3.0)
    root.mainloop()
