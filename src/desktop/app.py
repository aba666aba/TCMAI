import json
import logging
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Dict, List, Optional

from core.symptom_pipeline import SymptomPipeline
from desktop.service import TCMDesktopService

# 知识图谱可视化（可选依赖）
try:
    from desktop.kg_view import KnowledgeGraphView
    HAS_KG = True
except ImportError as exc:
    logger.warning("知识图谱模块加载失败: %s", exc)
    HAS_KG = False

logger = logging.getLogger(__name__)


class DesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.service = TCMDesktopService()
        self.result_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self.selected_history_id: Optional[int] = None
        self.query_in_flight = False
        self.palette = {
            "bg": "#f2efe8",
            "card": "#fcfaf7",
            "deep": "#1a3a2b",
            "primary": "#256a4b",
            "primary_hover": "#1d573c",
            "primary_light": "#409b6e",
            "accent": "#c8953e",
            "text": "#2b3d33",
            "muted": "#7a8e80",
            "warning_bg": "#fef6e9",
            "warning_fg": "#7a5a16",
            "warning_border": "#f0dca0",
            "border": "#d4ddd6",
            "tag_bg": "#e9f2ed",
            "tag_hover": "#256a4b",
            "input_bg": "#fefefe",
            "shadow": "#dfdcd5",
        }
        from core.tcm_config import UI_QUICK_SYMPTOMS
        self.symptom_options = list(UI_QUICK_SYMPTOMS)[:5]

        self._configure_root()
        self._build_style()
        self._build_layout()
        self._load_history()
        self._poll_queue()

    def _configure_root(self) -> None:
        self.root.title("中医药智能问答系统")
        self.root.minsize(1080, 720)
        self.root.configure(bg=self.palette["bg"])
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        # 窗口创建后延迟应用深色标题栏
        if sys.platform == "win32":
            self.root.after_idle(self._set_dark_titlebar)

    def _set_dark_titlebar(self) -> None:
        """通过 DWM API 设置深色标题栏"""
        try:
            import ctypes
            dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)

            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            DWMWA_CAPTION_COLOR = 35

            hwnd = ctypes.windll.user32.GetAncestor(
                self.root.winfo_id(), 2  # GA_ROOT
            )
            if not hwnd:
                return

            dark = ctypes.c_int(1)
            dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
                ctypes.byref(dark), ctypes.sizeof(dark)
            )

            # 标题栏颜色 BGR(深绿): 0x002B3A1A = RGB 0x1A3A2B
            color = ctypes.c_int(0x002B3A1A)
            try:
                dwmapi.DwmSetWindowAttribute(
                    hwnd, DWMWA_CAPTION_COLOR,
                    ctypes.byref(color), ctypes.sizeof(color)
                )
            except Exception:
                pass
        except Exception:
            pass

    def _build_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", font=("Microsoft YaHei UI", 10))
        style.configure("App.TFrame", background=self.palette["bg"])
        style.configure("Card.TFrame", background=self.palette["card"])
        style.configure("Hero.TFrame", background=self.palette["deep"])

        # Hero 标题
        style.configure(
            "Title.TLabel",
            background=self.palette["deep"],
            foreground="#ffffff",
            font=("Microsoft YaHei UI", 20, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.palette["deep"],
            foreground="#c8ddd2",
            font=("Microsoft YaHei UI", 10),
        )
        # 装饰性副标题
        style.configure(
            "Accent.TLabel",
            background=self.palette["deep"],
            foreground=self.palette["accent"],
            font=("Microsoft YaHei UI", 9),
        )
        # 卡片区标题
        style.configure(
            "SectionTitle.TLabel",
            background=self.palette["card"],
            foreground=self.palette["primary"],
            font=("Microsoft YaHei UI", 15, "bold"),
        )
        style.configure(
            "SectionSubtitle.TLabel",
            background=self.palette["card"],
            foreground=self.palette["muted"],
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "Body.TLabel",
            background=self.palette["card"],
            foreground=self.palette["text"],
        )
        style.configure(
            "Muted.TLabel",
            background=self.palette["card"],
            foreground=self.palette["muted"],
            font=("Microsoft YaHei UI", 9),
        )
        # 主按钮
        style.configure(
            "Primary.TButton",
            background=self.palette["primary"],
            foreground="#ffffff",
            padding=(22, 11),
            borderwidth=0,
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", self.palette["primary_hover"]), ("disabled", "#96b8ab")],
        )
        # 次要按钮
        style.configure(
            "Secondary.TButton",
            background="#e8efe9",
            foreground=self.palette["primary"],
            padding=(14, 9),
            borderwidth=0,
            font=("Microsoft YaHei UI", 10),
        )
        style.map("Secondary.TButton", background=[("active", "#d4e4da")])
        # 表格
        style.configure(
            "Treeview",
            font=("Microsoft YaHei UI", 10),
            rowheight=36,
            background="#ffffff",
            fieldbackground="#ffffff",
            foreground=self.palette["text"],
        )
        style.configure(
            "Treeview.Heading",
            font=("Microsoft YaHei UI", 10, "bold"),
            background="#e8efe9",
            foreground=self.palette["primary"],
        )
        # Notebook 标签
        style.configure(
            "TNotebook",
            background=self.palette["bg"],
            borderwidth=0,
        )
        style.configure(
            "TNotebook.Tab",
            background="#e0e8e2",
            foreground=self.palette["text"],
            padding=(20, 10),
            font=("Microsoft YaHei UI", 10),
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.palette["card"])],
            foreground=[("selected", self.palette["primary"])],
            padding=[("selected", (22, 12))],
            font=[("selected", ("Microsoft YaHei UI", 10, "bold"))],
        )
        # 滚动条
        style.configure(
            "Vertical.TScrollbar",
            background="#dce3dd",
            arrowcolor=self.palette["muted"],
            troughcolor=self.palette["bg"],
            borderwidth=0,
        )

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, style="App.TFrame", padding=(20, 16))
        container.pack(fill="both", expand=True)

        # ── Hero 区域 ──
        hero = ttk.Frame(container, style="Hero.TFrame", padding=(28, 20))
        hero.pack(fill="x")
        # 装饰顶线
        tk.Frame(hero, bg=self.palette["accent"], height=3).pack(fill="x", pady=(0, 12))
        ttk.Label(hero, text="中医药智能问答系统", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            hero,
            text="基于深度学习的中医药知识推荐 · 支持症状辨证与方剂推荐",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        # ── 阴影装饰（Hero 底部投影效果）──
        shadow_bar = tk.Frame(container, bg=self.palette["shadow"], height=1)
        shadow_bar.pack(fill="x", pady=(0, 0))

        # ── Notebook ──
        notebook = ttk.Notebook(container)
        notebook.pack(fill="both", expand=True, pady=(14, 0))

        self.home_tab = ttk.Frame(notebook, style="App.TFrame", padding=4)
        self.kg_tab = ttk.Frame(notebook, style="App.TFrame", padding=4)
        self.history_tab = ttk.Frame(notebook, style="App.TFrame", padding=4)
        self.about_tab = ttk.Frame(notebook, style="App.TFrame", padding=4)
        notebook.add(self.home_tab, text="首页")
        notebook.add(self.kg_tab, text="知识图谱")
        notebook.add(self.history_tab, text="历史记录")
        notebook.add(self.about_tab, text="关于")

        self._build_home_tab()
        self._build_kg_tab()
        self._build_history_tab()
        self._build_about_tab()

    def _build_home_tab(self) -> None:
        """上下布局：搜索栏 + 推荐结果 + 搜索后显示知识图谱"""
        container = tk.Frame(self.home_tab, bg=self.palette["bg"])
        container.pack(fill="both", expand=True)

        # ── 搜索区域（顶部固定） ──
        search_area = tk.Frame(container, bg=self.palette["card"], padx=20, pady=14,
                               highlightthickness=1, highlightbackground=self.palette["border"])
        search_area.pack(fill="x", pady=(0, 6))

        tk.Frame(search_area, bg=self.palette["accent"], height=2).pack(fill="x", pady=(0, 8))

        input_row = tk.Frame(search_area, bg=self.palette["card"])
        input_row.pack(fill="x")

        self.query_input = tk.Text(
            input_row, height=1, wrap="word",
            font=("Microsoft YaHei UI", 12),
            relief="solid", bd=1,
            background=self.palette["input_bg"],
            foreground=self.palette["text"],
            insertbackground=self.palette["primary"],
            highlightthickness=0, padx=12, pady=4,
        )
        self.query_input.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.clear_button = ttk.Button(
            input_row, text="清空", style="Primary.TButton", width=6,
            command=self._clear_query
        )
        self.clear_button.pack(side="right")

        self.submit_button = ttk.Button(
            input_row, text="搜索", style="Primary.TButton", width=6,
            command=self._submit_query
        )
        self.submit_button.pack(side="right", padx=(6, 0))

        # 快捷标签
        tags_row = tk.Frame(search_area, bg=self.palette["card"])
        tags_row.pack(fill="x", pady=(6, 0))
        tk.Label(tags_row, text="快捷: ", bg=self.palette["card"],
                 fg=self.palette["muted"], font=("Microsoft YaHei UI", 9)).pack(side="left")
        for symptom in self.symptom_options:
            btn = tk.Button(
                tags_row, text=symptom,
                relief="flat", bd=0,
                bg=self.palette["tag_bg"], fg=self.palette["primary"],
                activebackground=self.palette["tag_hover"], activeforeground="#ffffff",
                cursor="hand2", font=("Microsoft YaHei UI", 9),
                padx=10, pady=3,
                command=lambda v=symptom: self._append_symptom(v),
            )
            btn.pack(side="left", padx=2)

        # 状态行 + 进度条
        status_line = tk.Frame(search_area, bg=self.palette["card"])
        status_line.pack(fill="x", pady=(4, 0))
        self.status_var = tk.StringVar(value="输入症状，搜索中医药知识")
        tk.Label(status_line, textvariable=self.status_var,
                 bg=self.palette["card"], fg=self.palette["primary"],
                 font=("Microsoft YaHei UI", 10, "bold")).pack(side="left")
        self.progress = ttk.Progressbar(
            status_line, mode="indeterminate", length=140,
            style="green.Horizontal.TProgressbar"
        )
        self.progress.pack(side="right")

        # ── 结果区域（子标签页：推荐结果 + 知识图谱） ──
        self._home_notebook = ttk.Notebook(container)
        self._home_notebook.pack(fill="both", expand=True, pady=(4, 0))

        # ── 标签1：推荐结果 ──
        result_tab = ttk.Frame(self._home_notebook, style="App.TFrame", padding=4)
        self._home_notebook.add(result_tab, text="推荐结果")

        top_panel = ttk.Frame(result_tab, style="Card.TFrame", padding=10)
        top_panel.pack(fill="both", expand=True)

        tk.Frame(top_panel, bg=self.palette["primary"], height=2).pack(fill="x", pady=(0, 8))
        tk.Label(top_panel, text="推荐结果",
                 bg=self.palette["card"], fg=self.palette["primary"],
                 font=("Microsoft YaHei UI", 13, "bold")).pack(anchor="w")

        self.answer_text = tk.Text(
            top_panel, wrap="word",
            font=("Microsoft YaHei UI", 11),
            relief="flat", bd=0,
            background=self.palette["card"],
            foreground=self.palette["text"],
            padx=6, pady=6,
            height=10, state="disabled",
            highlightthickness=0,
        )
        self.answer_text.pack(fill="both", expand=True)

        # ── 标签2：知识图谱 ──
        kg_tab = ttk.Frame(self._home_notebook, style="App.TFrame", padding=4)
        self._home_notebook.add(kg_tab, text="知识图谱")

        self._home_kg_frame = tk.Frame(kg_tab, bg=self.palette["bg"])
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
                tk.Label(self._home_kg_frame, text=f"知识图谱加载失败: {exc}",
                         fg="#c00").pack(expand=True)

        # ── 免责声明（底部） ──
        disclaimer_frame = tk.Frame(
            container, bg=self.palette["warning_bg"],
            highlightthickness=1, highlightbackground=self.palette["warning_border"]
        )
        disclaimer_frame.pack(fill="x", pady=(4, 0))
        tk.Label(
            disclaimer_frame,
            text="⚕ 免责声明",
            bg=self.palette["warning_bg"],
            fg=self.palette["warning_fg"],
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(anchor="w", padx=14, pady=(4, 1))
        self.disclaimer_var = tk.StringVar(
            value="本系统仅提供中医药知识科普和参考信息，不构成医疗诊断或治疗建议。如有健康问题，请咨询专业医疗机构或执业医师。"
        )
        tk.Label(
            disclaimer_frame,
            textvariable=self.disclaimer_var,
            justify="left", wraplength=1200,
            bg=self.palette["warning_bg"],
            fg=self.palette["warning_fg"],
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", padx=14, pady=(0, 4))

    def _build_kg_tab(self) -> None:
        """独立知识图谱标签页"""
        if not HAS_KG:
            tk.Label(self.kg_tab, text="知识图谱模块不可用",
                     fg="#888", font=("Microsoft YaHei UI", 12)
                     ).pack(expand=True)
            return

        try:
            from desktop.kg_view import KnowledgeGraphView
            self._kg_tab_view = KnowledgeGraphView(self.kg_tab, standalone=True)
            self._kg_tab_view.pack(fill="both", expand=True)
            # 延迟加载，避免阻塞主界面初始化
            self.root.after(300, self._delayed_kg_overview)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            tk.Label(self.kg_tab, text=f"知识图谱加载失败: {exc}",
                     fg="#c00", font=("Microsoft YaHei UI", 12)
                     ).pack(expand=True)

    def _delayed_kg_overview(self):
        """延迟加载 KG 概览图"""
        try:
            if hasattr(self, '_kg_tab_view') and self._kg_tab_view is not None:
                self._kg_tab_view._show_overview_graph()
        except Exception as exc:
            logger.exception("知识图谱概览加载失败")
            tk.Label(self.kg_tab, text=f"知识图谱概览生成失败: {exc}",
                     fg="#c00").pack(expand=True)

    def _build_history_tab(self) -> None:
        wrapper = ttk.Frame(self.history_tab, style="App.TFrame")
        wrapper.pack(fill="both", expand=True)

        toolbar = ttk.Frame(wrapper, style="Card.TFrame", padding=16)
        toolbar.pack(fill="x")
        # 卡片内阴影顶线
        tk.Frame(toolbar, bg=self.palette["primary"], height=2).pack(fill="x", pady=(0, 12))
        ttk.Label(toolbar, text="查询历史", style="SectionTitle.TLabel").pack(side="left")
        self.history_search_var = tk.StringVar()
        entry = ttk.Entry(toolbar, textvariable=self.history_search_var, width=36, font=("Microsoft YaHei UI", 10))
        entry.pack(side="right", padx=(8, 0))
        entry.bind("<KeyRelease>", lambda _event: self._load_history())
        ttk.Button(toolbar, text="导出全部", style="Secondary.TButton", command=self._export_history).pack(
            side="right", padx=(8, 0)
        )
        ttk.Button(toolbar, text="清空历史", style="Secondary.TButton", command=self._clear_history).pack(
            side="right", padx=(8, 0)
        )
        ttk.Button(toolbar, text="刷新", style="Secondary.TButton", command=self._load_history).pack(
            side="right", padx=(8, 0)
        )

        content = ttk.Panedwindow(wrapper, orient="horizontal")
        content.pack(fill="both", expand=True, pady=(14, 0))
        list_frame = ttk.Frame(content, style="Card.TFrame", padding=16)
        detail_frame = ttk.Frame(content, style="Card.TFrame", padding=16)
        content.add(list_frame, weight=3)
        content.add(detail_frame, weight=2)

        # 列表装饰顶线
        tk.Frame(list_frame, bg=self.palette["primary"], height=2).pack(fill="x", pady=(0, 12))
        ttk.Label(list_frame, text="记录列表", style="SectionTitle.TLabel").pack(anchor="w")

        tree_frame = ttk.Frame(list_frame, style="Card.TFrame")
        tree_frame.pack(fill="both", expand=True, pady=(8, 0))

        tree_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical")
        tree_scrollbar.pack(side="right", fill="y")

        self.history_tree = ttk.Treeview(
            tree_frame,
            columns=("timestamp", "query"),
            show="headings",
            selectmode="browse",
            yscrollcommand=tree_scrollbar.set,
        )
        tree_scrollbar.config(command=self.history_tree.yview)
        self.history_tree.heading("timestamp", text="时间")
        self.history_tree.heading("query", text="问题")
        self.history_tree.column("timestamp", width=170, anchor="w")
        self.history_tree.column("query", width=510, anchor="w")
        self.history_tree.pack(fill="both", expand=True)
        self.history_tree.bind("<<TreeviewSelect>>", self._on_history_select)

        # 详情装饰顶线
        tk.Frame(detail_frame, bg=self.palette["primary"], height=2).pack(fill="x", pady=(0, 12))

        actions = ttk.Frame(detail_frame, style="Card.TFrame")
        actions.pack(fill="x")
        ttk.Label(actions, text="记录详情", style="SectionTitle.TLabel").pack(side="left")
        ttk.Button(
            actions, text="删除当前记录", style="Secondary.TButton", command=self._delete_selected_history
        ).pack(side="right")

        self.history_detail = ScrolledText(
            detail_frame,
            wrap="word",
            font=("Microsoft YaHei UI", 10),
            relief="flat",
            bd=0,
        )
        self.history_detail.pack(fill="both", expand=True, pady=(12, 0))
        self._configure_output_text(self.history_detail)
        self._set_text(self.history_detail, "选择左侧一条记录后，这里会显示完整问题、回答、症状和时间。")

    def _build_about_tab(self) -> None:
        frame = ttk.Frame(self.about_tab, style="Card.TFrame", padding=28)
        frame.pack(fill="both", expand=True)

        # 卡片内阴影顶线
        tk.Frame(frame, bg=self.palette["primary"], height=2).pack(fill="x", pady=(0, 16))

        ttk.Label(frame, text="关于系统", style="SectionTitle.TLabel").pack(anchor="w")
        ttk.Label(frame, text="中医药智能问答系统", style="SectionSubtitle.TLabel").pack(anchor="w", pady=(2, 18))

        sections = [
            (
                "📋 项目简介",
                "中医药智能问答系统用于提供中医药知识科普、症状参考分析和历史记录管理。\n基于 TCMNet Transformer 深度学习模型进行智能辨证与方剂推荐。",
            ),
            (
                "🔧 主要功能",
                "1. 输入症状描述进行智能辨证与方剂推荐\n2. 通过快捷标签快速补全症状描述\n3. 查看、搜索、导出和清理历史记录",
            ),
            (
                "⚕ 使用提醒",
                "本系统仅提供知识参考，不构成医疗诊断或治疗建议。\n如有持续不适，请及时咨询专业医疗机构或执业医师。",
            ),
            (
                "👨‍💻 开发团队",
                "李俊贤 · 王子璇 · 周洪宇 · 刘丁夺",
            ),
        ]
        for title, body in sections:
            card = tk.Frame(
                frame, bg="#f7faf7", highlightthickness=1, highlightbackground=self.palette["border"],
                padx=16, pady=14,
            )
            card.pack(fill="x", pady=(0, 10))
            tk.Label(
                card,
                text=title,
                bg="#f7faf7",
                fg=self.palette["primary"],
                font=("Microsoft YaHei UI", 12, "bold"),
            ).pack(anchor="w", padx=14, pady=(12, 6))
            tk.Label(
                card,
                text=body,
                justify="left",
                wraplength=1000,
                bg="#f7faf7",
                fg=self.palette["text"],
                font=("Microsoft YaHei UI", 10),
            ).pack(anchor="w", padx=14, pady=(0, 12))

    def _configure_output_text(self, widget: tk.Text) -> None:
        widget.configure(
            background="#fcfcfc",
            foreground=self.palette["text"],
            insertbackground=self.palette["primary"],
            highlightthickness=1,
            highlightbackground=self.palette["border"],
            highlightcolor=self.palette["primary_light"],
            padx=14,
            pady=12,
            state="disabled",
            relief="flat",
            borderwidth=0,
        )

    def _set_text(self, widget: tk.Text, value: str) -> None:
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

    def _clear_query(self) -> None:
        self.query_input.delete("1.0", "end")
        self.status_var.set("输入已清空")

    def _submit_query(self) -> None:
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

        self.query_in_flight = True
        self.submit_button.state(["disabled"])
        self.progress.start(10)
        self.status_var.set("正在分析您的问题，请稍候...")
        self._set_text(self.answer_text, "")

        thread = threading.Thread(target=self._run_query, args=(query,), daemon=True)
        thread.start()

    def _run_query(self, query: str) -> None:
        try:
            from src.desktop.service import detect_symptoms_nlu
            # NLU 优先，失败降级关键词
            detected = detect_symptoms_nlu(query)
            logger.info("检测到症状: %s", detected)
            result = self.service.query(query, detected)
            self.result_queue.put(("success", result))
        except RuntimeError as exc:
            logger.exception("AI 模型不可用")
            self.result_queue.put(("error", f"AI 模型不可用：{exc}"))
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
        self.submit_button.state(["!disabled"])
        if status == "error":
            self.status_var.set("AI 查询失败")
            messagebox.showerror("AI 查询失败", str(payload))
            return

        data: Dict[str, Any] = payload
        answer = data.get("answer", "")

        self.status_var.set("✅ AI 模型回答完成")
        self.disclaimer_var.set(
            data.get("disclaimer") or "本系统仅提供中医药知识科普和参考信息，不构成医疗诊断或治疗建议。"
        )
        self._set_text(self.answer_text, answer)
        self._load_history()

        # ── 切换到推荐结果标签页 ──
        if hasattr(self, '_home_notebook'):
            self._home_notebook.select(0)  # 0 = 推荐结果

        # ── 搜索后更新知识图谱标签页 ──
        if self._home_kg_view is not None:
            try:
                knowledge = data.get("knowledge", {})
                symptoms = knowledge.get("symptoms", [])
                syndrome_info = knowledge.get("syndrome", {})
                syndrome_name = syndrome_info.get("name") if isinstance(syndrome_info, dict) else None
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

    def _load_history(self) -> None:
        search = self.history_search_var.get().strip().lower() if hasattr(self, "history_search_var") else ""
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)

        import difflib
        for record in self.service.list_history():
            haystack = f"{record.get('query', '')} {record.get('answer', '')}".lower()
            if search:
                # 精确子串匹配
                if search in haystack:
                    pass
                # 模糊匹配：搜索词字符都在记录中且顺序一致
                elif all(c in haystack for c in search):
                    # 字符集匹配通过，再用评分过滤
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
            (item for item in self.service.list_history() if item.get("id") == self.selected_history_id),
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
            "即将导出最近 500 条历史记录。是否继续？",
        ):
            return

        file_path = filedialog.asksaveasfilename(
            title="导出历史记录",
            defaultextension=".json",
            initialfile="tcm_history_export.json",
            filetypes=[("JSON 文件", "*.json"), ("文本文件", "*.txt"), ("所有文件", "*.*")],
        )
        if not file_path:
            return

        path = Path(file_path)
        try:
            if path.suffix.lower() == ".txt":
                lines: List[str] = []
                for item in records:
                    lines.append(
                        f"时间：{item.get('timestamp', '')}\n"
                        f"问题：{item.get('query', '')}\n"
                        f"回答：{item.get('answer', '')}\n"
                        + "-" * 50
                    )
                path.write_text("\n".join(lines), encoding="utf-8")
            else:
                path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            messagebox.showinfo("导出成功", f"历史记录已导出到：\n{path}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))

    def _on_close(self) -> None:
        self.service.close()
        self.root.destroy()


def run_desktop_app() -> None:
    root = tk.Tk()
    DesktopApp(root)
    root.state("zoomed")  # 窗口最大化
    root.mainloop()
