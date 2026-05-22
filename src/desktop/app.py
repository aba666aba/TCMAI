import json
import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Dict, List, Optional

from core.symptom_pipeline import SymptomPipeline
from desktop.service import TCMDesktopService

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
        self.symptom_options = list(UI_QUICK_SYMPTOMS)

        self._configure_root()
        self._build_style()
        self._build_layout()
        self._load_history()
        self._poll_queue()

    def _configure_root(self) -> None:
        self.root.title("中医药智能问答系统")
        self.root.geometry("1280x840")
        self.root.minsize(1080, 720)
        self.root.configure(bg=self.palette["bg"])
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

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
            padding=(20, 8),
            font=("Microsoft YaHei UI", 10),
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.palette["card"])],
            foreground=[("selected", self.palette["primary"])],
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
        ttk.Label(
            hero,
            text="快速模式（规则引擎） · AI 模式（深度学习）",
            style="Accent.TLabel",
        ).pack(anchor="w", pady=(6, 0))

        # ── 阴影装饰（Hero 底部投影效果）──
        shadow_bar = tk.Frame(container, bg=self.palette["shadow"], height=1)
        shadow_bar.pack(fill="x", pady=(0, 0))

        # ── Notebook ──
        notebook = ttk.Notebook(container)
        notebook.pack(fill="both", expand=True, pady=(14, 0))

        self.home_tab = ttk.Frame(notebook, style="App.TFrame", padding=4)
        self.history_tab = ttk.Frame(notebook, style="App.TFrame", padding=4)
        self.about_tab = ttk.Frame(notebook, style="App.TFrame", padding=4)
        notebook.add(self.home_tab, text="首页")
        notebook.add(self.history_tab, text="历史记录")
        notebook.add(self.about_tab, text="关于")

        self._build_home_tab()
        self._build_history_tab()
        self._build_about_tab()

    def _build_home_tab(self) -> None:
        pane = ttk.Panedwindow(self.home_tab, orient="horizontal")
        pane.pack(fill="both", expand=True)

        left = ttk.Frame(pane, style="Card.TFrame", padding=24)
        right = ttk.Frame(pane, style="Card.TFrame", padding=24)
        pane.add(left, weight=2)
        pane.add(right, weight=3)

        # ── 左侧面板 ──

        # 卡片内阴影顶线
        tk.Frame(left, bg=self.palette["primary"], height=2).pack(fill="x", pady=(0, 16))

        ttk.Label(left, text="症状输入", style="SectionTitle.TLabel").pack(anchor="w")
        ttk.Label(left, text="描述您的症状或直接点击快捷标签", style="SectionSubtitle.TLabel").pack(anchor="w", pady=(2, 14))

        # 输入框
        input_frame = tk.Frame(left, bg=self.palette["border"], bd=1, relief="solid")
        input_frame.pack(fill="x")
        self.query_input = ScrolledText(
            input_frame,
            height=8,
            wrap="word",
            font=("Microsoft YaHei UI", 11),
            relief="flat",
            bd=0,
        )
        self.query_input.pack(fill="both", padx=2, pady=2)
        self.query_input.configure(
            background=self.palette["input_bg"],
            foreground=self.palette["text"],
            insertbackground=self.palette["primary"],
        )

        # 分隔线
        tk.Frame(left, bg=self.palette["border"], height=1).pack(fill="x", pady=(16, 10))

        # 快捷标签
        ttk.Label(left, text="快捷症状", style="Body.TLabel").pack(anchor="w", pady=(0, 8))
        tags = ttk.Frame(left, style="Card.TFrame")
        tags.pack(fill="x")
        for index, symptom in enumerate(self.symptom_options):
            button = tk.Button(
                tags,
                text=symptom,
                relief="flat",
                bd=0,
                bg=self.palette["tag_bg"],
                fg=self.palette["primary"],
                activebackground=self.palette["tag_hover"],
                activeforeground="#ffffff",
                cursor="hand2",
                font=("Microsoft YaHei UI", 10),
                padx=14,
                pady=6,
                command=lambda value=symptom: self._append_symptom(value),
            )
            button.grid(row=index // 4, column=index % 4, padx=5, pady=5, sticky="ew")
            tags.grid_columnconfigure(index % 4, weight=1)

        # 操作按钮
        actions = ttk.Frame(left, style="Card.TFrame")
        actions.pack(fill="x", pady=(16, 0))
        self.submit_button = ttk.Button(
            actions, text="提交查询", style="Primary.TButton", command=self._submit_query
        )
        self.submit_button.pack(side="left")
        ttk.Button(actions, text="清空", style="Secondary.TButton", command=self._clear_query).pack(
            side="left", padx=(8, 0)
        )

        # 状态
        status_frame = tk.Frame(left, bg=self.palette["card"])
        status_frame.pack(fill="x", pady=(12, 0))
        self.status_var = tk.StringVar(value="准备就绪")
        ttk.Label(status_frame, textvariable=self.status_var, style="Muted.TLabel").pack(anchor="w")
        self.progress = ttk.Progressbar(left, mode="indeterminate", length=200)
        self.progress.pack(fill="x", pady=(4, 0))

        # Progressbar 颜色
        style = ttk.Style()
        style.configure("green.Horizontal.TProgressbar", background=self.palette["primary"], troughcolor=self.palette["bg"], borderwidth=0)
        self.progress.configure(style="green.Horizontal.TProgressbar")

        # ── 右侧面板 ──

        # 卡片内阴影顶线
        tk.Frame(right, bg=self.palette["primary"], height=2).pack(fill="x", pady=(0, 16))

        ttk.Label(right, text="智能回答", style="SectionTitle.TLabel").pack(anchor="w")
        ttk.Label(right, text="提交症状后将在此显示辨证结果与推荐方案", style="SectionSubtitle.TLabel").pack(anchor="w", pady=(2, 14))

        # 回答内容
        answer_frame = tk.Frame(right, bg=self.palette["border"], bd=1, relief="solid")
        answer_frame.pack(fill="both", expand=True)
        self.answer_text = ScrolledText(
            answer_frame,
            wrap="word",
            height=14,
            font=("Microsoft YaHei UI", 11),
            relief="flat",
            bd=0,
        )
        self.answer_text.pack(fill="both", padx=2, pady=2)
        self.answer_text.configure(
            background=self.palette["input_bg"],
            foreground=self.palette["text"],
            insertbackground=self.palette["primary"],
        )
        self._configure_output_text(self.answer_text)

        # 相关知识
        tk.Frame(right, bg=self.palette["border"], height=1).pack(fill="x", pady=(12, 10))
        ttk.Label(right, text="相关知识", style="Body.TLabel").pack(anchor="w", pady=(0, 6))
        knowledge_frame = tk.Frame(right, bg=self.palette["border"], bd=1, relief="solid")
        knowledge_frame.pack(fill="x")
        self.knowledge_text = ScrolledText(
            knowledge_frame,
            wrap="word",
            height=5,
            font=("Microsoft YaHei UI", 10),
            relief="flat",
            bd=0,
        )
        self.knowledge_text.pack(fill="both", padx=2, pady=2)
        self.knowledge_text.configure(
            background=self.palette["input_bg"],
            foreground=self.palette["muted"],
        )
        self._configure_output_text(self.knowledge_text)

        # ── 免责声明（首页最底部）──
        disclaimer_frame = tk.Frame(
            self.home_tab, bg=self.palette["warning_bg"], highlightthickness=1, highlightbackground=self.palette["warning_border"]
        )
        disclaimer_frame.pack(fill="x", pady=(0, 0), side="bottom")
        tk.Label(
            disclaimer_frame,
            text="⚕ 免责声明",
            bg=self.palette["warning_bg"],
            fg=self.palette["warning_fg"],
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(anchor="w", padx=14, pady=(8, 2))
        self.disclaimer_var = tk.StringVar(
            value="本系统仅提供中医药知识科普和参考信息，不构成医疗诊断或治疗建议。如有健康问题，请咨询专业医疗机构或执业医师。"
        )
        tk.Label(
            disclaimer_frame,
            textvariable=self.disclaimer_var,
            justify="left",
            wraplength=880,
            bg=self.palette["warning_bg"],
            fg=self.palette["warning_fg"],
            font=("Microsoft YaHei UI", 10),
        ).pack(anchor="w", padx=14, pady=(0, 10))

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

        self.history_tree = ttk.Treeview(
            list_frame,
            columns=("timestamp", "query"),
            show="headings",
            selectmode="browse",
        )
        self.history_tree.heading("timestamp", text="时间")
        self.history_tree.heading("query", text="问题")
        self.history_tree.column("timestamp", width=170, anchor="w")
        self.history_tree.column("query", width=510, anchor="w")
        self.history_tree.pack(fill="both", expand=True, pady=(8, 0))
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
        ttk.Label(frame, text="中医药智能问答系统 — TCMAI v2.1.0", style="SectionSubtitle.TLabel").pack(anchor="w", pady=(2, 18))

        sections = [
            (
                "📋 项目简介",
                "中医药智能问答系统用于提供中医药知识科普、症状参考分析和历史记录管理。\n支持快速模式（规则引擎）和 AI 模式（深度学习推荐）两种运行方式。",
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

    def _configure_output_text(self, widget: ScrolledText) -> None:
        widget.configure(
            background="#ffffff",
            foreground=self.palette["text"],
            insertbackground=self.palette["primary"],
            highlightthickness=1,
            highlightbackground=self.palette["border"],
            highlightcolor=self.palette["primary_light"],
            padx=12,
            pady=10,
            state="disabled",
        )

    def _set_text(self, widget: ScrolledText, value: str) -> None:
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
        self._set_text(self.knowledge_text, "")

        thread = threading.Thread(target=self._run_query, args=(query,), daemon=True)
        thread.start()

    def _run_query(self, query: str) -> None:
        try:
            pipeline = SymptomPipeline()
            detected = pipeline.detect(query)
            result = self.service.query(query, detected)
            self.result_queue.put(("success", result))
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
            self.status_var.set("查询失败")
            messagebox.showerror("查询失败", str(payload))
            return

        data: Dict[str, Any] = payload
        self.status_var.set("查询完成")
        self.disclaimer_var.set(
            data.get("disclaimer") or "本系统仅提供中医药知识科普和参考信息，不构成医疗诊断或治疗建议。"
        )
        self._set_text(self.answer_text, data.get("answer", ""))
        knowledge = data.get("knowledge")
        if isinstance(knowledge, dict):
            knowledge = json.dumps(knowledge, ensure_ascii=False, indent=2)
        self._set_text(self.knowledge_text, str(knowledge or "暂无相关知识"))
        self._load_history()

    def _load_history(self) -> None:
        search = self.history_search_var.get().strip().lower() if hasattr(self, "history_search_var") else ""
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)

        for record in self.service.list_history():
            haystack = f"{record.get('query', '')} {record.get('answer', '')}".lower()
            if search and search not in haystack:
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
            "导出内容会保持当前脱敏状态，并包含最近 500 条历史记录。是否继续？",
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
    root.mainloop()
