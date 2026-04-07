import json
import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Dict, List, Optional

from core.fallback import detect_symptoms
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
            "bg": "#f3f1ea",
            "card": "#fcfbf7",
            "deep": "#1f4f3c",
            "primary": "#2d6a4f",
            "primary_light": "#52b788",
            "text": "#24352d",
            "muted": "#6b7a72",
            "warning": "#fff3d7",
            "border": "#d7ded8",
        }
        self.symptom_options = ["乏力", "食欲不振", "失眠", "头痛", "咳嗽", "腹痛", "腹泻", "便秘"]

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
        style.configure(
            "Title.TLabel",
            background=self.palette["deep"],
            foreground="#ffffff",
            font=("Microsoft YaHei UI", 18, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.palette["deep"],
            foreground="#d8efe4",
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "SectionTitle.TLabel",
            background=self.palette["card"],
            foreground=self.palette["primary"],
            font=("Microsoft YaHei UI", 14, "bold"),
        )
        style.configure("Body.TLabel", background=self.palette["card"], foreground=self.palette["text"])
        style.configure("Muted.TLabel", background=self.palette["card"], foreground=self.palette["muted"])
        style.configure(
            "Primary.TButton",
            background=self.palette["primary"],
            foreground="#ffffff",
            padding=(18, 10),
            borderwidth=0,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", self.palette["deep"]), ("disabled", "#96b8ab")],
        )
        style.configure(
            "Secondary.TButton",
            background="#e8efe9",
            foreground=self.palette["primary"],
            padding=(12, 8),
            borderwidth=0,
        )
        style.map("Secondary.TButton", background=[("active", "#dce8e0")])
        style.configure(
            "Treeview",
            font=("Microsoft YaHei UI", 10),
            rowheight=34,
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

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, style="App.TFrame", padding=20)
        container.pack(fill="both", expand=True)

        hero = ttk.Frame(container, style="Hero.TFrame", padding=(24, 18))
        hero.pack(fill="x")
        ttk.Label(hero, text="中医药智能问答系统", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            hero,
            text="独立桌面软件界面，支持问答、历史记录和本地导出。",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(6, 0))

        notebook = ttk.Notebook(container)
        notebook.pack(fill="both", expand=True, pady=(16, 0))

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

        ttk.Label(left, text="请输入您的问题或症状", style="SectionTitle.TLabel").pack(anchor="w")
        ttk.Label(left, text="支持模式选择、症状快捷标签和结果展示。", style="Muted.TLabel").pack(anchor="w", pady=(6, 18))

        self.mode_var = tk.StringVar(value="fast")
        ttk.Label(left, text="回答模式", style="Body.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Radiobutton(left, text="快速模式", value="fast", variable=self.mode_var).pack(anchor="w", pady=4)
        ttk.Radiobutton(left, text="AI 模式（当前环境会自动降级）", value="ai", variable=self.mode_var).pack(anchor="w", pady=4)

        ttk.Label(left, text="问题输入", style="Body.TLabel").pack(anchor="w", pady=(16, 8))
        self.query_input = ScrolledText(
            left,
            height=10,
            wrap="word",
            font=("Microsoft YaHei UI", 11),
            relief="flat",
            bd=1,
        )
        self.query_input.pack(fill="x")
        self.query_input.configure(
            background="#ffffff",
            foreground=self.palette["text"],
            insertbackground=self.palette["text"],
            highlightthickness=1,
            highlightbackground=self.palette["border"],
            highlightcolor=self.palette["primary_light"],
        )

        ttk.Label(left, text="常见症状", style="Body.TLabel").pack(anchor="w", pady=(16, 8))
        tags = ttk.Frame(left, style="Card.TFrame")
        tags.pack(fill="x")
        for index, symptom in enumerate(self.symptom_options):
            button = tk.Button(
                tags,
                text=symptom,
                relief="flat",
                bd=0,
                bg="#edf5f0",
                fg=self.palette["primary"],
                activebackground=self.palette["primary"],
                activeforeground="#ffffff",
                cursor="hand2",
                font=("Microsoft YaHei UI", 10, "bold"),
                padx=12,
                pady=6,
                command=lambda value=symptom: self._append_symptom(value),
            )
            button.grid(row=index // 4, column=index % 4, padx=6, pady=6, sticky="ew")
            tags.grid_columnconfigure(index % 4, weight=1)

        actions = ttk.Frame(left, style="Card.TFrame")
        actions.pack(fill="x", pady=(18, 0))
        self.submit_button = ttk.Button(
            actions, text="提交查询", style="Primary.TButton", command=self._submit_query
        )
        self.submit_button.pack(side="left")
        ttk.Button(actions, text="清空输入", style="Secondary.TButton", command=self._clear_query).pack(
            side="left", padx=(10, 0)
        )

        self.status_var = tk.StringVar(value="准备就绪")
        ttk.Label(left, textvariable=self.status_var, style="Muted.TLabel").pack(anchor="w", pady=(12, 6))
        self.progress = ttk.Progressbar(left, mode="indeterminate")
        self.progress.pack(fill="x")

        ttk.Label(right, text="智能回答", style="SectionTitle.TLabel").pack(anchor="w")
        self.result_mode_var = tk.StringVar(value="等待查询")
        ttk.Label(right, textvariable=self.result_mode_var, style="Muted.TLabel").pack(anchor="w", pady=(4, 12))

        ttk.Label(right, text="回答内容", style="Body.TLabel").pack(anchor="w")
        self.answer_text = ScrolledText(
            right,
            wrap="word",
            height=16,
            font=("Microsoft YaHei UI", 11),
            relief="flat",
            bd=1,
        )
        self.answer_text.pack(fill="both", expand=True)
        self._configure_output_text(self.answer_text)

        ttk.Label(right, text="相关知识", style="Body.TLabel").pack(anchor="w", pady=(14, 6))
        self.knowledge_text = ScrolledText(
            right,
            wrap="word",
            height=7,
            font=("Microsoft YaHei UI", 10),
            relief="flat",
            bd=1,
        )
        self.knowledge_text.pack(fill="x")
        self._configure_output_text(self.knowledge_text)

        disclaimer_frame = tk.Frame(
            right, bg=self.palette["warning"], highlightthickness=1, highlightbackground="#efc463"
        )
        disclaimer_frame.pack(fill="x", pady=(14, 0))
        tk.Label(
            disclaimer_frame,
            text="免责声明",
            bg=self.palette["warning"],
            fg="#7a5a16",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(anchor="w", padx=12, pady=(10, 2))
        self.disclaimer_var = tk.StringVar(
            value="本系统仅提供中医药知识科普和参考信息，不构成医疗诊断或治疗建议。"
        )
        tk.Label(
            disclaimer_frame,
            textvariable=self.disclaimer_var,
            justify="left",
            wraplength=520,
            bg=self.palette["warning"],
            fg="#7a5a16",
            font=("Microsoft YaHei UI", 10),
        ).pack(anchor="w", padx=12, pady=(0, 10))

    def _build_history_tab(self) -> None:
        wrapper = ttk.Frame(self.history_tab, style="App.TFrame")
        wrapper.pack(fill="both", expand=True)

        toolbar = ttk.Frame(wrapper, style="Card.TFrame", padding=16)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="查询历史", style="SectionTitle.TLabel").pack(side="left")
        self.history_search_var = tk.StringVar()
        entry = ttk.Entry(toolbar, textvariable=self.history_search_var, width=36)
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

        self.history_tree = ttk.Treeview(
            list_frame,
            columns=("timestamp", "mode", "query"),
            show="headings",
            selectmode="browse",
        )
        self.history_tree.heading("timestamp", text="时间")
        self.history_tree.heading("mode", text="模式")
        self.history_tree.heading("query", text="问题")
        self.history_tree.column("timestamp", width=170, anchor="w")
        self.history_tree.column("mode", width=90, anchor="center")
        self.history_tree.column("query", width=420, anchor="w")
        self.history_tree.pack(fill="both", expand=True)
        self.history_tree.bind("<<TreeviewSelect>>", self._on_history_select)

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
            bd=1,
        )
        self.history_detail.pack(fill="both", expand=True, pady=(12, 0))
        self._configure_output_text(self.history_detail)
        self._set_text(self.history_detail, "选择左侧一条记录后，这里会显示完整问题、回答、症状和时间。")

    def _build_about_tab(self) -> None:
        frame = ttk.Frame(self.about_tab, style="Card.TFrame", padding=24)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="关于系统", style="SectionTitle.TLabel").pack(anchor="w")
        sections = [
            (
                "项目简介",
                "中医药智能问答系统用于提供中医药知识科普、症状参考分析和历史记录管理。",
            ),
            (
                "主要功能",
                "1. 输入问题并进行智能问答。\n2. 通过症状快捷标签快速补全描述。\n3. 在软件内查看、搜索、导出和清理历史记录。",
            ),
            (
                "使用提醒",
                "本系统仅提供知识参考，不构成医疗诊断或治疗建议。如有持续不适，请及时咨询专业医疗机构或执业医师。",
            ),
            ("开发团队", "李俊贤、王子璇、周浩宇、刘丁夺"),
        ]
        for title, body in sections:
            card = tk.Frame(frame, bg="#f7faf7", highlightthickness=1, highlightbackground=self.palette["border"])
            card.pack(fill="x", pady=8)
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
            insertbackground=self.palette["text"],
            highlightthickness=1,
            highlightbackground=self.palette["border"],
            highlightcolor=self.palette["primary_light"],
            padx=10,
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
        self.result_mode_var.set("处理中")
        self._set_text(self.answer_text, "")
        self._set_text(self.knowledge_text, "")

        thread = threading.Thread(target=self._run_query, args=(query, self.mode_var.get()), daemon=True)
        thread.start()

    def _run_query(self, query: str, mode: str) -> None:
        try:
            result = self.service.query(query, detect_symptoms(query), mode)
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
            self.result_mode_var.set("出现错误")
            messagebox.showerror("查询失败", str(payload))
            return

        data: Dict[str, Any] = payload
        self.status_var.set("查询完成")
        self.result_mode_var.set(
            f"当前结果：{'AI 模式' if data.get('mode') == 'ai' else '快速模式'}"
        )
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
            mode = "AI" if record.get("mode") == "ai" else "快速"
            self.history_tree.insert(
                "",
                "end",
                iid=str(record["id"]),
                values=(record.get("timestamp", ""), mode, preview),
            )

    def _on_history_select(self, _event: Any) -> None:
        selection = self.history_tree.selection()
        if not selection:
            return
        self.selected_history_id = int(selection[0])
        record = next(
            (item for item in self.service.history_data if item.get("id") == self.selected_history_id),
            None,
        )
        if not record:
            return
        symptoms = "、".join(record.get("symptoms", [])) or "未识别"
        detail = (
            f"时间：{record.get('timestamp', '')}\n"
            f"模式：{'AI 模式' if record.get('mode') == 'ai' else '快速模式'}\n"
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
                        f"模式：{item.get('mode', '')}\n"
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
