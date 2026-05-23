"""
TCMAI 启动画面

炫酷的加载界面，在后台初始化时展示品牌形象和加载进度。
"""
import tkinter as tk
import threading
import time
from pathlib import Path
from typing import Optional, Callable


class SplashScreen:
    """启动画面"""

    # 加载阶段定义
    STAGES = [
        ("正在初始化系统...", 0.05),
        ("正在加载核心模块...", 0.12),
        ("正在建立知识库连接...", 0.20),
        ("正在加载知识图谱...", 0.35),
        ("正在启动智能引擎...", 0.55),
        ("正在加载模型参数...", 0.70),
        ("正在优化推理管线...", 0.85),
        ("准备就绪，即将启动...", 0.95),
    ]

    def __init__(self, parent: Optional[tk.Tk] = None):
        """
        Args:
            parent: 如有父窗口，创建为 Toplevel 覆盖层；
                     否则创建独立的 tk.Tk 窗口（向后兼容）。
        """
        if parent is None:
            self.root = tk.Tk()
            self._is_toplevel = False
        else:
            self.root = tk.Toplevel(parent)
            self._is_toplevel = True

        self.root.overrideredirect(True)  # 无标题栏
        self.root.attributes("-topmost", True)

        # 全屏
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{sw}x{sh}+0+0")

        # 颜色
        self.BG_DEEP = "#0d2818"      # 最深绿
        self.BG_MID = "#1a3a2b"       # 中绿
        self.ACCENT = "#c8953e"       # 金色
        self.ACCENT_LIGHT = "#e8c46a" # 亮金
        self.TEXT_LIGHT = "#e8f0ea"   # 浅白绿
        self.TEXT_MUTED = "#8aae96"   # 灰绿

        self._progress = 0.0
        self._message = "正在初始化..."
        self._running = True
        self._on_close: Optional[Callable] = None

        self._build_ui()

    def _build_ui(self):
        """构建全屏启动画面UI"""
        # 全屏主容器（居中内容）
        outer = tk.Frame(self.root, bg=self.BG_DEEP)
        outer.pack(fill="both", expand=True)

        # 内容容器（固定宽度居中）
        content_width = 600
        inner = tk.Frame(outer, bg=self.BG_DEEP, width=content_width)
        inner.place(relx=0.5, rely=0.5, anchor="center", width=content_width)

        # ── 顶部装饰线 ──
        tk.Frame(inner, bg=self.ACCENT, height=4).pack(fill="x", pady=(0, 35))

        # ── 品牌图标（装饰性圆点） ──
        icon_frame = tk.Frame(inner, bg=self.BG_DEEP)
        icon_frame.pack(pady=(0, 16))

        for i, color in enumerate([self.ACCENT, self.ACCENT_LIGHT, self.ACCENT]):
            dot = tk.Canvas(icon_frame, width=18, height=18, bg=self.BG_DEEP,
                            highlightthickness=0)
            dot.create_oval(1, 1, 17, 17, fill=color, outline="")
            dot.pack(side="left", padx=6)

        # ── 标题 ──
        title = tk.Label(
            inner, text="中医药智能问答系统",
            bg=self.BG_DEEP, fg=self.TEXT_LIGHT,
            font=("Microsoft YaHei UI", 32, "bold"),
        )
        title.pack(pady=(14, 8))

        # ── 副标题 ──
        subtitle = tk.Label(
            inner, text="基于深度学习的中医药知识推荐",
            bg=self.BG_DEEP, fg=self.TEXT_MUTED,
            font=("Microsoft YaHei UI", 14, "bold"),
        )
        subtitle.pack(pady=(0, 24))

        # ── 加载消息 ──
        self.msg_var = tk.StringVar(value=self._message)
        msg = tk.Label(
            inner, textvariable=self.msg_var,
            bg=self.BG_DEEP, fg=self.ACCENT,
            font=("Microsoft YaHei UI", 18, "bold"),
        )
        msg.pack(pady=(0, 20))

        # ── 进度条 ──
        progress_container = tk.Frame(inner, bg=self.BG_MID, height=10,
                                       highlightthickness=0)
        progress_container.pack(fill="x", padx=40)
        progress_container.pack_propagate(False)

        self.progress_bar = tk.Frame(progress_container, bg=self.ACCENT,
                                      height=10, highlightthickness=0)
        self.progress_bar.place(x=0, y=0, width=0, height=10)

        # ─️ 底部文字 ──
        bottom = tk.Label(
            inner, text="基于 TCMNet Transformer 架构",
            bg=self.BG_DEEP, fg=self.TEXT_MUTED,
            font=("Microsoft YaHei UI", 12, "bold"),
        )
        bottom.pack(pady=(24, 0))

        # ── 底部装饰线 ──
        tk.Frame(inner, bg=self.ACCENT, height=2).pack(fill="x", pady=(20, 0))

        self.root.update_idletasks()

    # ═══════════════════════════════════════════════
    # 公共接口
    # ═══════════════════════════════════════════════

    def update_progress(self, value: float, message: str = ""):
        """更新进度（0~1）和消息"""
        self._progress = max(0.0, min(1.0, value))
        if message:
            self._message = message
            self.msg_var.set(message)
        # 更新进度条宽度
        w = int(520 * self._progress)  # 进度条容器宽度
        self.progress_bar.place(width=max(4, w))
        self.root.update_idletasks()

    def set_stage(self, index: int):
        """设置到预定义的加载阶段"""
        if 0 <= index < len(self.STAGES):
            msg, prog = self.STAGES[index]
            self.update_progress(prog, msg)

    def auto_progress(self, duration: float = 3.0, callback: Optional[Callable] = None):
        """自动推进进度（用于模拟加载）"""
        def _animate():
            steps = 30
            for i in range(steps):
                if not self._running:
                    break
                t = i / steps
                # 从当前进度到0.95
                current = self._progress
                target = 0.95
                progress = current + (target - current) * (i / steps)
                # 匹配最近的阶段消息
                for stage_idx, (msg, stage_prog) in enumerate(self.STAGES):
                    if progress <= stage_prog:
                        self.update_progress(progress, msg)
                        break
                else:
                    self.update_progress(progress, "准备就绪，即将启动...")
                time.sleep(duration / steps)
            self.update_progress(1.0, "启动完成！")
            if callback:
                self.root.after(0, callback)
        threading.Thread(target=_animate, daemon=True).start()

    def close(self):
        """关闭启动画面"""
        self._running = False
        self.root.destroy()

    def show_and_wait(self, work_function: Callable, duration: float = 4.0):
        """
        在工作函数执行期间显示启动画面

        Args:
            work_function: 后台执行的初始化函数
            duration: 最小显示时间（秒）
        """
        result = [None]
        error = [None]

        def _worker():
            try:
                # 先跑一段自动进度
                for i in range(len(self.STAGES)):
                    if not self._running:
                        return
                    self.set_stage(i)
                    time.sleep(duration / len(self.STAGES) * 0.3)

                # 执行实际工作
                self.update_progress(0.7, "正在完成初始化...")
                work_function()

                self.update_progress(1.0, "启动完成！")
                time.sleep(0.5)
            except Exception as e:
                error[0] = e
            finally:
                if error[0] is None:
                    self.root.after(0, self.close)

        threading.Thread(target=_worker, daemon=True).start()
        self.root.mainloop()

        if error[0]:
            raise error[0]

        return result[0]
