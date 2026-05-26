"""
TCMAI 启动动画 — 深色背景缩放隐去，系统名称淡出
"""

import logging

import customtkinter as ctk

logger = logging.getLogger(__name__)


def _lerp_color(c1: str, c2: str, t: float) -> str:
    """线性插值两个十六进制颜色"""
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


class SplashScreen:
    """启动动画：全屏深色 → 3s 两段式淡出 → 主窗口露出"""

    BG_DEEP = "#0d2818"
    ACCENT = "#c8953e"
    STEPS = 60

    # 两段式动画：文字先消失，背景再淡化
    PHASE1_END = 0.55  # 前 55%：文字融入背景

    def __init__(self, parent: ctk.CTk | None = None):
        if parent is None:
            self.root = ctk.CTk()
        else:
            self.root = ctk.CTkToplevel(parent)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{sw}x{sh}+0+0")
        self.root.configure(fg_color=self.BG_DEEP)

        outer = ctk.CTkFrame(self.root, fg_color=self.BG_DEEP)
        outer.pack(fill="both", expand=True)

        self._title_label = ctk.CTkLabel(
            outer,
            text="TCMAI",
            font=("Microsoft YaHei UI", 64, "bold"),
            text_color=self.ACCENT,
        )
        self._title_label.place(relx=0.5, rely=0.45, anchor="center")

        self._sub_label = ctk.CTkLabel(
            outer,
            text="中医药智能方剂生成系统",
            font=("Microsoft YaHei UI", 18),
            text_color="#ffffff",
        )
        self._sub_label.place(relx=0.5, rely=0.55, anchor="center")

        self.root.update_idletasks()
        self._current_step = self.STEPS

    def animate_and_close(self, duration: float = 3.0):
        """两段式动画：文字融入背景 → 窗口淡出"""
        self._step_delay = max(16, int((duration / self.STEPS) * 1000))
        self._current_step = self.STEPS
        self._fade_step()

    def _fade_step(self):
        if not self.root.winfo_exists():
            return
        self._current_step -= 1
        if self._current_step <= 0:
            self.close()
            return

        progress = 1.0 - (self._current_step / self.STEPS)

        if progress < self.PHASE1_END:
            # ── 阶段一：文字融入背景 ──
            t = progress / self.PHASE1_END  # 0→1 within phase 1
            # 标题：金色 → 背景色
            title_color = _lerp_color(self.ACCENT, self.BG_DEEP, t)
            # 副标题：白色 → 背景色
            sub_color = _lerp_color("#ffffff", self.BG_DEEP, t)
            try:
                self._title_label.configure(text_color=title_color)
                self._sub_label.configure(text_color=sub_color)
            except Exception:
                pass
            # 标题微微上移
            self._title_label.place(relx=0.5, rely=0.45 - t * 40 / 600, anchor="center")
        else:
            # ── 阶段二：窗口透明度淡出 ──
            t = (progress - self.PHASE1_END) / (1.0 - self.PHASE1_END)
            alpha = 1.0 - t
            try:
                self.root.attributes("-alpha", max(alpha, 0.01))
            except Exception:
                pass

        self.root.after(self._step_delay, self._fade_step)

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass
