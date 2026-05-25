"""
TCMAI 启动动画 — 深色背景缩放隐去，系统名称淡出
"""

import logging
import tkinter as tk
import threading
import time
from typing import Optional


logger = logging.getLogger(__name__)


class SplashScreen:
    """启动动画：全屏深色 → 1.2s 缩放消失 → 主窗口露出"""

    BG_DEEP = "#0d2818"
    ACCENT = "#c8953e"

    def __init__(self, parent: Optional[tk.Tk] = None):
        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{sw}x{sh}+0+0")
        self.root.configure(bg=self.BG_DEEP)

        outer = tk.Frame(self.root, bg=self.BG_DEEP)
        outer.pack(fill="both", expand=True)

        tk.Label(
            outer, text="TCMAI",
            bg=self.BG_DEEP, fg=self.ACCENT,
            font=("Microsoft YaHei UI", 64, "bold"),
        ).place(relx=0.5, rely=0.45, anchor="center")

        tk.Label(
            outer, text="中医药智能方剂生成系统",
            bg=self.BG_DEEP, fg="#ffffff",
            font=("Microsoft YaHei UI", 18),
        ).place(relx=0.5, rely=0.55, anchor="center")

        self.root.update_idletasks()

    def animate_and_close(self, duration: float = 1.2):
        """执行缩放动画后自动关闭"""
        def _anim():
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            steps = 30
            for i in range(steps + 1):
                if not self.root.winfo_exists():
                    return
                t = i / steps
                eased = t * t * (3 - 2 * t)
                scale = 1.0 - eased * 0.85
                try:
                    self.root.geometry(
                        f"{int(sw*scale)}x{int(sh*scale)}+{(sw-int(sw*scale))//2}+{(sh-int(sh*scale))//2}"
                    )
                except Exception:
                    pass
                time.sleep(duration / steps)
            self.close()

        threading.Thread(target=_anim, daemon=True).start()

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass
