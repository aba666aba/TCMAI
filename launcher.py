import logging
import logging.handlers
import os
import queue
import sys
import threading
import time
from pathlib import Path


if getattr(sys, "frozen", False):
    ROOT_DIR = Path(sys.executable).parent
    # PyInstaller 依赖包目录
    internal_path = ROOT_DIR / "_internal"
    if internal_path.exists():
        sys.path.insert(0, str(internal_path))
    # 项目源码目录（打包时放置在根目录）
    src_path = ROOT_DIR / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
else:
    ROOT_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(ROOT_DIR / "src"))

os.environ["TCMAI_ROOT"] = str(ROOT_DIR)

# 日志配置
_log_level = getattr(logging, os.environ.get("TCMAI_LOG_LEVEL", "INFO").upper(), logging.INFO)
_log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_log_dir = ROOT_DIR / "data" / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)
_file_handler = logging.handlers.RotatingFileHandler(
    _log_dir / "tcmai.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_file_handler.setFormatter(_log_format)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_log_format)
logging.basicConfig(level=_log_level, handlers=[_file_handler, _console_handler])
_logger = logging.getLogger("launcher")


def main() -> None:
    try:
        from desktop.splash import SplashScreen
        from desktop.runtime import hide_console_window, show_error_dialog
        import tkinter as tk

        hide_console_window()
        _logger.info("中医药智能问答系统启动")

        # ── 先创建主窗口（隐藏），再盖启动动画 ──
        from desktop.app import DesktopApp
        root = tk.Tk()
        root.withdraw()
        app = DesktopApp(root)
        root.state("zoomed")

        # ── 启动动画（覆盖在主窗口之上）──
        splash = SplashScreen(parent=root)
        splash.animate_and_close(duration=1.2)

        # ── 进入主循环 ──
        root.deiconify()
        root.mainloop()

    except Exception as exc:
        _logger.exception("程序运行失败")
        try:
            show_error_dialog("中医药智能问答系统", f"程序运行失败：\n{exc}")
        except Exception:
            pass


# ──────────────────────────────────────────────
# _BgLoader 暂注释——跳过启动画面后不再使用
# 保留代码以备后续恢复
# ──────────────────────────────────────────────
# class _BgLoader:
#     MIN_DISPLAY = 9.0
#     def __init__(self, splash, logger): ...
#     def start(self): ...
#     def _bg_work(self): ...
#     def _send_stage(self, index): ...
#     def _send(self, message): ...
#     def _poll_progress(self): ...

