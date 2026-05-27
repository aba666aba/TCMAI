import logging
import logging.handlers
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

# ── 日志配置 ──
_log_root = ROOT_DIR
_log_level = getattr(logging, os.environ.get("TCMAI_LOG_LEVEL", "INFO").upper(), logging.INFO)
_log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_log_dir = _log_root / "data" / "logs"
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
        import customtkinter as ctk

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        ctk.set_widget_scaling(1.25)

        from desktop.splash import SplashScreen
        from desktop.runtime import hide_console_window, show_error_dialog

        hide_console_window()
        _logger.info("中医药智能问答系统启动")

        # ── 先显示启动画面，再构建主界面 ──
        from desktop.app import DesktopApp
        root = ctk.CTk()
        root.state("zoomed")
        splash = SplashScreen(parent=root)
        root.update()

        app = DesktopApp(root)
        root.update()

        splash.animate_and_close(duration=3.0)
        root.mainloop()

    except Exception as exc:
        _logger.exception("程序运行失败")
        try:
            show_error_dialog("中医药智能问答系统", f"程序运行失败：\n{exc}")
        except Exception:
            pass


if __name__ == "__main__":
    main()