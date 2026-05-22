import logging
import logging.handlers
import os
import sys
from pathlib import Path


if getattr(sys, "frozen", False):
    ROOT_DIR = Path(sys.executable).parent
    internal_path = ROOT_DIR / "_internal"
    if internal_path.exists():
        sys.path.insert(0, str(internal_path))
        sys.path.insert(0, str(internal_path / "src"))
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
        from desktop.runtime import hide_console_window, show_error_dialog
        from desktop.app import run_desktop_app

        hide_console_window()
        _logger.info("中医药智能问答系统启动")
        run_desktop_app()
    except Exception as exc:
        _logger.exception("程序运行失败")
        try:
            show_error_dialog("中医药智能问答系统", f"程序运行失败：\n{exc}")
        except Exception:
            pass


if __name__ == "__main__":
    main()

