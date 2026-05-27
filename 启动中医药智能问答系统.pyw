# -*- coding: utf-8 -*-
"""
中医药智能问答系统 — 双击启动入口

使用前请先运行 setup.bat 完成环境配置。
"""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import logging
import logging.handlers
import os


# ── Tcl/Tk 运行时路径（便携 Python embeddable 版不含） ──
_tcl_dir = ROOT / "python" / "tcl"
if _tcl_dir.exists():
    os.environ.setdefault("TCL_LIBRARY", str(_tcl_dir / "tcl8.6"))
    os.environ.setdefault("TK_LIBRARY", str(_tcl_dir / "tk8.6"))

_log_level = getattr(
    logging, os.environ.get("TCMAI_LOG_LEVEL", "INFO").upper(), logging.INFO
)
_log_format = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_log_format)
_log_dir = ROOT / "data" / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)
_file_handler = logging.handlers.RotatingFileHandler(
    _log_dir / "tcmai.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_file_handler.setFormatter(_log_format)
logging.basicConfig(
    level=_log_level,
    handlers=[_console_handler, _file_handler],
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        from desktop.runtime import hide_console_window, show_error_dialog
        from desktop.app import run_desktop_app

        hide_console_window()
        run_desktop_app()
    except Exception as exc:
        logger.error("启动异常：%s", exc, exc_info=True)
        try:
            show_error_dialog(
                "中医药智能问答系统",
                "系统启动失败：\n"
                + str(exc)
                + "\n\n可能原因：\n1. 未运行 setup.bat 配置环境\n2. Python 运行环境不完整\n3. tkinter 组件不可用\n4. 依赖未安装",
            )
        except Exception:
            pass
