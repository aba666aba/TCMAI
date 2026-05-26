# -*- coding: utf-8 -*-
"""
中医药智能问答系统 — 双击启动入口

自动使用项目 venv 虚拟环境运行。
"""
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_PYTHONW = ROOT / "venv" / "Scripts" / "pythonw.exe"

# 如果当前不是 venv 环境，自动重启动
_expected = ROOT / "venv" / "Scripts"
_current = Path(sys.executable).parent
if _current.resolve() != _expected.resolve():
    if VENV_PYTHONW.exists():
        script = ROOT / "launcher.py"
        subprocess.Popen([str(VENV_PYTHONW), str(script)])
        sys.exit(0)

# ── 以下仅在 venv 环境下执行 ──
import logging
import logging.handlers
import os

sys.path.insert(0, str(ROOT / "src"))
os.environ["TCMAI_ROOT"] = str(ROOT)

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
                + "\n\n可能原因：\n1. Python 运行环境不完整\n2. tkinter 组件不可用\n3. 依赖未安装",
            )
        except Exception:
            pass
