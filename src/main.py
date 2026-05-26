import logging
import logging.handlers
import os
import sys
from pathlib import Path

TCMAI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TCMAI_ROOT, "src"))
os.environ["TCMAI_ROOT"] = TCMAI_ROOT

_log_level = getattr(logging, os.environ.get("TCMAI_LOG_LEVEL", "INFO").upper(), logging.INFO)
_log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# stdout
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_log_format)

# file (rotating: 5MB x 3 backups)
_log_dir = Path(TCMAI_ROOT) / "data" / "logs"
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
        from desktop.app import run_desktop_app
        from desktop.runtime import hide_console_window, show_error_dialog

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
