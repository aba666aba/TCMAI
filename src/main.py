import logging
import os
import sys


TCMAI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TCMAI_ROOT, "src"))
os.environ["TCMAI_ROOT"] = TCMAI_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
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
