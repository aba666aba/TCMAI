# Copyright (C) 2026 aba666aba
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# 本文件是中医药智能问答系统 (TCMAI) 的一部分。
#
# 本程序是自由软件：你可以依据自由软件基金会发布的 GNU Affero 通用公共
# 许可证条款（第 3 版或你选择的任何更新版本）重新分发和/或修改本程序。
#
# 本程序按“希望有用”的原则分发，但不提供任何担保，甚至不提供适销性或
# 特定用途适用性的默示担保。详见 GNU Affero 通用公共许可证。
#
# 你应已随本程序收到 GNU Affero 通用公共许可证副本。如未收到，
# 请见 <https://www.gnu.org/licenses/>。
#
# 注意：以网络服务方式提供本程序的服务端修改版本，同样必须按 AGPL-3.0
# 第 13 条向所有用户提供对应源代码。
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
