import logging
import logging.handlers
import os
import subprocess
import sys
from pathlib import Path


if getattr(sys, "frozen", False):
    APP_ROOT = Path(sys.executable).parent
    internal_path = APP_ROOT / "_internal"
    if internal_path.exists():
        sys.path.insert(0, str(internal_path))
    # PyInstaller onedir 模式下，所有 datas 在 _internal/ 下
    ROOT_DIR = internal_path
else:
    APP_ROOT = ROOT_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(ROOT_DIR / "src"))

os.environ["TCMAI_ROOT"] = str(ROOT_DIR)


def _fix_venv_home() -> bool:
    """便携版自动修复 pyvenv.cfg 的 home 路径。

    首次从 U 盘拷到任意位置运行时，pyvenv.cfg 的 home 路径仍是旧的绝对路径。
    此函数检查并修正 home 为当前目录下的 python/。

    Returns:
        True 表示已修改 pyvenv.cfg，调用方应重启脚本使新路径生效。
    """
    if getattr(sys, "frozen", False):
        return False  # PyInstaller 打包版无需修正
    venv_cfg = ROOT_DIR / "venv" / "pyvenv.cfg"
    python_dir = ROOT_DIR / "python"
    if not python_dir.exists():
        return False  # 没有 python/ 目录就不修正（可能是纯开发环境）
    if not venv_cfg.exists():
        return False  # 没有 venv 也不处理
    try:
        expected_home = str(python_dir.resolve())
        lines = venv_cfg.read_text("utf-8").splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.startswith("home = "):
                current = line[6:].strip()
                if current == expected_home:
                    return False  # 路径已正确
                lines[i] = f"home = {expected_home}\n"
                break
        venv_cfg.write_text("".join(lines), "utf-8")
        return True
    except OSError:
        return False  # 只读/权限不足时静默跳过


if _fix_venv_home():
    # pyvenv.cfg 已修正，重启 Python 进程使新 home 生效
    subprocess.run([sys.executable, __file__, *sys.argv[1:]])
    sys.exit(0)


# 日志配置 — 使用 APP_ROOT（frozen）或 ROOT_DIR（开发）
_log_root = APP_ROOT if getattr(sys, "frozen", False) else ROOT_DIR
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