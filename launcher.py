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


def main() -> None:
    try:
        from desktop.runtime import hide_console_window, show_error_dialog
        from desktop.app import run_desktop_app

        hide_console_window()
        run_desktop_app()
    except Exception as exc:
        try:
            show_error_dialog("中医药智能问答系统", f"程序运行失败：\n{exc}")
        except Exception:
            pass


if __name__ == "__main__":
    main()

