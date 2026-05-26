import ctypes
import logging
import sys
from ctypes import wintypes

logger = logging.getLogger(__name__)


def hide_console_window() -> None:
    if sys.platform != "win32":
        return

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32.GetConsoleWindow.restype = wintypes.HWND
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        hwnd = kernel32.GetConsoleWindow()
        if hwnd:
            user32.ShowWindow(hwnd, 0)
    except Exception as exc:
        logger.warning("隐藏控制台窗口失败: %s", exc)


def show_error_dialog(title: str, message: str) -> None:
    if sys.platform != "win32":
        return

    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.MessageBoxW(None, message, title, 0x10)
    except Exception as exc:
        logger.warning("显示错误对话框失败: %s", exc)
