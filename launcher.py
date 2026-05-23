import logging
import logging.handlers
import os
import queue
import sys
import threading
import time
from pathlib import Path


if getattr(sys, "frozen", False):
    ROOT_DIR = Path(sys.executable).parent
    # PyInstaller 依赖包目录
    internal_path = ROOT_DIR / "_internal"
    if internal_path.exists():
        sys.path.insert(0, str(internal_path))
    # 项目源码目录（打包时放置在根目录）
    src_path = ROOT_DIR / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
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
        from desktop.splash import SplashScreen
        from desktop.runtime import hide_console_window, show_error_dialog
        import tkinter as tk

        hide_console_window()
        _logger.info("中医药智能问答系统启动")

        # ── 先创建主窗口（初始隐藏，避免闪烁）──
        from desktop.app import DesktopApp
        root = tk.Tk()
        root.withdraw()
        app = DesktopApp(root)
        root.state("zoomed")  # 最大化（尚未显示）

        # ── 创建全屏置顶启动画面（覆盖在主窗口之上）──
        splash = SplashScreen(parent=root)
        root.deiconify()  # 显示主窗口（被 splash 完全覆盖）

        # ── 后台加载知识图谱和推理管线 ──
        _BgLoader(splash, _logger).start()

        # ── 进入主循环（splash 和主窗口共享同一个事件循环）──
        # 后台加载完成后 splash 自动关闭，露出主窗口
        root.mainloop()

    except Exception as exc:
        _logger.exception("程序运行失败")
        try:
            show_error_dialog("中医药智能问答系统", f"程序运行失败：\n{exc}")
        except Exception:
            pass


class _BgLoader:
    """
    Splash 期间在后台线程预加载知识图谱。
    通过队列将进度推送到主线程，由 after() 轮询更新 Splash UI。
    """

    MIN_DISPLAY = 9.0  # Splash 最少显示秒数

    def __init__(self, splash, logger):
        self.splash = splash
        self.log = logger
        self._progress_q = queue.Queue()  # (stage_index_or_None, progress, message)
        self._ready = False
        self._start_ts = time.time()

    def start(self):
        """启动后台加载线程和主线程轮询"""
        threading.Thread(target=self._bg_work, daemon=True).start()
        self.splash.root.after(200, self._poll_progress)

    # ── 后台线程 ──

    def _bg_work(self):
        """后台线程：执行真实初始化"""
        try:
            self._send_stage(2)  # "正在建立知识库连接..."
            time.sleep(1.5)

            self._send_stage(3)  # "正在加载知识图谱..."
            self.log.info("后台预加载知识图谱...")
            from desktop.kg_view import get_kg
            get_kg()
            self.log.info("知识图谱预加载完成")

            self._send_stage(4)  # "正在启动智能引擎..."
            time.sleep(1.5)

            self._send(0.65, "正在初始化推理管线...")
            self.log.info("后台预加载推理管线...")
            try:
                from tcm_recommender.data_pipeline import TCMDataPipeline
                _pipeline = TCMDataPipeline()
                _pipeline.load_all()
                self.log.info("推理管线预加载完成")
            except Exception as exc:
                self.log.warning("推理管线预加载失败(不影响启动): %s", exc)

            self._send(0.85, "准备就绪...")
            time.sleep(1.0)
        except Exception as exc:
            self.log.warning("后台初始化异常(不影响启动): %s", exc)
        finally:
            self._ready = True

    def _send_stage(self, index):
        self._progress_q.put(("stage", index, None))

    def _send(self, progress, message):
        self._progress_q.put(("progress", progress, message))

    # ── 主线程轮询 ──

    def _poll_progress(self):
        """主线程执行：消费进度队列，更新 Splash UI"""
        try:
            while True:
                kind, a, b = self._progress_q.get_nowait()
                if kind == "stage":
                    self.splash.set_stage(a)
                elif kind == "progress":
                    self.splash.update_progress(a, b)
        except queue.Empty:
            pass

        # 检查是否完成
        elapsed = time.time() - self._start_ts
        if self._ready and elapsed >= self.MIN_DISPLAY:
            self.log.info("后台初始化完成，启动主窗口")
            self.splash.close()
            return

        # 继续轮询
        self.splash.root.after(200, self._poll_progress)

