#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中医药智能问答系统 -- 自解压便携启动器

此脚本被 PyInstaller --onefile 编译为一个 stub.exe。
项目 ZIP 包被附加在 stub.exe 末尾，含 8 字节脚标记录 ZIP 大小。

运行方式：
  首次运行 → 自动解压到 ./TCMAI/ 目录 → 启动程序
  再次运行 → 检测已解压 → 直接启动程序
"""

import os
import struct
import subprocess
import sys
import zipfile
import tempfile
from pathlib import Path


# ── 路径配置 ──
if getattr(sys, "frozen", False):
    EXE_DIR = Path(sys.executable).parent
else:
    EXE_DIR = Path(__file__).resolve().parent.parent

TARGET_DIR = EXE_DIR / "TCMAI"
ZIP_FOOTER_SIZE = 8  # 脚标：unsigned long long (8 bytes)


# ── 核心函数 ──

def get_zip_size(exe_path: Path) -> int:
    """读取 exe 末尾 8 字节的 ZIP 大小脚标"""
    with open(exe_path, "rb") as f:
        f.seek(-ZIP_FOOTER_SIZE, os.SEEK_END)
        (size,) = struct.unpack("<Q", f.read(ZIP_FOOTER_SIZE))
    return size


def extract_zip(exe_path: Path, target: Path, zip_size: int):
    """将嵌入 exe 的 ZIP 解压到目标目录"""
    target.mkdir(parents=True, exist_ok=True)

    # 把 ZIP 部分流式写入临时文件再解压（避免 1GB+ 内存占用）
    tmp_zip = target.parent / "_tmp_bundle.zip"
    try:
        with open(exe_path, "rb") as f:
            f.seek(-zip_size - ZIP_FOOTER_SIZE, os.SEEK_END)
            with open(tmp_zip, "wb") as out:
                remaining = zip_size
                while remaining > 0:
                    chunk_size = min(remaining, 8 * 1024 * 1024)  # 8 MB 块
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    remaining -= len(chunk)

        if not tmp_zip.exists() or tmp_zip.stat().st_size == 0:
            raise RuntimeError("ZIP 临时文件写入失败")

        # 解压
        with zipfile.ZipFile(str(tmp_zip), "r") as zf:
            total = len(zf.infolist())
            for i, entry in enumerate(zf.infolist()):
                zf.extract(entry, target)
                # 每 10% 或在最后显示进度
                if total > 100 and (i + 1) % (total // 10) == 0:
                    pct = (i + 1) * 100 // total
                    print(f"  解压中... {pct}% ({i+1}/{total})")
                elif i == total - 1:
                    print(f"  解压完成！({total} 文件)")

    finally:
        if tmp_zip.exists():
            tmp_zip.unlink()

    # 写入已解压标记
    (target / ".extracted").write_text("1", encoding="utf-8")


def is_extracted(target: Path) -> bool:
    """检查程序是否已解压"""
    return (target / ".extracted").exists() and (target / "launcher.py").exists()


def launch_app(target: Path) -> bool:
    """从解压目录启动程序"""
    pythonw = target / "venv" / "Scripts" / "pythonw.exe"
    launcher = target / "launcher.py"

    if not launcher.exists():
        print(f"[错误] 未找到启动脚本: {launcher}")
        return False

    if pythonw.exists():
        subprocess.Popen([str(pythonw), str(launcher)])
    else:
        # 无 venv 时尝试系统 Python
        subprocess.Popen([sys.executable or "python", str(launcher)])
    return True


def run_from_dev():
    """开发模式：直接运行同目录的启动器"""
    launcher = EXE_DIR / "launcher.py"
    if launcher.exists():
        subprocess.run([sys.executable, str(launcher)])
    else:
        print("[错误] 开发模式下找不到 launcher.py")


# ── 主入口 ──

def main():
    # 判断运行模式
    if not getattr(sys, "frozen", False):
        run_from_dev()
        return

    exe_path = Path(sys.executable)

    # 检查 ZIP 脚标
    try:
        zip_size = get_zip_size(exe_path)
    except (OSError, struct.error) as e:
        print(f"[错误] 无法读取程序数据包: {e}")
        input("按 Enter 退出...")
        return

    if zip_size <= 0:
        print("[错误] 程序数据包无效")
        input("按 Enter 退出...")
        return

    # 解压（如需要）
    if not is_extracted(TARGET_DIR):
        print("=" * 50)
        print("  中医药智能问答系统 -- 首次解压中")
        print("=" * 50)
        print(f"  目标目录: {TARGET_DIR}")
        print(f"  数据大小: {zip_size / (1024**3):.1f} GB")
        print("  请耐心等待，大型文件解压可能需要数分钟...")
        print()

        try:
            extract_zip(exe_path, TARGET_DIR, zip_size)
        except Exception as e:
            print(f"\n[错误] 解压失败: {e}")
            print("  可能原因：磁盘空间不足 / 权限不够 / 文件被占用")
            input("按 Enter 退出...")
            return

        print()
        print("  解压完成！正在启动程序...")

    # 启动
    if not launch_app(TARGET_DIR):
        print(f"[错误] 启动失败，请尝试手动运行: {TARGET_DIR / 'launcher.py'}")
        input("按 Enter 退出...")


if __name__ == "__main__":
    main()
