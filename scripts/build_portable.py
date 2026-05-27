#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中医药智能问答系统 -- 便携包构建脚本

将整个项目打包为 ZIP 文件（Windows 原生支持，右击 -> 全部提取）。
客户电脑无需安装任何软件。

用法：
    python scripts\build_portable.py
"""

import os
import sys
import zipfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist"
ZIP_OUTPUT = DIST_DIR / "中医药智能问答系统_Setup.zip"

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".qwen", ".mypy_cache",
    ".pytest_cache", "build", "dist",
}
EXCLUDE_FILES = {".gitignore", ".gitattributes", "CACHEDIR.TAG"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pyd", ".whl"}


def log(msg):
    """立即刷新输出的日志函数"""
    print(msg, flush=True)


def collect_files():
    """收集需要打包的文件列表"""
    files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dp = Path(dirpath)
        parts = dp.relative_to(ROOT).parts
        if any(p in EXCLUDE_DIRS for p in parts):
            dirnames.clear()
            continue
        for fname in filenames:
            fp = dp / fname
            if fname in EXCLUDE_FILES or fp.suffix in EXCLUDE_SUFFIXES:
                continue
            files.append(fp.relative_to(ROOT))
    return sorted(files, key=lambda p: str(p).lower())


def create_zip(zip_path, files):
    """打包为 ZIP 文件"""
    total = len(files)
    total_bytes = sum((ROOT / f).stat().st_size for f in files)
    log("    共 %d 个文件，%.1f MB" % (total, total_bytes / (1024 * 1024)))
    log("    正在压缩（ZIP_DEFLATED），大文件可能需要数分钟...")
    log("")
    log("    序号   进度  文件名")

    t0 = time.time()
    written = 0
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for i, rel_fp in enumerate(files):
            abs_path = ROOT / rel_fp
            arcname = rel_fp.as_posix()
            zf.write(str(abs_path), arcname)
            written += abs_path.stat().st_size

            # 每 1000 个文件或每 500MB 输出一次进度
            if (i + 1) % 1000 == 0 or written >= (total_bytes // 2):
                elapsed = time.time() - t0
                pct = (i + 1) * 100 // total
                mb = written / (1024 * 1024)
                log("    %5d/%d (%d%%)  %.0f MB  %s" % (
                    i + 1, total, pct, mb, rel_fp))
                if written >= total_bytes // 2:
                    written = -999_999_999  # 只触发一次

    elapsed = time.time() - t0
    sz = zip_path.stat().st_size
    ratio = sz / total_bytes * 100 if total_bytes > 0 else 0
    log("")
    log("    [完成] %d 个文件打包完毕，耗时 %.1f 分钟" % (total, elapsed / 60))
    log("    原始大小: %.1f MB -> ZIP 大小: %.1f MB (%.0f%%)" % (
        total_bytes / (1024 * 1024), sz / (1024 * 1024), ratio))


def main():
    log("=" * 55)
    log("   中医药智能问答系统 -- 便携包构建工具")
    log("=" * 55)
    log("")

    # 前置检查
    log("[检查] 便携 Python 路径")
    py = ROOT / "python" / "python.exe"
    log("  %s -> %s" % ("OK" if py.exists() else "ERR", py))

    log("[检查] 项目源码")
    src = ROOT / "src" / "desktop" / "app.py"
    log("  %s -> %s" % ("OK" if src.exists() else "ERR", src))

    log("[检查] 数据库")
    db = ROOT / "data" / "tcm_knowledge.db"
    log("  %s -> %s" % ("OK" if db.exists() else "WARN(无)", db))

    log("[检查] NLU 模型")
    mdl = ROOT / "models" / "nlu" / "qwen"
    has_model = mdl.exists() and any(mdl.iterdir())
    log("  %s -> %s" % ("OK" if has_model else "WARN(未下载)", mdl))

    log("")
    log("[扫描] 项目文件...")
    files = collect_files()
    total_bytes = sum((ROOT / f).stat().st_size for f in files)
    log("  找到 %d 个文件，共 %.1f MB" % (len(files), total_bytes / (1024 * 1024)))

    if not files:
        log("[错误] 没有找到任何文件，请检查项目目录")
        sys.exit(1)

    # 创建输出目录
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    # 打包
    log("")
    log("[打包] 开始创建 ZIP...")
    try:
        # 删除旧的 ZIP（如果存在）
        if ZIP_OUTPUT.exists():
            ZIP_OUTPUT.unlink()

        create_zip(ZIP_OUTPUT, files)
    except Exception as e:
        log("[错误] 打包失败: %s" % e)
        if ZIP_OUTPUT.exists():
            ZIP_OUTPUT.unlink()
        sys.exit(1)

    # 完成
    sz = ZIP_OUTPUT.stat().st_size
    log("")
    log("=" * 55)
    log("  构建完成！")
    log("  输出: %s" % ZIP_OUTPUT)
    log("  大小: %.1f MB" % (sz / (1024 * 1024)))
    log("=" * 55)
    log("")
    log("使用说明：")
    log("  1. 将 %s 复制到目标电脑" % ZIP_OUTPUT.name)
    log("  2. 右键 -> 全部提取（Windows 原生支持，无需装任何软件）")
    log("  3. 选择目标文件夹")
    log("  4. 解压后双击「启动中医药智能问答系统.pyw」运行")
    log("")
    log("  [提示] 整个目录可拷到 U 盘带走，即插即用")
    log("")


if __name__ == "__main__":
    main()
