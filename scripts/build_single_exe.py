#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中医药智能问答系统 -- 单文件自解压便携包构建脚本

将整个项目打包成单独一个 exe，双击自动解压运行。

原理：
  1. 创建项目 ZIP 包
  2. 用 PyInstaller --onefile 编译自解压启动器 (launcher_stub.py)
  3. 将 ZIP 附加到 stub.exe 末尾，写入 8 字节脚标
  4. 最终产物是一个完整的单文件便携 exe

用法：
  python scripts\build_single_exe.py
"""

import os
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build" / "single_exe"
STUB_SRC = ROOT / "scripts" / "launcher_stub.py"
ZIP_PATH = DIST_DIR / "app_bundle.zip"
STUB_EXE = DIST_DIR / "_stub.exe"
OUTPUT_EXE = DIST_DIR / "中医药智能问答系统_Portable.exe"

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".qwen", ".mypy_cache",
    ".pytest_cache", "build", "dist",
}
EXCLUDE_FILES = {".gitignore", ".gitattributes", "CACHEDIR.TAG"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
EXCLUDE_DIST_PATTERNS = {
    "中医药智能问答系统_*.exe",
    "中医药智能问答系统_Setup.*",
    "中医药智能问答系统_TEMP.*",
    "*.sed", "_extract.cmd",
    "build_log.txt", "test_*",
}


def log(msg: str):
    print(msg, flush=True)


def cleanup_dist():
    """清理 dist 目录中的旧文件"""
    for f in DIST_DIR.iterdir():
        if f.suffix in {".exe", ".zip", ".sed", ".cmd", ".txt"}:
            if any(f.match(p) for p in EXCLUDE_DIST_PATTERNS):
                continue
            f.unlink()
            log(f"  [清理] 删除旧文件: {f.name}")


def collect_files() -> list:
    """收集需要打包的文件"""
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
            rel = fp.relative_to(ROOT)
            if rel.parts and rel.parts[0] == "dist":
                continue
            files.append(rel)
    return sorted(files, key=lambda p: str(p).casefold())


def create_app_zip(files: list) -> Path:
    """创建项目 ZIP 包，返回 ZIP 路径"""
    log("")
    log("  [打包] 正在创建项目 ZIP 包 ({:,} 个文件)".format(len(files)))

    total_raw = 0
    t0 = time.time()

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(str(ZIP_PATH), "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for i, rel_fp in enumerate(files):
            abs_path = ROOT / rel_fp
            try:
                arcname = rel_fp.as_posix()
                zf.write(str(abs_path), arcname)
                total_raw += abs_path.stat().st_size
                if (i + 1) % 2000 == 0:
                    pct = (i + 1) * 100 // len(files)
                    log("    {:>5}/{} ({:>2}%)  {:.0f} MB".format(
                        i + 1, len(files), pct, total_raw / (1024 * 1024)))
            except (OSError, PermissionError) as e:
                log("    [跳过] {} — {}".format(rel_fp, e))

    elapsed = time.time() - t0
    zip_size = ZIP_PATH.stat().st_size
    ratio = zip_size / total_raw * 100 if total_raw > 0 else 0
    log("  [完成] 原始 {:.1f} MB → ZIP {:.1f} MB ({:.0f}%)  耗时 {:.0f} 秒".format(
        total_raw / (1024 * 1024), zip_size / (1024 * 1024), ratio, elapsed))
    return ZIP_PATH


def compile_stub() -> Path:
    """用 PyInstaller 编译自解压启动器"""
    log("")
    log("  [编译] 正在编译自解压启动器...")

    # 清理旧的编译产物
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    if STUB_EXE.exists():
        STUB_EXE.unlink()

    pyinstaller = ROOT / "venv" / "Scripts" / "pyinstaller.exe"
    if not pyinstaller.exists():
        pyinstaller = "pyinstaller"

    cmd = [
        str(pyinstaller),
        "--onefile",
        "--distpath", str(DIST_DIR),
        "--workpath", str(BUILD_DIR / "work"),
        "--specpath", str(BUILD_DIR),
        "--name", "_stub",
        "--noconfirm",
        "--clean",
        "--console",             # 首次解压显示进度，后续秒退
        "--uac-admin",           # 请求管理员权限（如需写入 C:）
        str(STUB_SRC),
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0

    # 删除编译中间产物
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR, ignore_errors=True)
    # 删除多余的 spec 文件
    spec_file = ROOT / "_stub.spec"
    if spec_file.exists():
        spec_file.unlink()

    if result.returncode != 0:
        log("  [错误] PyInstaller 编译失败 (返回码 {})".format(result.returncode))
        log("  stderr: " + result.stderr[-500:])
        raise RuntimeError("PyInstaller 编译失败")

    if not STUB_EXE.exists():
        raise RuntimeError("PyInstaller 未生成 exe")

    sz = STUB_EXE.stat().st_size
    log("  [完成] stub.exe: {:.1f} MB  耗时 {:.0f} 秒".format(
        sz / (1024 * 1024), elapsed))
    return STUB_EXE


def append_zip(stub: Path, zip_path: Path) -> Path:
    """将 ZIP 附加到 stub.exe 末尾，写入脚标"""
    log("")
    log("  [组装] 正在合并 exe + ZIP...")

    zip_size = zip_path.stat().st_size

    # copy /b stub.exe + app_bundle.zip = output.exe
    t0 = time.time()
    with open(OUTPUT_EXE, "wb") as out:
        # 写入 stub
        with open(stub, "rb") as f:
            shutil.copyfileobj(f, out)
        # 写入 ZIP
        with open(zip_path, "rb") as f:
            shutil.copyfileobj(f, out)
        # 写入 8 字节脚标（ZIP 大小）
        out.write(struct.pack("<Q", zip_size))

    elapsed = time.time() - t0
    exe_size = OUTPUT_EXE.stat().st_size

    # 验证
    with open(OUTPUT_EXE, "rb") as f:
        f.seek(-8, os.SEEK_END)
        read_size = struct.unpack("<Q", f.read(8))[0]
    assert read_size == zip_size, f"脚标验证失败: 期望 {zip_size}, 实际 {read_size}"

    log("  [完成] 最终 exe: {:.1f} MB (stub {:.1f} MB + ZIP {:.1f} MB)  耗时 {:.0f} 秒".format(
        exe_size / (1024 * 1024),
        stub.stat().st_size / (1024 * 1024),
        zip_size / (1024 * 1024),
        elapsed))
    return OUTPUT_EXE


def cleanup():
    """清理中间文件"""
    for f in [ZIP_PATH, STUB_EXE]:
        if f.exists():
            f.unlink()
            log(f"  [清理] 删除: {f.name}")


def main():
    log("=" * 55)
    log("   中医药智能问答系统 -- 单文件便携包构建")
    log("=" * 55)
    log("")

    DIST_DIR.mkdir(parents=True, exist_ok=True)

    # 清理旧文件
    log("[01/6] 清理旧的构建产物...")
    cleanup_dist()

    # 扫描文件
    log("")
    log("[02/6] 扫描项目文件...")
    files = collect_files()
    total_bytes = sum((ROOT / f).stat().st_size for f in files)
    log("  找到 {:,} 个文件，共 {:.1f} MB".format(len(files), total_bytes / (1024 * 1024)))
    if not files:
        log("  [错误] 没有找到可打包的文件")
        sys.exit(1)

    # 创建 ZIP
    log("")
    log("[03/6] 创建项目 ZIP 数据包...")
    zip_path = create_app_zip(files)

    # 编译启动器
    log("")
    log("[04/6] 编译自解压启动器...")
    stub_path = compile_stub()

    # 组装
    log("")
    log("[05/6] 组装最终 exe...")
    output = append_zip(stub_path, zip_path)

    # 清理
    log("")
    log("[06/6] 清理临时文件...")
    cleanup()

    # 完成
    sz = output.stat().st_size
    log("")
    log("=" * 55)
    log("  构建成功！")
    log("")
    log("  输出: {}".format(output))
    log("  大小: {:.1f} MB".format(sz / (1024 * 1024)))
    log("")
    log("  使用说明：")
    log("  1. 将 {} 复制到任意位置".format(output.name))
    log("  2. 双击运行")
    log("  3. 首次运行自动解压到同目录下的 TCMAI/ 文件夹")
    log("  4. 后续运行秒开，无需再次解压")
    log("")
    log("  [提示] 整个 TCMAI/ 文件夹可拷贝到 U 盘使用")
    log("=" * 55)


if __name__ == "__main__":
    main()
