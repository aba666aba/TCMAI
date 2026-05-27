#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中医药智能问答系统 -- 单文件自解压安装包构建脚本

使用 Windows 内置 IExpress 生成一个 Setup.exe，内含全部项目文件。
用户双击 Setup.exe → 选择解压目录 → 自动解压完成 → 直接运行。

构建方式：项目根目录下执行
    python scripts\build_sfx_exe.py

依赖：IExpress (Windows 系统自带)，无其他外部工具。
"""

import os
import sys
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist"
ZIP_PATH = DIST_DIR / "app_bundle.zip"
SETUP_EXE = DIST_DIR / "中医药智能问答系统_Setup.exe"
SED_PATH = DIST_DIR / "setup.sed"
EXTRACT_CMD = DIST_DIR / "_extract.cmd"

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".qwen", ".mypy_cache",
    ".pytest_cache", "build",
    # 排除旧的构建产物目录
    "dist",
}
EXCLUDE_FILES = {".gitignore", ".gitattributes", "CACHEDIR.TAG"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
# 排除 dist/ 中可能遗留的旧文件（但保留 app_bundle.zip 和最终 exe）
EXCLUDE_ROOT_PATTERNS = {
    "中医药智能问答系统_Setup_*.exe",
    "中医药智能问答系统_TEMP.zip",
    "中医药智能问答系统_Setup.zip",
    "*.sed",
    "_extract.cmd",
    "build_log.txt",
}

# IExpress SED 模板（UTF-16LE 编码）
# 使用 KeepCompressed=0 让 IExpress 把文件解压到目标目录
SED_TEMPLATE = """\
[Version]
Class=IEXPRESS
SEDVersion=3

[Options]
PackagePurpose=InstallApp
ShowInstallProgramWindow=0
HideExtractAnimation=1
UseLongFileName=1
InsideCompressed=0
CompressionType=MSZIP
KeepCompressed=0
OverwriteMode=1
ExtractOnlyIfNewer=0
SelDirPrompt=请选择解压目标文件夹（程序将解压到所选目录下）：
InstallPrompt=中医药智能问答系统 安装程序
UninstallPrompt=
DisplayLicense=0
FinishMessage=解压完成！请双击「启动中医药智能问答系统.pyw」运行程序。
AskSaveTargetDirectory=1
LongFileNames=1
ShortFileNames=0

[SourceFiles]
SourceFiles0=FILE0
SourceFiles1=FILE1

[SourceFiles0]
%FILE0="{ZIP_SRC}"
%TARGET0=%TARGET%\\app_bundle.zip

[SourceFiles1]
%FILE1="{CMD_SRC}"
%TARGET1=%TARGET%\\_extract.cmd

[Strings]
InstallProgram=%TARGET%\\_extract.cmd
"""


def log(msg: str) -> None:
    print(msg, flush=True)


def collect_files() -> list:
    """收集需要打包的文件列表（返回相对路径）"""
    files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dp = Path(dirpath)
        parts = dp.relative_to(ROOT).parts
        # 排除黑名单目录
        if any(p in EXCLUDE_DIRS for p in parts):
            dirnames.clear()
            continue
        for fname in filenames:
            fp = dp / fname
            if fname in EXCLUDE_FILES or fp.suffix in EXCLUDE_SUFFIXES:
                continue
            rel = fp.relative_to(ROOT)
            # 排除 dist/ 中的旧文件（但保留即将写入的 app_bundle.zip）
            if rel.parts and rel.parts[0] == "dist":
                continue
            files.append(rel)
    return sorted(files, key=lambda p: str(p).casefold())


def create_app_zip(files: list) -> int:
    """创建项目 ZIP 包，返回未压缩总字节数"""
    log("")
    log("  [打包] 正在创建项目 ZIP 包...")
    log("    共 %d 个文件" % len(files))

    total_raw = 0
    t0 = time.time()
    written = 0

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    # 删除旧 ZIP
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(str(ZIP_PATH), "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for i, rel_fp in enumerate(files):
            abs_path = ROOT / rel_fp
            try:
                arcname = rel_fp.as_posix()
                zf.write(str(abs_path), arcname)
                sz = abs_path.stat().st_size
                total_raw += sz
                written += sz

                # 每 2000 文件或每 500MB 输出一次进度
                if (i + 1) % 2000 == 0:
                    elapsed = time.time() - t0
                    pct = (i + 1) * 100 // len(files)
                    log("    %5d/%d (%d%%)  %.0f MB  %s" % (
                        i + 1, len(files), pct, written / (1024 * 1024), rel_fp))
            except (OSError, PermissionError) as e:
                log("    [跳过] %s — %s" % (rel_fp, e))

    elapsed = time.time() - t0
    zip_size = ZIP_PATH.stat().st_size
    ratio = zip_size / total_raw * 100 if total_raw > 0 else 0
    log("")
    log("    [完成] ZIP 包大小: %.1f MB -> %.1f MB (%.0f%%)  耗时 %.0f 秒" % (
        total_raw / (1024 * 1024), zip_size / (1024 * 1024), ratio, elapsed))
    return total_raw


def create_extract_script():
    """创建解压后自动执行的 _extract.cmd"""
    content = """\
@echo off
chcp 65001 >nul
title 中医药智能问答系统 - 正在解压...

echo ============================================
echo   中医药智能问答系统 - 解压中
echo ============================================
echo.

:: 获取当前脚本所在目录（即 IExpress 目标目录）
set "APP_DIR=%~dp0"
set "ZIP_FILE=%APP_DIR%app_bundle.zip"

:: 检查 ZIP 文件
if not exist "%ZIP_FILE%" (
    echo [错误] 找不到程序数据包 app_bundle.zip
    echo   目录: %APP_DIR%
    pause
    exit /b 1
)

echo 正在解压程序文件到：
echo   %APP_DIR%
echo 请耐心等待，大型文件解压可能需要数分钟...
echo.

:: 使用 PowerShell 解压（Windows 10/11 原生支持）
powershell -NoProfile -Command "
    try {
        $zip = '%ZIP_FILE%';
        $dest = '%APP_DIR%';
        Write-Host ('正在解压: ' + $zip);
        Add-Type -AssemblyName System.IO.Compression.FileSystem;
        $archive = [System.IO.Compression.ZipFile]::OpenRead($zip);
        $count = $archive.Entries.Count;
        $i = 0;
        foreach ($entry in $archive.Entries) {
            $i++;
            if ($i %% 1000 -eq 0) {
                Write-Host ('' + $i + '/' + $count + '  ' + $entry.FullName);
            }
            $destPath = [System.IO.Path]::Combine($dest, $entry.FullName);
            if ($entry.FullName -match '/$' -or $entry.FullName -match '\\\\$') {
                [System.IO.Directory]::CreateDirectory($destPath) | Out-Null;
            } else {
                [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($destPath)) | Out-Null;
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $destPath, $true);
            }
        }
        $archive.Dispose();
        Write-Host '解压完成！';
    } catch {
        Write-Host ('解压失败: ' + $_.Exception.Message);
        exit 1;
    }
"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [错误] 解压失败，可能原因：
    echo   1. 磁盘空间不足（需至少 5 GB 可用空间）
    echo   2. 权限不足（请以管理员身份运行）
    echo   3. 文件被占用
    pause
    exit /b 1
)

:: 清理临时文件
echo 正在清理临时文件...
del /f /q "%ZIP_FILE%" >nul 2>&1
del /f /q "%~f0" >nul 2>&1

echo.
echo ============================================
echo   解压完成！
echo.
echo   请双击「启动中医药智能问答系统.pyw」运行
echo ============================================
echo.
pause
"""
    EXTRACT_CMD.write_text(content, encoding="utf-8-sig")
    log("  [OK] 解压脚本已创建")


def generate_sed():
    """生成 IExpress SED 配置文件（UTF-16LE 编码）"""
    sed_content = SED_TEMPLATE.format(
        ZIP_SRC=str(ZIP_PATH.resolve()),
        CMD_SRC=str(EXTRACT_CMD.resolve()),
        TARGET_DIR="%TARGET%\\",
    )
    # IExpress 要求 SED 文件使用 UTF-16LE 编码
    SED_PATH.write_text(sed_content, encoding="utf-16le")
    log("  [OK] SED 配置已生成")


def check_iexpress() -> bool:
    """检查 IExpress 是否可用"""
    iexpress = Path(os.environ.get("SystemRoot", "C:\\Windows")) / "System32" / "iexpress.exe"
    if iexpress.exists():
        return True
    # 尝试从 PATH 查找
    try:
        subprocess.run(["iexpress.exe", "/?"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_iexpress():
    """调用 IExpress 生成 Setup.exe"""
    log("")
    log("  [构建] 正在调用 IExpress 生成安装包...")
    log("    这可能耗时数分钟到数十分钟，请耐心等待。")
    log("")

    # 删除旧的 Setup.exe
    if SETUP_EXE.exists():
        SETUP_EXE.unlink()

    # 切换到 DIST_DIR 运行，因为 IExpress 会在 SED 所在目录生成输出
    cmd = [
        "iexpress.exe",
        "/N",           # 静默构建（不弹向导）
        str(SED_PATH),
    ]

    t0 = time.time()
    try:
        # IExpress 构建时会弹一个进度窗，使用 CREATE_NO_WINDOW 可能影响
        # 使用 shell=True 让系统处理文件关联
        result = subprocess.run(
            cmd,
            cwd=str(DIST_DIR),
            capture_output=True,
            text=True,
            timeout=3600,  # 最多 1 小时
        )
        elapsed = time.time() - t0
        log("    IExpress 返回码: %d" % result.returncode)

        if result.stdout:
            log("    stdout: " + result.stdout.strip())
        if result.stderr:
            log("    stderr: " + result.stderr.strip())

        if result.returncode != 0:
            log("  [错误] IExpress 构建失败（返回码 %d）" % result.returncode)
            log("  提示：可手动运行 iexpress.exe 打开 SED 文件调试")
            # 失败时不清理，保留文件供调试
            return False

    except subprocess.TimeoutExpired:
        log("  [错误] IExpress 超时（超过 1 小时）")
        return False
    except FileNotFoundError:
        log("  [错误] 找不到 iexpress.exe，请确认是 Windows 系统")
        return False

    # IExpress 输出的 exe 文件名为 SED 中设定的名称或默认名
    # 查找生成的 exe
    generated = list(DIST_DIR.glob("*.exe"))
    generated = [f for f in generated if f.name != SETUP_EXE.name and "Setup" in f.name]

    # 有时 IExpress 会以 SED 文件名生成 exe
    sed_exe = DIST_DIR / "setup.exe"
    if sed_exe.exists():
        shutil.move(str(sed_exe), str(SETUP_EXE))

    if not SETUP_EXE.exists():
        log("  [错误] IExpress 未生成预期的 exe 文件")
        log("    生成的 exe: %s" % [str(f) for f in generated])
        return False

    sz = SETUP_EXE.stat().st_size
    log("")
    log("    [OK] Setup.exe 已生成: %s" % SETUP_EXE)
    log("    大小: %.1f MB" % (sz / (1024 * 1024)))
    log("    耗时: %.0f 秒" % elapsed)
    return True


def cleanup():
    """清理临时文件"""
    for f in [ZIP_PATH, SED_PATH, EXTRACT_CMD]:
        if f.exists():
            f.unlink()
            log("  [清理] 删除: %s" % f.name)


def main():
    log("=" * 55)
    log("   中医药智能问答系统 -- 单文件自解压安装包")
    log("=" * 55)
    log("")

    # 1. 检查系统
    log("[01/5] 检查 IExpress 工具")
    if not check_iexpress():
        log("  [错误] 未找到 iexpress.exe，此工具仅在 Windows 系统可用。")
        sys.exit(1)
    log("  OK")

    # 2. 扫描文件
    log("")
    log("[02/5] 扫描项目文件...")
    files = collect_files()
    total_bytes = sum((ROOT / f).stat().st_size for f in files)
    log("  找到 %d 个文件，共 %.1f MB" % (len(files), total_bytes / (1024 * 1024)))
    if not files:
        log("  [错误] 没有找到任何可打包的文件")
        sys.exit(1)

    # 3. 创建 ZIP + 解压脚本
    log("")
    log("[03/5] 创建项目数据包...")
    create_app_zip(files)
    create_extract_script()

    # 4. 生成 IExpress SED
    log("")
    log("[04/5] 生成 IExpress SED 配置...")
    generate_sed()

    # 5. 构建 Setup.exe
    log("")
    log("[05/5] 构建单文件安装包...")
    if not run_iexpress():
        cleanup()
        log("")
        log("=" * 55)
        log("  构建失败！请检查上方错误信息。")
        log("=" * 55)
        sys.exit(1)

    # 清理临时文件
    cleanup()

    # 完成
    sz = SETUP_EXE.stat().st_size
    log("")
    log("=" * 55)
    log("  构建成功！")
    log("")
    log("  输出: %s" % SETUP_EXE)
    log("  大小: %.1f MB" % (sz / (1024 * 1024)))
    log("")
    log("  使用说明：")
    log("  1. 将 %s 复制到目标电脑" % SETUP_EXE.name)
    log("  2. 双击运行")
    log("  3. 选择解压目录")
    log("  4. 等待解压完成")
    log("  5. 解压后双击「启动中医药智能问答系统.pyw」运行")
    log("")
    log("  [提示] 首次解压后，下次可直接运行，无需再次解压。")
    log("=" * 55)


if __name__ == "__main__":
    main()
