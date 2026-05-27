@echo off
chcp 65001 >nul
title 中医药智能问答系统 - 单文件便携包构建

echo ============================================
echo   中医药智能问答系统 - 单文件便携包构建
echo   生成单独一个 exe，双击即用
echo ============================================
echo.

cd /d "%~dp0.."

set PYTHON=

if exist "venv\Scripts\python.exe" (
    set PYTHON=venv\Scripts\python.exe
) else if exist "python\python.exe" (
    set PYTHON=python\python.exe
) else (
    echo [错误] 未找到 Python 运行环境。
    pause
    exit /b 1
)

echo 使用 Python: %PYTHON%
echo 构建可能需要 5-15 分钟，请耐心等待...
echo.

"%PYTHON%" scripts\build_single_exe.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [错误] 构建失败，请检查上方错误信息。
    pause
    exit /b 1
)

echo.
pause
