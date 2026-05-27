@echo off
chcp 65001 >nul
title 中医药智能问答系统 - 便携包构建

echo ============================================
echo   中医药智能问答系统 - 便携包构建
echo   使用 Windows 内置 IExpress 打包
echo   客户电脑无需安装任何软件
echo ============================================
echo.

cd /d "%~dp0.."

:: 设置 Python 输出编码为 UTF-8（防止 GBK 编码错误）
set PYTHONIOENCODING=utf-8

:: 检查 python.exe（优先使用便携 Python）
if exist "python\python.exe" (
    set PYTHON=python\python.exe
) else (
    where python >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo [错误] 未找到 Python，请先安装 Python 或使用便携版。
        pause
        exit /b 1
    )
    set PYTHON=python
)

echo [1/1] 开始构建便携包（可能需要几分钟到几十分钟）...
echo.
"%PYTHON%" scripts\build_portable.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [错误] 构建失败，请检查上方错误信息。
    pause
    exit /b 1
)

echo.
pause
