@echo off
chcp 65001 >nul
title 中医药智能问答系统 - 单文件安装包构建

echo ============================================
echo   中医药智能问答系统 - 单文件安装包构建
echo   使用 Windows 内置 IExpress
echo   生成的 Setup.exe 单独一个文件就能运行
echo ============================================
echo.

cd /d "%~dp0.."

set PYTHON=

:: 优先使用 venv Python，其次便携 Python
if exist "venv\Scripts\python.exe" (
    set PYTHON=venv\Scripts\python.exe
) else if exist "python\python.exe" (
    set PYTHON=python\python.exe
) else (
    echo [错误] 未找到 Python 运行环境。
    echo   请确保已创建虚拟环境或在项目根目录有便携 Python。
    pause
    exit /b 1
)

echo 使用 Python: %PYTHON%
echo.
echo [1/1] 开始构建单文件安装包...
echo.

"%PYTHON%" scripts\build_sfx_exe.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [错误] 构建失败，请检查上方错误信息。
    pause
    exit /b 1
)

echo.
pause
