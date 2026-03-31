@echo off
chcp 65001 >nul
title 中医药智能问答系统 - 打包工具

echo ============================================================
echo   中医药智能问答系统 - EXE 打包工具
echo ============================================================
echo.

REM 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python
    pause
    exit /b 1
)

echo [1/4] 检查依赖...
pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo [提示] 正在安装 PyInstaller...
    pip install pyinstaller
) else (
    echo [OK] PyInstaller 已安装
)

echo.
echo [2/4] 清理旧文件...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
echo [OK] 清理完成

echo.
echo [3/4] 开始打包...
echo [提示] 打包过程约需 1-3 分钟，请耐心等待...
echo.

pyinstaller tcm_qa.spec

echo.
echo [4/4] 检查打包结果...
if exist "dist\中医药智能问答系统.exe" (
    echo [OK] 打包成功！
    echo.
    echo ============================================================
    echo   打包完成！
    echo ============================================================
    echo.
    echo 可执行文件位置：dist\中医药智能问答系统.exe
    echo.
    echo 使用方法：
    echo 1. 将 dist 文件夹复制到任意位置
    echo 2. 双击"中医药智能问答系统.exe"运行
    echo.
    echo 文件夹大小：约 30-50 MB
    echo.
    dir /h "dist\中医药智能问答系统.exe"
) else (
    echo [错误] 打包失败
    echo.
    echo 请查看上方的错误信息
)

echo.
pause
