@echo off
chcp 65001 >nul
title 中医药智能问答系统 - 打包程序

echo ============================================
echo   中医药智能问答系统 - EXE 打包脚本
echo ============================================
echo.

:: 进入项目根目录
cd /d "%~dp0.."

:: 检查 PyInstaller（通过 python -m 调用最可靠）
python -m PyInstaller --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 未找到 PyInstaller，请先安装：
    echo   pip install pyinstaller
    pause
    exit /b 1
)

echo [1/4] 清理旧构建...
if exist "build" rmdir /s /q "build" >nul
if exist "dist" rmdir /s /q "dist" >nul
if exist "__pycache__" rmdir /s /q "__pycache__" >nul
echo   [OK]

echo [2/4] 检查依赖...
python -c "import torch; print('  PyTorch', torch.__version__, '(CUDA:', torch.cuda.is_available(), ')')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo   [警告] PyTorch 未安装，AI 模式将不可用
)

echo [3/4] 开始打包（预计 3-10 分钟）...
echo   打包后程序位于 dist\TCMAI_Desktop\
echo.
python -m PyInstaller --clean tcmai.spec
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 打包失败，请检查上方错误信息
    pause
    exit /b 1
)

echo [4/4] 构建完成！
echo.
echo ============================================
echo   打包成功！
echo   程序位置：dist\TCMAI_Desktop\
echo   启动文件：dist\TCMAI_Desktop\中医药智能问答系统.exe
echo   快捷启动：dist\TCMAI_Desktop\快速启动.bat
echo.
echo   总大小：请查看 dist\TCMAI_Desktop\ 目录
echo ============================================

pause
