@echo off
chcp 65001 >nul
title 构建中医药智能问答系统安装包

setlocal enabledelayedexpansion

echo ============================================
echo   构建中医药智能问答系统安装包
echo ============================================
echo.

:: 设置路径
set "ROOT=%~dp0.."
set "PYTHON=D:\Python\python.exe"
set "SPEC_FILE=%ROOT%\TCMAI.spec"
set "ISS_FILE=%ROOT%\installer.iss"
set "OUTPUT_DIR=%ROOT%\installer_output"

:: 检查 Python
if not exist "%PYTHON%" (
    echo [错误] 未找到 Python: %PYTHON%
    pause
    exit /b 1
)

:: 检查 PyInstaller
"%PYTHON%" -m PyInstaller --version >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo [错误] PyInstaller 未安装
    echo 请先运行: "%PYTHON%" -m pip install pyinstaller
    pause
    exit /b 1
)

:: 清理旧的构建文件
echo [1/4] 清理旧的构建文件...
if exist "%ROOT%\build" rd /s /q "%ROOT%\build"
if exist "%ROOT%\dist" rd /s /q "%ROOT%\dist"

:: 运行 PyInstaller
echo.
echo [2/4] 运行 PyInstaller 打包...
"%PYTHON%" -m PyInstaller "%SPEC_FILE%" --clean --noconfirm
if !ERRORLEVEL! neq 0 (
    echo [错误] PyInstaller 打包失败
    pause
    exit /b 1
)

:: 检查 Inno Setup
set "ISCC="
for %%i in (ISCC.exe) do set "ISCC=%%~$PATH:i"
if "%ISCC%"=="" (
    :: 尝试默认安装路径
    if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
        set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    ) else if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
        set "ISCC=C:\Program Files\Inno Setup 6\ISCC.exe"
    ) else if exist "E:\Inno_Setup_6\Inno Setup 6\ISCC.exe" (
        set "ISCC=E:\Inno_Setup_6\Inno Setup 6\ISCC.exe"
    )
)

if "%ISCC%"=="" (
    echo.
    echo [警告] 未找到 Inno Setup 6
    echo 请下载并安装 Inno Setup 6: https://jrsoftware.org/isinfo.php
    echo 安装后重新运行此脚本
    echo.
    echo PyInstaller 打包已完成，文件位于: %ROOT%\dist\TCMAI
    pause
    exit /b 0
)

:: 运行 Inno Setup
echo.
echo [3/4] 运行 Inno Setup 创建安装包...
"%ISCC%" "%ISS_FILE%"
if !ERRORLEVEL! neq 0 (
    echo [错误] Inno Setup 编译失败
    pause
    exit /b 1
)

:: 完成
echo.
echo [4/4] 构建完成！
echo.
echo 安装包位置: %OUTPUT_DIR%\中医药智能问答系统_Setup.exe
echo.
echo ============================================
pause
