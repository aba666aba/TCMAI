@echo off
chcp 65001 >nul
title 构建中医药智能问答系统安装包

setlocal enabledelayedexpansion

echo ============================================
echo   构建中医药智能问答系统安装包
echo ============================================
echo.

:: ---- 项目路径（相对本脚本定位，不写死盘符）----
set "ROOT=%~dp0.."
set "SPEC_FILE=%ROOT%\TCMAI.spec"
set "ISS_FILE=%ROOT%\installer.iss"
set "OUTPUT_DIR=%ROOT%\installer_output"

:: ---- 定位 Python：优先环境变量 TCMAI_PYTHON，其次 PATH ----
if defined TCMAI_PYTHON (
    set "PYTHON=%TCMAI_PYTHON%"
) else (
    set "PYTHON="
    for %%p in (python.exe) do if not defined PYTHON set "PYTHON=%%~$PATH:p"
    if not defined PYTHON (
        for %%p in (python3.exe) do if not defined PYTHON set "PYTHON=%%~$PATH:p"
    )
)

if not defined PYTHON (
    echo [错误] 未找到 Python。
    echo        请把 Python 加入 PATH，或先设置环境变量后再运行本脚本：
    echo            set TCMAI_PYTHON=D:\你的路径\python.exe
    pause
    exit /b 1
)

echo [环境] Python: %PYTHON%
"%PYTHON%" --version
echo.

:: ---- 检查 PyInstaller ----
"%PYTHON%" -m PyInstaller --version >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo [错误] PyInstaller 未安装
    echo        请先运行: "%PYTHON%" -m pip install pyinstaller
    pause
    exit /b 1
)

:: ---- 检查 spec 文件，缺失则用 PyInstaller 生成一份基础版本 ----
if not exist "%SPEC_FILE%" (
    echo [提示] 未找到 %SPEC_FILE%
    echo        正在生成基础 spec（入口 src\main.py）...
    pushd "%ROOT%"
    "%PYTHON%" -m PyInstaller.utils.cliutils.makespec --name TCMAI --windowed --onedir ^
        --paths "%ROOT%\src" --add-data "src;src" "src\main.py"
    if !ERRORLEVEL! neq 0 (
        echo [错误] 生成 spec 失败。可手动创建 src\main.py 对应的 spec 后重试。
        popd
        pause
        exit /b 1
    )
    popd
    echo        已生成 %SPEC_FILE%
    echo.
)

:: ---- 清理旧构建产物 ----
echo [1/4] 清理旧的构建文件...
if exist "%ROOT%\build" rd /s /q "%ROOT%\build"
if exist "%ROOT%\dist" rd /s /q "%ROOT%\dist"

:: ---- PyInstaller 打包 ----
echo.
echo [2/4] 运行 PyInstaller 打包...
pushd "%ROOT%"
"%PYTHON%" -m PyInstaller "%SPEC_FILE%" --clean --noconfirm
set "PYI_RC=!ERRORLEVEL!"
popd
if !PYI_RC! neq 0 (
    echo [错误] PyInstaller 打包失败
    pause
    exit /b 1
)

:: ---- 定位 Inno Setup ----
set "ISCC="
for %%i in (ISCC.exe) do set "ISCC=%%~$PATH:i"
if not defined ISCC (
    for %%c in (
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
        "C:\Program Files\Inno Setup 6\ISCC.exe"
        "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
    ) do if not defined ISCC if exist %%c set "ISCC=%%~c"
)
:: 允许用环境变量指定
if defined TCMAI_ISCC set "ISCC=%TCMAI_ISCC%"

if not defined ISCC (
    echo.
    echo [警告] 未找到 Inno Setup 6
    echo        请下载安装: https://jrsoftware.org/isinfo.php
    echo        安装后重新运行本脚本，或设置 TCMAI_ISCC 指向 ISCC.exe
    echo.
    echo        PyInstaller 打包已完成，文件位于: %ROOT%\dist\TCMAI
    pause
    exit /b 0
)

:: ---- Inno Setup 生成安装包 ----
echo.
echo [3/4] 运行 Inno Setup 创建安装包...
pushd "%ROOT%"
"%ISCC%" "%ISS_FILE%"
set "ISS_RC=!ERRORLEVEL!"
popd
if !ISS_RC! neq 0 (
    echo [错误] Inno Setup 编译失败
    pause
    exit /b 1
)

:: ---- 完成 ----
echo.
echo [4/4] 构建完成！
echo.
echo 安装包位置: %OUTPUT_DIR%\中医药智能问答系统_Setup.exe
echo.
echo ============================================
pause
