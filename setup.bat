@echo off
chcp 65001 >nul
title 中医药智能问答系统 - 环境配置

setlocal enabledelayedexpansion

echo ============================================
echo   中医药智能问答系统 - 环境配置
echo   本脚本仅需在首次运行时执行一次
echo ============================================
echo.

:: 获取当前目录（项目根目录）
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

echo  项目目录: %ROOT%
echo.

:: ── 1. 修复 pyvenv.cfg ──
echo [1/5] 修复虚拟环境路径...
set "VENV_CFG=%ROOT%\venv\pyvenv.cfg"
set "PYTHON_HOME=%ROOT%\python"

if exist "%VENV_CFG%" (
    powershell -NoProfile -Command ^
        "$c = Get-Content '%VENV_CFG%'; " ^
        "$changed = $false; " ^
        "for ($i = 0; $i -lt $c.Count; $i++) { " ^
        "  if ($c[$i] -match '^home = ') { " ^
        "    $old = ($c[$i] -split '= ')[1].Trim(); " ^
        "    if ($old -ne '%PYTHON_HOME%') { " ^
        "      $c[$i] = 'home = %PYTHON_HOME%'; " ^
        "      $changed = $true; " ^
        "    } " ^
        "    break; " ^
        "  } " ^
        "}; " ^
        "if ($changed) { Set-Content '%VENV_CFG%' $c -Encoding UTF8; Write-Host '  已更新 pyvenv.cfg 路径' } " ^
        "else { Write-Host '  pyvenv.cfg 路径已正确，无需修改' }"
) else (
    echo  [跳过] 未找到 pyvenv.cfg
)

echo.

:: ── 2. 设置 Tcl/Tk 环境变量 ──
echo [2/5] 设置 Tcl/Tk 环境变量...
set "TCL_DIR=%ROOT%\python\tcl"

if exist "%TCL_DIR%\tcl8.6" (
    setx TCL_LIBRARY "%TCL_DIR%\tcl8.6" >nul
    echo  已设置 TCL_LIBRARY
) else (
    echo  [跳过] tcl8.6 目录不存在
)

if exist "%TCL_DIR%\tk8.6" (
    setx TK_LIBRARY "%TCL_DIR%\tk8.6" >nul
    echo  已设置 TK_LIBRARY
) else (
    echo  [跳过] tk8.6 目录不存在
)

echo.

:: ── 3. 添加到用户 PATH ──
echo [3/5] 添加程序目录到系统 PATH...
powershell -NoProfile -Command ^
    "$pathsToAdd = @('%ROOT%\python', '%ROOT%\venv\Scripts'); " ^
    "$userPath = [Environment]::GetEnvironmentVariable('Path', 'User'); " ^
    "$parts = $userPath -split ';' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }; " ^
    "$added = $false; " ^
    "foreach ($p in $pathsToAdd) { " ^
    "  if ($parts -notcontains $p) { " ^
    "    $parts += $p; " ^
    "    Write-Host ('  已添加: ' + $p); " ^
    "    $added = $true; " ^
    "  } else { " ^
    "    Write-Host ('  已存在: ' + $p); " ^
    "  } " ^
    "}; " ^
    "if ($added) { " ^
    "  [Environment]::SetEnvironmentVariable('Path', ($parts -join ';'), 'User'); " ^
    "  Write-Host '  PATH 已更新'; " ^
    "} else { " ^
    "  Write-Host '  PATH 无需修改'; " ^
    "}"

echo.

:: ── 4. 创建桌面快捷方式 ──
echo [4/5] 创建桌面快捷方式...
set "PYW_ENTRY=%ROOT%\启动中医药智能问答系统.pyw"

powershell -NoProfile -Command ^
    "$ws = New-Object -ComObject WScript.Shell; " ^
    "$desktop = [Environment]::GetFolderPath('Desktop'); " ^
    "$sc = $ws.CreateShortcut($desktop + '\中医药智能问答系统.lnk'); " ^
    "$sc.TargetPath = '%PYW_ENTRY%'; " ^
    "$sc.WorkingDirectory = '%ROOT%'; " ^
    "$sc.Description = '中医药智能问答系统'; " ^
    "$sc.Save(); " ^
    "Write-Host ('  已创建桌面快捷方式: ' + $desktop + '\中医药智能问答系统.lnk')"

echo.

:: ── 5. 备份卸载脚本到持久位置 ──
echo [5/5] 备份卸载脚本以便后续恢复...

set "BACKUP_DIR=%LOCALAPPDATA%\TCMAI"
if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"
copy /y "%~f0" "%BACKUP_DIR%\setup.bat" >nul
copy /y "%~dp0cleanup.bat" "%BACKUP_DIR%\cleanup.bat" >nul
echo  卸载脚本已备份到: %BACKUP_DIR%
echo  删除文件夹前请运行: %BACKUP_DIR%\cleanup.bat

echo.
echo ============================================
echo   环境配置完成！
echo   桌面上已添加「中医药智能问答系统」快捷方式
echo   双击即可运行
echo.
echo   如需卸载，请运行 cleanup.bat
echo ============================================
echo.
pause
