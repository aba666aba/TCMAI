@echo off
chcp 65001 >nul
title 中医药智能问答系统 - 环境恢复

setlocal enabledelayedexpansion

echo ============================================
echo   中医药智能问答系统 - 环境恢复
echo   撤销 setup.bat 的所有更改
echo.
echo   请在删除程序文件夹之前运行此脚本
echo ============================================
echo.

:: ── 1. 从 PATH 移除程序目录 ──
echo [1/3] 从系统 PATH 中移除程序目录...

powershell -NoProfile -Command ^
    "$userPath = [Environment]::GetEnvironmentVariable('Path', 'User'); " ^
    "if ($userPath) { " ^
    "  $parts = $userPath -split ';' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }; " ^
    "  $filtered = $parts | Where-Object { " ^
    "    $_ -notmatch '\\\\python$' -and $_ -notmatch '\\\\venv\\\\Scripts$' " ^
    "  }; " ^
    "  if ($filtered.Count -ne $parts.Count) { " ^
    "    [Environment]::SetEnvironmentVariable('Path', ($filtered -join ';'), 'User'); " ^
    "    Write-Host '  已从 PATH 中移除程序目录'; " ^
    "  } else { " ^
    "    Write-Host '  PATH 中未找到程序目录'; " ^
    "  } " ^
    "}"

echo.

:: ── 2. 删除环境变量 ──
echo [2/3] 删除 Tcl/Tk 环境变量...

for %%v in (TCL_LIBRARY TK_LIBRARY) do (
    reg delete "HKCU\Environment" /v %%v /f >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo  已删除 %%v
    ) else (
        echo  [跳过] %%v 不存在
    )
)

:: 通知系统环境变量已更改
powershell -NoProfile -Command "[Environment]::SetEnvironmentVariable('TCL_LIBRARY', $null, 'User')" >nul 2>&1
powershell -NoProfile -Command "[Environment]::SetEnvironmentVariable('TK_LIBRARY', $null, 'User')" >nul 2>&1

echo.

:: ── 3. 删除桌面快捷方式 ──
echo [3/3] 删除桌面快捷方式...

set "SHORTCUT=%USERPROFILE%\Desktop\中医药智能问答系统.lnk"
if exist "%SHORTCUT%" (
    del /f /q "%SHORTCUT%" >nul 2>&1
    echo  已删除桌面快捷方式
) else (
    echo  [跳过] 桌面快捷方式不存在
)

:: 也检查是否存在中文用户名路径
set "SHORTCUT2=%PUBLIC%\Desktop\中医药智能问答系统.lnk"
if exist "%SHORTCUT2%" (
    del /f /q "%SHORTCUT2%" >nul 2>&1
    echo  已删除公共桌面快捷方式
)

echo.
echo ============================================
echo   环境恢复完成！
echo   所有 setup.bat 的更改已撤销
echo   现在可以安全删除程序文件夹
echo ============================================
echo.
pause
