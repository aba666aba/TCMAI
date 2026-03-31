@echo off
chcp 65001 >nul
title 中医药智能问答系统 - 快速启动

echo ========================================
echo     中医药智能问答系统 - 快速启动
echo ========================================
echo.

cd /d "%~dp0"

REM 直接启动，显示所有输出
python src\main.py

pause
