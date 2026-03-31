@echo off
chcp 65001 >nul
title 打包中...

echo 开始打包，请稍候...
python -m PyInstaller --name="中医药智能问答系统" --windowed --onefile --add-data "src\web\templates;src\web\templates" --add-data "data;data" --hidden-import flask --hidden-import sqlite3 launcher.py

if exist "dist\中医药智能问答系统.exe" (
    echo.
    echo 打包成功！
    echo 文件位置：dist\中医药智能问答系统.exe
) else (
    echo.
    echo 打包失败，请检查错误信息
)

pause
