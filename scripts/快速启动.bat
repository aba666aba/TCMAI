@echo off
chcp 65001 >nul
cd /d "%~dp0.."
start "" pythonw "知识图谱中药方生成系统.pyw"
exit
