# -*- coding: utf-8 -*-
"""TCMAI 桌面启动器——双击即启动中医药智能问答系统（无控制台窗口）。"""
import os
import subprocess
import sys

ROOT = r"E:\AAATOM\NewLai\TCMAI"
PYW = os.path.join(ROOT, "启动中医药智能问答系统.pyw")
PYTHONW = r"D:\Python\pythonw.exe"

if not os.path.exists(PYTHONW):
    # 兜底：从 PATH 找 pythonw
    for cand in (r"D:\Python\pythonw.exe", r"C:\Python312\pythonw.exe"):
        if os.path.exists(cand):
            PYTHONW = cand
            break

subprocess.Popen([PYTHONW, PYW], cwd=ROOT, close_fds=True)
sys.exit(0)
