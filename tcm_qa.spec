# -*- mode: python ; coding: utf-8 -*-

"""
中医药智能问答系统 - PyInstaller 打包配置
使用方法：pyinstaller tcm_qa.spec
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath('__file__'))
src_path = os.path.join(PROJECT_ROOT, 'src')

block_cipher = None

# 收集所有数据文件
datas = [
    # 模板文件
    (os.path.join(src_path, 'web', 'templates'), 'src\\web\\templates'),
    # 静态文件
    (os.path.join(src_path, 'web', 'templates', 'static'), 'src\\web\\templates\\static'),
    # 数据库
    (os.path.join(PROJECT_ROOT, 'data'), 'data'),
    # 文档
    (os.path.join(PROJECT_ROOT, 'docs'), 'docs'),
]

# 收集隐藏导入
hiddenimports = [
    'flask',
    'sqlite3',
    'core.knowledge_base',
    'core.symptom_analysis',
    'core.result_verify',
    'core.fallback',
    'core.data_desensitize',
    'web.app',
]

a = Analysis(
    ['launcher.py'],
    pathex=[PROJECT_ROOT, src_path],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='中医药智能问答系统',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # 不显示控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 可以添加图标：icon='icon.ico'
)
