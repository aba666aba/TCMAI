# -*- mode: python ; coding: utf-8 -*-
#
# 中医药智能问答系统 - PyInstaller 打包配置
# 运行：pyinstaller tcmai.spec

import sys
from pathlib import Path

# 项目根目录（打包脚本已 cd 到项目目录）
root = Path.cwd()

# ── 数据文件清单 ──
# (源路径, 目标目录)
datas = [
    # 核心数据
    (str(root / "data" / "tcm_knowledge.db"), "data"),
    (str(root / "data" / "train_data.npz"), "data"),
    # 深度学习模型
    (str(root / "tcm_recommender" / "best_model.pth"), "tcm_recommender"),
    (str(root / "tcm_recommender" / "best_model.json"), "tcm_recommender"),
    # 扩展配置（被 recommend.py 和 data_pipeline.py 引用）
    (str(root / "tcm_recommender" / "extended_config.py"), "tcm_recommender"),
    # 快速启动脚本（打包后放入根目录）
    (str(root / "scripts" / "快速启动.bat"), "."),
]

# ── 源码目录（作为包加入）──
# PyInstaller 会自动扫描 imports，但 src/ 是动态路径加载的，需手动加入
src_core = root / "src" / "core"
src_desktop = root / "src" / "desktop"
tcm_rec = root / "tcm_recommender"

# 收集 src/core/ 下的所有 .py
for f in sorted(src_core.glob("*.py")):
    datas.append((str(f), "src/core"))
# 收集 src/desktop/ 下的所有 .py
for f in sorted(src_desktop.glob("*.py")):
    datas.append((str(f), "src/desktop"))
# 收集 tcm_recommender/ 下的 .py（不包含数据文件）
for f in sorted(tcm_rec.glob("*.py")):
    datas.append((str(f), "tcm_recommender"))

# ── 隐式导入 ──
# PyInstaller 无法自动检测到的动态/延迟加载
hiddenimports = [
    # 项目模块
    "tcm_recommender",
    "tcm_recommender.model",
    "tcm_recommender.train",
    "tcm_recommender.recommend",
    "tcm_recommender.data_pipeline",
    "tcm_recommender.data_expansion",
    "tcm_recommender.extended_config",
    "core",
    "core.tcm_config",
    "core.symptom_pipeline",
    "core.symptom_analysis",
    "core.fallback",
    "core.knowledge_base",
    "core.result_verify",
    "core.data_desensitize",
    "desktop",
    "desktop.app",
    "desktop.service",
    # 第三方依赖
    "torch",
    "torch.nn",
    "torch.nn.modules",
    "torch.optim",
    "torch.utils",
    "torch.cuda",
    "torch.amp",
    "numpy",
    "tqdm",
    "concurrent.futures",
    "queue",
    "threading",
    "json",
    "logging",
    "logging.handlers",
    "sqlite3",
    "tkinter",
    "tkinter.scrolledtext",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "tkinter.ttk",
    "pathlib",
    "dataclasses",
    "hashlib",
    "re",
    "datetime",
    "tempfile",
    "contextlib",
    "platform",
]

# ── 排除不需要的大型库 ──
excludes = [
    "torchvision",     # 未使用，节省 ~600MB
    "matplotlib",      # 未使用
    "scipy",           # 未使用
    "pandas",          # 未使用
    "PIL",             # 未使用
    "cv2",             # 未使用
    "tensorboard",     # 未使用
]

block_cipher = None

a = Analysis(
    [str(root / "launcher.py")],
    pathex=[str(root), str(root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="中医药智能问答系统",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

# ── 目录模式（onedir）──
# torch+cu124 约 2GB，无法使用 onefile 模式
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TCMAI_Desktop",
)
