# -*- mode: python ; coding: utf-8 -*-
# ============================================================
# PyInstaller 打包配置 — 中医药智能问答系统 (TCMAI)
#
# 由 scripts/build_installer.bat 使用；若本文件不存在，该脚本会自动生成。
#
# 打包内容：
#   · src/、tcm_recommender/ 项目源码
#   · data/tcm_knowledge.db  知识库
#   · models/nlu/qwen/       NLU 模型（体积大，见下方说明）
#   · tcm_recommender/best_model.pth 推荐模型
#
# 输出：dist/TCMAI/TCMAI.exe
# ============================================================
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

# onefile=True 时合并为单个 exe；会显著变慢（每次启动需解压）
ONEFILE = os.environ.get("TCMAI_ONEFILE", "0") == "1"
# 是否把 NLU 模型一起打进去（约 1GB，会让安装包非常大）
INCLUDE_NLU_MODEL = os.environ.get("TCMAI_INCLUDE_NLU_MODEL", "1") == "1"

PROJECT_ROOT = Path(SPECPATH).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

# ---- 需要显式收集的包（含动态导入/extensions）----
hiddenimports = []
for pkg in (
    "sklearn", "sklearn.ensemble", "sklearn.tree", "sklearn.utils",
    "scipy", "scipy.sparse", "scipy.linalg", "scipy.stats", "scipy.special",
    "jieba", "networkx", "customtkinter", "PIL", "sqlite3",
    "transformers", "torch", "safetensors", "tokenizers",
):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

# ---- 需要一起打包的数据文件 ----
datas = [
    (str(SRC_DIR), "src"),                       # 源码目录（运行时通过 sys.path 导入）
    (str(PROJECT_ROOT / "tcm_recommender"), "tcm_recommender"),
]

for rel in ("data/tcm_knowledge.db", "recipes.json", "NOTICE", "LICENSE"):
    p = PROJECT_ROOT / rel
    if p.exists():
        datas.append((str(p), str(Path(rel).parent) if Path(rel).parent != Path(".") else "."))

for rel in ("tcm_recommender/best_model.pth", "tcm_recommender/best_model.json"):
    p = PROJECT_ROOT / rel
    if p.exists():
        datas.append((str(p), "tcm_recommender"))

if INCLUDE_NLU_MODEL:
    nlu_dir = PROJECT_ROOT / "models" / "nlu" / "qwen"
    if nlu_dir.exists():
        # 排除过大的训练态权重，只保留推理需要的文件
        for f in nlu_dir.iterdir():
            if f.is_file() and f.suffix not in (".bin", ".h5", ".msgpack", ".ot"):
                datas.append((str(f), "models/nlu/qwen"))

# 中文字体：存在则一并打包，避免打包后界面中文显示为方块
for font in ("msyh.ttc", "msyh.ttf", "simhei.ttf"):
    fp = PROJECT_ROOT / font
    if fp.exists():
        datas.append((str(fp), "."))

block_cipher = None

a = Analysis(
    [str(SRC_DIR / "main.py")],
    pathex=[str(PROJECT_ROOT), str(SRC_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib", "pandas", "notebook", "IPython", "pytest",
        "tkinter.test", "test", "unittest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if ONEFILE:
    exe = EXE(
        pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [],
        name="TCMAI",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,          # 无控制台窗口（GUI 程序）
        disable_windowed_traceback=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
else:
    exe = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name="TCMAI",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe, a.binaries, a.zipfiles, a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name="TCMAI",
    )
