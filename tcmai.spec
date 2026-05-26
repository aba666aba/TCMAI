# -*- mode: python ; coding: utf-8 -*-
#
# 中医药智能问答系统 - PyInstaller 打包配置
# 运行：pyinstaller --clean tcmai.spec

import sys
from pathlib import Path

# 项目根目录（打包脚本已 cd 到项目目录）
root = Path.cwd()

# ── 数据文件清单（仅数据，不包含源码 .py 文件）──
# 编译模块由 hiddenimports 自动处理
datas = [
    # 核心数据
    (str(root / "data" / "tcm_knowledge.db"), "data"),
    (str(root / "data" / "kg_cache.pkl"), "data"),
    # 推荐模型（已训练好的 .pth）
    (str(root / "tcm_recommender" / "best_model.pth"), "tcm_recommender"),
    # 快速启动脚本
    (str(root / "scripts" / "快速启动.bat"), "."),
    # NLU 模型（Qwen2.5 离线）
    (str(root / "models" / "nlu" / "qwen" / "model.safetensors"), "models/nlu/qwen"),
    (str(root / "models" / "nlu" / "qwen" / "config.json"), "models/nlu/qwen"),
    (str(root / "models" / "nlu" / "qwen" / "generation_config.json"), "models/nlu/qwen"),
    (str(root / "models" / "nlu" / "qwen" / "tokenizer_config.json"), "models/nlu/qwen"),
    (str(root / "models" / "nlu" / "qwen" / "tokenizer.json"), "models/nlu/qwen"),
    (str(root / "models" / "nlu" / "qwen" / "vocab.json"), "models/nlu/qwen"),
    (str(root / "models" / "nlu" / "qwen" / "merges.txt"), "models/nlu/qwen"),
    (str(root / "models" / "nlu" / "qwen" / "configuration.json"), "models/nlu/qwen"),
]

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
    "nlu",
    "nlu.engine",
    "core",
    "core.tcm_config",
    "core.symptom_pipeline",
    "core.symptom_analysis",
    "core.knowledge_base",
    "core.result_verify",
    "core.data_desensitize",
    "core.intelligent_search",
    "desktop",
    "desktop.app",
    "desktop.service",
    "desktop.splash",
    "desktop.kg_view",
    "desktop.runtime",
    # 第三方依赖
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
    # torch / transformers
    "torch",
    "transformers",
    "bitsandbytes",
    "accelerate",
]

# ── 排除不需要的大型库 ──
excludes = [
    "torchvision", "torchaudio",     # 未使用
    "scipy", "pandas", "PIL", "cv2",  # 未使用
    "tensorboard", "matplotlib",      # 未使用
    "PyQt5", "PySide2", "PySide6",    # 用 tkinter，不用 qt
    "sklearn", "scikit-learn",        # 未使用
    "notebook", "jupyter", "ipykernel",  # jupyter 全家桶
    "plotly", "bokeh", "altair",       # 可视化库
    "dask", "distributed",             # 分布式计算
    "bottleneck", "numexpr",           # numpy 加速器
    "numba", "llvmlite",               # JIT
    "h5py",                            # HDF5 文件
    "sqlalchemy",                      # ORM
    "nltk",                            # NLP
    "statsmodels",                     # 统计
    "scikit-image", "skimage",         # 图像处理
    "xarray",                          # 多维数组
    "patsy",                           # 统计公式
    "win32com", "pythoncom", "pywintypes",  # COM 组件
    "altair", "intake", "nbconvert",   # 数据可视化/转换
    "pyarrow", "lz4",                  # 数据格式
    "datasets",                        # 训练用，非推理所需
    # ── 加速排除：hooks-contrib 大量无关库 hook（确定不用的）──
    "IPython", "jedi", "parso",        # IPython 交互式（～50MB）
    "sphinx", "docutils",              # 文档生成（～30MB）
    "black", "yapf", "blib2to3",       # 代码格式化
    "pygments",                        # 代码高亮
    "pytest", "py",                    # 测试框架
    "zmq",                             # ZeroMQ（～10MB DLL）
    "lxml",                            # XML 处理（～15MB）
    "bcrypt",                          # 密码哈希
    "astroid",                         # Python AST 工具
    "pydantic",                        # 数据验证（未使用）
    "rich",                            # 终端富文本
    "markdown",                        # Markdown 渲染
    "panel", "pyviz_comms", "xyzservices",  # panel 可视化
    "bokeh",                           # 可视化
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
    [],
    a.zipfiles,
    [],
    [],
    name="中医药智能问答系统",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                  # 关闭 UPX，避免 PyTorch DLL 解压耗时
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
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TCMAI_Desktop",
)
