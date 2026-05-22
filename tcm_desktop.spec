# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


project_root = Path.cwd()
app_name = "TCMAI_Desktop"

datas = [
    (str(project_root / "data"), "data"),
    (str(project_root / "LICENSE"), "."),
    (str(project_root / "NOTICE"), "."),
    (str(project_root / "COPYRIGHT.txt"), "."),
]

a = Analysis(
    ["launcher.pyw"],
    pathex=[str(project_root), str(project_root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "desktop",
        "desktop.app",
        "desktop.runtime",
        "desktop.service",
        "core",
        "core.fallback",
        "core.knowledge_base",
        "core.result_verify",
        "core.symptom_analysis",
        "core.symptom_pipeline",
        "core.data_desensitize",
        "core.tcm_config",
        "tcm_recommender",
        "tcm_recommender.recommend",
        "tcm_recommender.model",
        "tkinter",
        "tkinter.ttk",
        "tkinter.scrolledtext",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=app_name,
)

