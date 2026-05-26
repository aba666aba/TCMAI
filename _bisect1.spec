# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['E:\\Tom\\TCMAI\\_bisect1.py'],
    pathex=['src', '.'],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torchvision', 'torchaudio', 'scipy', 'pandas', 'PIL', 'matplotlib', 'win32com', 'pythoncom', 'pywintypes', 'PyQt5', 'numba', 'dask', 'distributed', 'h5py', 'pyarrow', 'nltk'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='_bisect1',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    name='_bisect1',
)
