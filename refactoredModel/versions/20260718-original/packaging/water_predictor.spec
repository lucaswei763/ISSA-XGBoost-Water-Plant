# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('models', 'models'), ('data', 'data')]
binaries = []
hiddenimports = [
    'torch',
    'torch.nn',
    'torch.utils.data',
    'xgboost',
    'sklearn',
    'sklearn.preprocessing',
    'sklearn.metrics',
    'pandas',
    'numpy',
    'matplotlib',
    'matplotlib.pyplot',
    'joblib',
    'openpyxl',
    'customtkinter',
    'tkinter',
    'lstm_xgb_stacked_v2'
]

# Collect customtkinter data
tmp_ret = collect_all('customtkinter')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

a = Analysis(
    ['ui_app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    a.binaries,
    a.datas,
    [],
    name='\u6c34\u5382\u6295\u77fe\u91cf\u9884\u6d4b\u7cfb\u7edf',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

