# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('models', 'models'), ('data', 'data')]
binaries = []
hiddenimports = ['sklearn', 'sklearn.preprocessing', 'sklearn.impute', 'sklearn.metrics', 'sklearn.model_selection', 'sklearn.covariance', 'scipy.stats', 'pandas', 'numpy', 'joblib', 'openpyxl', 'customtkinter', 'tkinter', 'data_loader', 'matplotlib', 'matplotlib.pyplot', 'shap']
tmp_ret = collect_all('xgboost')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['ui_app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'torchaudio', 'tensorflow'],
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
    name='WaterPredictor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=['xgboost.dll', 'vcomp140.dll'],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
