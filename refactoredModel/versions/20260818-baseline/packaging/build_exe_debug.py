import os
import sys

custom_cache_dir = os.path.join(os.getcwd(), 'pyinstaller_cache')
os.makedirs(custom_cache_dir, exist_ok=True)

os.environ['PYINSTALLER_CONFIG_DIR'] = custom_cache_dir

from PyInstaller.__main__ import run

sys.argv = [
    'pyinstaller',
    '--clean', '--onefile', '--console',
    '--name', 'WaterPredictor_Debug',
    '--add-data', 'models;models',
    '--add-data', 'data;data',
    '--collect-all', 'xgboost',
    '--exclude-module', 'torch',
    '--exclude-module', 'torchvision',
    '--exclude-module', 'torchaudio',
    '--exclude-module', 'tensorflow',
    '--exclude-module', 'numba',
    '--hidden-import', 'sklearn',
    '--hidden-import', 'sklearn.preprocessing',
    '--hidden-import', 'sklearn.impute',
    '--hidden-import', 'sklearn.metrics',
    '--hidden-import', 'sklearn.model_selection',
    '--hidden-import', 'sklearn.covariance',
    '--hidden-import', 'scipy.stats',
    '--hidden-import', 'pandas',
    '--hidden-import', 'numpy',
    '--hidden-import', 'joblib',
    '--hidden-import', 'openpyxl',
    '--hidden-import', 'customtkinter',
    '--hidden-import', 'tkinter',
    '--hidden-import', 'data_loader',
    '--hidden-import', 'matplotlib',
    '--hidden-import', 'matplotlib.pyplot',
    '--upx-exclude', 'xgboost.dll',
    '--upx-exclude', 'vcomp140.dll',
    '--workpath', 'build_debug',
    '--distpath', 'dist_debug',
    'ui_app.py'
]

run()