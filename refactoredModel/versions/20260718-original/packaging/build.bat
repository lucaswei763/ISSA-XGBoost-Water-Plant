@echo off
echo ========================================
echo 打包水厂投矾量预测系统
echo ========================================

REM 安装打包工具
pip install pyinstaller

REM 清理旧文件
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

REM 打包 - 使用您原来的方式
pyinstaller --onefile ^
    --name "水厂投矾量预测系统" ^
    --add-data "models;models" ^
    --add-data "data;data" ^
    --hidden-import torch ^
    --hidden-import torch.nn ^
    --hidden-import torch.utils.data ^
    --hidden-import xgboost ^
    --hidden-import sklearn ^
    --hidden-import sklearn.preprocessing ^
    --hidden-import sklearn.metrics ^
    --hidden-import pandas ^
    --hidden-import numpy ^
    --hidden-import matplotlib ^
    --hidden-import matplotlib.pyplot ^
    --hidden-import joblib ^
    --hidden-import openpyxl ^
    --hidden-import customtkinter ^
    --hidden-import tkinter ^
    --hidden-import lstm_xgb_stacked_v4 ^
    ui_app.py

echo.
echo ✅ 打包完成！可执行文件位于: dist/水厂投矾量预测系统.exe
pause