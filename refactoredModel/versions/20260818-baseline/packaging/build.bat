@echo off
chcp 65001 >nul
echo ========================================
echo 水厂投矾量预测系统 - 打包脚本
echo ========================================
echo.

REM 安装打包工具
pip install pyinstaller -q

REM 清理旧文件
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

echo [1/3] 开始打包...
pyinstaller --onefile --noconsole ^
    --name "水厂投矾量预测系统" ^
    --add-data "models;models" ^
    --add-data "data;data" ^
    --collect-all xgboost ^
    --collect-all shap ^
    --exclude-module torch ^
    --exclude-module torchvision ^
    --exclude-module torchaudio ^
    --exclude-module tensorflow ^
    --exclude-module numba ^
    --hidden-import sklearn ^
    --hidden-import sklearn.preprocessing ^
    --hidden-import sklearn.impute ^
    --hidden-import sklearn.metrics ^
    --hidden-import sklearn.model_selection ^
    --hidden-import sklearn.covariance ^
    --hidden-import scipy.stats ^
    --hidden-import pandas ^
    --hidden-import numpy ^
    --hidden-import joblib ^
    --hidden-import openpyxl ^
    --hidden-import customtkinter ^
    --hidden-import tkinter ^
    --hidden-import data_loader ^
    --hidden-import matplotlib ^
    --hidden-import matplotlib.pyplot ^
    ui_app.py

if %errorlevel% neq 0 (
    echo [X] 打包失败！
    pause
    exit /b 1
)

echo [2/3] 清理编译残留...
if exist build rmdir /s /q build
del /q *.spec 2>nul

echo [3/3] 验证输出...
if exist "dist\水厂投矾量预测系统.exe" (
    for %%A in ("dist\水厂投矾量预测系统.exe") do (
        set /a size_mb=%%~zA/1048576
        echo [OK] 打包成功！
        echo      文件: dist\水厂投矾量预测系统.exe
        echo      大小: !size_mb! MB
    )
) else (
    echo [X] exe文件未生成！
)

echo.
echo ========================================
echo 打包完成！测试步骤：
echo 1. 关闭本窗口
echo 2. 双击 dist\水厂投矾量预测系统.exe
echo 3. 等待10-20秒启动
echo 4. 输入测试数据点"开始预测"
echo 5. 检查exe旁边是否生成 data\runtime_records.json
echo ========================================
pause
