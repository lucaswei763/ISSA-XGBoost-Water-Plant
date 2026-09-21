#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文件名称：refactoredModel/generate_report_metrics.py
所属类别：重构核心生产代码 (Refactored Core Production)

功能描述：
    水厂投药量预测模型在专业技术汇报、科技成果鉴定以及 PPT 演示时所需的“亮点指标与工业落地指标”一键生成计算工具。
    主要职责：
    1. 自动连接并读取本地 SQLite 数据库中的水务记录；
    2. 自动载入最新模型特征与权重，进行最近两年半工况的数据前向推断；
    3. 统计两套核心指标体系：
       - 统计学拟合指标：包括 R² 决定系数、MAE、RMSE；
       - 业务可落地指标：测算误差分别在 ±10%、±15%、±20% 以内的天数占比（对应工业现场加矾容忍阈值）；
    4. 在控制台直接输出科技创新亮点与指标汇总看板。

运行与使用方法：
    直接在控制台执行：
    python generate_report_metrics.py

调用与依赖关系：
    - 导入并使用 `utils` 中的特征工程列表与扩展函数。
    - 用于在向水厂领导层或专家组做项目汇报时，提供客观量化的数据证明。
"""

import sqlite3
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=> 正在加载模型与最新数据计算汇报指标...\n")

# 1. 加载数据与模型
conn = sqlite3.connect('data/水务数据中心.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [t[0] for t in cursor.fetchall()]

if 'water_records' in tables:
    df = pd.read_sql_query("SELECT * FROM water_records", conn)
elif 'daily_records' in tables:
    df = pd.read_sql_query("SELECT * FROM daily_records", conn)
else:
    df = pd.read_sql_query(f"SELECT * FROM {tables[0]}", conn)

date_col = next((c for c in df.columns if '日期' in c or 'date' in c.lower()), df.columns[0])
df['日期'] = pd.to_datetime(df[date_col], format='%Y年%m月%d日' if '年' in str(df[date_col].iloc[0]) else None)
conn.close()

df = df.sort_values(by='日期').reset_index(drop=True)

# 截断旧时代数据，只评估当前规则
cutoff_date = df['日期'].max() - pd.Timedelta(days=900)
df = df[df['日期'] >= cutoff_date].reset_index(drop=True)

from utils import BASE_FEATURE_COLS, TARGET_COL, add_engineered_features

target_col = TARGET_COL
df = df[df[target_col] <= 5000].reset_index(drop=True) # 剔除异常值

# Clean invalid characters to nan
df_clean = df.copy()
df_clean.replace(['/', '\\', '', ' '], np.nan, inplace=True)
df_clean[BASE_FEATURE_COLS + [target_col]] = df_clean[BASE_FEATURE_COLS + [target_col]].apply(pd.to_numeric, errors='coerce')
df_clean = df_clean.dropna(subset=[target_col]).copy()

# Load model artifacts
import os
model_dir = 'models/resnet' if os.path.exists('models/resnet/best_model.pkl') else ('models/xgboost' if os.path.exists('models/xgboost/best_model.pkl') else 'models')
imputer = joblib.load(os.path.join(model_dir, 'imputer.pkl'))
scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
selected_features = joblib.load(os.path.join(model_dir, 'selected_features.pkl'))
model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))

# Impute using median values from training
df_clean[BASE_FEATURE_COLS] = imputer.transform(df_clean[BASE_FEATURE_COLS])

# Generate features
df_clean = add_engineered_features(df_clean)

X = df_clean[selected_features].copy()
y_true = df_clean[target_col].values

X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

# 计算指标
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

# 计算业务指标：误差百分比
errors = y_pred - y_true
abs_percent_errors = np.abs(errors / y_true) * 100
within_10_pct = np.mean(abs_percent_errors <= 10) * 100
within_15_pct = np.mean(abs_percent_errors <= 15) * 100
within_20_pct = np.mean(abs_percent_errors <= 20) * 100

print("=" * 60)
print("🏆 水厂投矾量预测模型 - 核心汇报指标")
print("=" * 60)
print(f"📌 1. 统计学拟合精度 (基于近两年半数据)")
print(f"   - 决定系数 (R²): {r2:.4f}  (>0.8表示极高解释力)")
print(f"   - 平均绝对误差 (MAE): {mae:.2f} kg/天")
print(f"   - 均方根误差 (RMSE): {rmse:.2f} kg/天")
print("-" * 60)
print(f"🎯 2. 工业落地指标 (业务容忍度)")
print(f"   - 预测误差在 ±10% 以内的天数占比: {within_10_pct:.1f}%")
print(f"   - 预测误差在 ±15% 以内的天数占比: {within_15_pct:.1f}%")
print(f"   - 预测误差在 ±20% 以内的天数占比: {within_20_pct:.1f}%")
print("-" * 60)
print(f"🛡️ 3. 模型核心创新点 (亮点总结)")
print("   1. 引入 ISSA (改进麻雀算法) 进行超参数寻优，跳出传统网格搜索的局部最优陷阱。")
print("   2. 采用 TimeSeriesSplit 严格时间序列交叉验证，杜绝未来数据泄露，指标真实可靠。")
print("   3. 设计『滑动平滑避震器』，屏蔽传感器毛刺脏数据，彻底消灭了预测值的剧烈锯齿跳动。")
print("   4. 动态样本权重衰减机制，成功克服了水厂加药规则的『概念漂移(Concept Drift)』问题。")
print("=" * 60)