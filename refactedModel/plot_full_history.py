#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
水厂投矾量：2021-2026 逐年全量历史数据预测与对比
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys

# 设置 Mac/Windows 中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=> 正在加载完整数据库...")
conn = sqlite3.connect('data/water_data.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [t[0] for t in cursor.fetchall()]

if 'merged_data' in tables:
    df = pd.read_sql_query("SELECT * FROM merged_data", conn)
    date_col = next((c for c in df.columns if '日期' in c or 'date' in c.lower()), df.columns[0])
    df['日期'] = pd.to_datetime(df[date_col])
else:
    consumption_table = next((t for t in tables if 'consumption' in t.lower() or '药耗' in t), tables[0])
    quality_table = next((t for t in tables if 'quality' in t.lower() or '水质' in t),
                         tables[1] if len(tables) > 1 else tables[0])
    df_c = pd.read_sql_query(f"SELECT * FROM {consumption_table}", conn)
    df_q = pd.read_sql_query(f"SELECT * FROM {quality_table}", conn)
    dc_c = next((c for c in df_c.columns if '日期' in c or 'date' in c.lower()), df_c.columns[0])
    dc_q = next((c for c in df_q.columns if '日期' in c or 'date' in c.lower()), df_q.columns[0])
    df_c[dc_c] = pd.to_datetime(df_c[dc_c])
    df_q[dc_q] = pd.to_datetime(df_q[dc_q])
    df = pd.merge(df_c, df_q, left_on=dc_c, right_on=dc_q, how='inner')
    df['日期'] = df[dc_c]
conn.close()

# 严格按时间排序
df = df.sort_values(by='日期').reset_index(drop=True)
print(f"✓ 已加载全量数据: {df['日期'].min().date()} 至 {df['日期'].max().date()} (共 {len(df)} 条)")

target_col = next(
    (c for c in df.columns if any(kw in c.lower() for kw in ['矾', 'alum', '投矾', '药耗']) and 'lag' not in c.lower()),
    None)

df = df[df[target_col] <= 5000].reset_index(drop=True)

# ==================== 复刻特征平滑工程 ====================
print("=> 正在复刻特征平滑工程...")
df_clean = df.copy()
numeric_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c != '日期']
df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill').fillna(df_clean[numeric_cols].median())

df_clean['年'] = df_clean['日期'].dt.year
df_clean['月'] = df_clean['日期'].dt.month
df_clean['星期几'] = df_clean['日期'].dt.dayofweek
df_clean['是否为周末'] = (df_clean['星期几'] >= 5).astype(int)

for lag in [1, 2, 3]:
    df_clean[f'{target_col}_lag_{lag}天'] = df_clean[target_col].shift(lag)

turb_col = next((c for c in df_clean.columns if '浊度' in c or 'turbidity' in c.lower()), None)
flow_col = next((c for c in df_clean.columns if '流量' in c or 'flow' in c.lower() or 'supply' in c.lower()), None)

if turb_col:
    df_clean[f'{turb_col}_3天平滑'] = df_clean[turb_col].rolling(3, min_periods=1).mean()
    df_clean[f'{turb_col}_7天平滑'] = df_clean[turb_col].rolling(7, min_periods=1).mean()
if flow_col:
    df_clean[f'{flow_col}_3天平滑'] = df_clean[flow_col].rolling(3, min_periods=1).mean()
    df_clean[f'{flow_col}_7天平滑'] = df_clean[flow_col].rolling(7, min_periods=1).mean()
if turb_col and flow_col:
    df_clean['平滑后浊度_流量交互'] = df_clean[f'{turb_col}_3天平滑'] * df_clean[f'{flow_col}_3天平滑']

df_clean = df_clean.fillna(method='bfill').fillna(0)

# ==================== 加载模型并预测 ====================
print("=> 正在加载最新模型进行预测...")
try:
    scaler = joblib.load('models/scaler.pkl')
    selected_features = joblib.load('models/selected_features.pkl')
    model = joblib.load('models/best_model.pkl')
except Exception as e:
    print(f"❌ 模型加载失败，请确保已经运行过最新的 main.py。报错: {e}")
    sys.exit(1)

X_full = df_clean[selected_features].copy()
y_true = df_clean[target_col].values

X_full_scaled = scaler.transform(X_full)
y_pred = model.predict(X_full_scaled)

# ==================== 分年份独立绘图 ====================
print("=> 正在按年份拆分绘图...")
years = df_clean['年'].unique()
n_years = len(years)

# 动态设置画布高度，每年分给 4.5 的高度
fig, axes = plt.subplots(n_years, 1, figsize=(16, 4.5 * n_years))
if n_years == 1:
    axes = [axes]

# 找出那条 900 天前的数据截断线
cutoff_date = df_clean['日期'].max() - pd.Timedelta(days=900)

for i, year in enumerate(years):
    ax = axes[i]
    mask = df_clean['年'] == year
    df_year = df_clean[mask]
    y_true_year = y_true[mask]
    y_pred_year = y_pred[mask]

    ax.plot(df_year['日期'], y_true_year, 'b-', linewidth=1.5, label='实际投矾量', alpha=0.8)
    ax.plot(df_year['日期'], y_pred_year, 'r-', linewidth=2.0, label='模型预测值 (基于近期规则)', alpha=0.8)

    # 检查分界线是否落在这一个年份里，如果是，就画出来
    start_of_year = pd.Timestamp(year=year, month=1, day=1)
    end_of_year = pd.Timestamp(year=year, month=12, day=31)
    if start_of_year <= cutoff_date <= end_of_year:
        ax.axvline(x=cutoff_date, color='green', linestyle='--', linewidth=3,
                   label='训练数据截断线 (右侧为当前模型认知区)')

    ax.set_title(f'【{year}年】 投矾量预测与实际对比', fontsize=14, fontweight='bold')
    ax.set_ylabel('投矾量 (kg)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

plt.tight_layout()
output_path = 'outputs/yearly_history_comparison.png'
plt.savefig(output_path, dpi=150)
print(f"✅ 逐年全景图已保存至: {output_path}")
plt.show()