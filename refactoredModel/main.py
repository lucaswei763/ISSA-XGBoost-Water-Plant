#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
水厂投矾量数据分析与建模系统 (终极平滑 + 异常剔除 + ISSA版)
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings

try:
    from issa_optimizer import ISSA_XGBoost
except ImportError:
    print("❌ 致命错误: 找不到 issa_optimizer.py 文件！")
    sys.exit(1)

warnings.filterwarnings('ignore')

required_dirs = ['data', 'models', 'outputs']
for dir_name in required_dirs:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.figsize'] = (14, 7)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

print("=" * 80)
print("水厂投矾量智能建模系统 (最终打包准备版)")
print("=" * 80)


def load_data():
    db_path = 'data/water_data.db'
    if not os.path.exists(db_path):
        sys.exit("错误：数据库文件不存在")

    conn = sqlite3.connect(db_path)
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
    df = df.sort_values(by='日期').reset_index(drop=True)

    # 截断远古数据
    cutoff_date = df['日期'].max() - pd.Timedelta(days=900)
    df = df[df['日期'] >= cutoff_date].reset_index(drop=True)
    return df


df = load_data()

target_col = next(
    (c for c in df.columns if any(kw in c.lower() for kw in ['矾', 'alum', '投矾', '药耗']) and 'lag' not in c.lower()),
    None)
if not target_col:
    sys.exit("错误：无法识别目标变量")

df_clean = df.copy()

# 👑 终极修复：物理极限异常值剔除 (拔掉10000kg的通天柱)
original_len = len(df_clean)
df_clean = df_clean[df_clean[target_col] <= 5000].reset_index(drop=True)
print(f"  [数据清洗] 剔除了 {original_len - len(df_clean)} 条超出物理极限的异常脏数据！")

numeric_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c != '日期']
if numeric_cols:
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

new_cols = [c for c in df_clean.columns if c not in df.columns]
if new_cols:
    df_clean[new_cols] = df_clean[new_cols].fillna(method='ffill').fillna(0)

exclude_patterns = ['日期', '星期']
feature_candidates = [c for c in df_clean.columns if
                      c != target_col and c != '日期' and df_clean[c].dtype in ['int64', 'float64'] and not any(
                          p in c for p in exclude_patterns)]

corr_results = []
for col in feature_candidates:
    valid = df_clean[[target_col, col]].dropna()
    if len(valid) > 10:
        spearman_rho, _ = spearmanr(valid[target_col], valid[col])
        corr_results.append({'特征': col, '斯皮尔曼_ρ': spearman_rho})

selected_features = feature_candidates
if corr_results:
    corr_df = pd.DataFrame(corr_results).sort_values('斯皮尔曼_ρ', key=abs, ascending=False)
    selected_features = corr_df[abs(corr_df['斯皮尔曼_ρ']) >= 0.2]['特征'].tolist()

essential_keywords = ['平滑', 'lag_1天', '交互', '温度', 'ph']
for col in df_clean.columns:
    if any(k in col.lower() for k in
           essential_keywords) and col not in selected_features and col != target_col and col != '日期':
        selected_features.append(col)

noisy_cols = [c for c in [turb_col, flow_col] if c is not None]
final_features = list(set([f for f in selected_features if f not in noisy_cols]))

X = df_clean[final_features].copy()
y = df_clean[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)
sample_weights_train = np.linspace(0.1, 1.0, len(y_train))

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(final_features, 'models/selected_features.pkl')

print("\n【步骤6/6】模型训练与评估...")

issa = ISSA_XGBoost(pop_size=10, max_iter=10, cv_splits=CV_FOLDS)
best_xgb_params = issa.optimize(X_train_scaled, y_train.values)

from xgboost import XGBRegressor

best_xgb = XGBRegressor(
    **best_xgb_params,
    min_child_weight=7,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0
)
best_xgb.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)

r2 = r2_score(y_test, best_xgb.predict(X_test_scaled))

print(f"\n{'=' * 60}")
print(f"🥇 最终评估: ISSA-XGBoost (测试集R²: {r2:.4f})")
print(f"{'=' * 60}")

import json

metadata = {
    'model_type': 'XGBoost',
    'version': '4.0_Final_Cleaned',
    'features': final_features,
    'target_col': target_col,
    'model_file': 'best_model.pkl',
    'scaler_path': 'scaler.pkl'
}
with open('models/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)
joblib.dump(best_xgb, 'models/best_model.pkl')

print("✅ 终极版训练完成！可以开始打包了！")