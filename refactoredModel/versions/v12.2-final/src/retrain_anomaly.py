import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope
import joblib

from data_loader import load_raw_data, clean_data

print("=" * 50)
print("重新训练异常检测器")
print("=" * 50)

df = load_raw_data()
df, _ = clean_data(df)

# 使用5个关键特征
anomaly_features = ['浑浊度_0点', '原水量_Km3', '温度_C', '氨氮_mg_per_L', 'pH值']

# 确保特征存在
for f in anomaly_features:
    if f not in df.columns:
        print(f"⚠️ {f} 不在数据中")

# 提取特征并处理NaN
X = df[anomaly_features].values
# 删除NaN行
mask = ~np.isnan(X).any(axis=1)
X = X[mask]
print(f"处理NaN后数据: {X.shape}")

# 训练异常检测器
print(f"训练数据: {X.shape}")
anomaly_detector = EllipticEnvelope(
    contamination=0.02, 
    random_state=42,
    support_fraction=0.95
)
anomaly_detector.fit(X)

# 计算阈值（取异常分数的第98百分位数）
scores = anomaly_detector.decision_function(X)
threshold = np.percentile(scores, 2)  # 2%最异常的数据
print(f"异常阈值: {threshold}")

# 保存模型
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

joblib.dump(anomaly_detector, os.path.join(model_dir, 'anomaly_detector.pkl'))
joblib.dump(anomaly_features, os.path.join(model_dir, 'anomaly_features.pkl'))

# 保存特征统计信息
feat_stats = {}
for f in anomaly_features:
    if f in df.columns:
        feat_stats[f] = {
            'mean': float(df[f].mean()),
            'std': float(df[f].std()),
            'min': float(df[f].min()),
            'max': float(df[f].max())
        }
joblib.dump(feat_stats, os.path.join(model_dir, 'feature_stats.pkl'))

# 保存特征范围（JSON格式）
import json
feat_ranges = {}
for f in anomaly_features:
    if f in df.columns:
        feat_ranges[f] = {
            'min': float(df[f].min()),
            'max': float(df[f].max()),
            'mean': float(df[f].mean()),
            'std': float(df[f].std())
        }
with open(os.path.join(model_dir, 'feature_ranges.json'), 'w', encoding='utf-8') as f:
    json.dump(feat_ranges, f, ensure_ascii=False, indent=2)

# 保存默认值
defaults = {}
for f in anomaly_features:
    if f in df.columns:
        defaults[f] = float(df[f].median())
    else:
        defaults[f] = 0.0

# 添加滞后特征默认值
if '耗用矾量_kg' in df.columns:
    defaults['lag1'] = float(df['耗用矾量_kg'].median())
    defaults['lag2'] = float(df['耗用矾量_kg'].median())
    defaults['耗用矾量_kg'] = float(df['耗用矾量_kg'].median())
    defaults['_target_std'] = float(df['耗用矾量_kg'].std())
    # v12.2: 单位投矾滞后特征（与目标单位一致）
    if '原水量_Km3' in df.columns and df['原水量_Km3'].notna().sum() > 0:
        unit_alum = df['耗用矾量_kg'] / df['原水量_Km3'].clip(lower=0.01)
        unit_alum = unit_alum[(unit_alum > 0) & (unit_alum < 100)]
        defaults['lag1_unit'] = float(unit_alum.median()) if len(unit_alum) else 0.0
        defaults['lag2_unit'] = float(unit_alum.median()) if len(unit_alum) else 0.0
        defaults['_target_unit_std'] = float(unit_alum.std()) if len(unit_alum) else 0.0

# 添加小时原水量默认值
if '原水量_Km3_hour' in df.columns:
    defaults['原水量_Km3_hour'] = float(df['原水量_Km3_hour'].median())
elif '原水量_Km3' in df.columns:
    defaults['原水量_Km3_hour'] = float(df['原水量_Km3'].median() / 24.0)

# v12.2: 模型特征默认值（库区水位/消耗电 不在 UI 输入, 预测时用中位数补全）
for f, fallback in [('库区水位_m', 0.0), ('消耗电_kWh', 0.0)]:
    if f in df.columns:
        defaults[f] = float(df[f].median())
    else:
        defaults[f] = fallback

with open(os.path.join(model_dir, 'feature_defaults.json'), 'w', encoding='utf-8') as f:
    json.dump(defaults, f, ensure_ascii=False, indent=2)

print("\n✅ 训练完成！")
print(f"异常检测特征: {anomaly_features}")
print(f"特征统计: {feat_stats.keys()}")
print(f"默认值: {defaults}")