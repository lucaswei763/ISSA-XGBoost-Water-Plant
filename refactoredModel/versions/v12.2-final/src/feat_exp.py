# feat_exp.py - 特征工程实验（单位投矾模型, 严格正比约束）
# 约束: 特征中不含任何水量特征, 保证总量=单位×水量严格正比
# 评估: 固定 70/30 时序切分, 与 v12.1 完全一致
import sys, os, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import data_loader as dl

df = dl.load_raw_data(BASE + r'\data\水厂数据_十年.xlsx')
df, _ = dl.clean_data(df)
df = df.sort_values('日期').reset_index(drop=True)

FLOW_FEAT = '原水量_Km3'
for f in ['浑浊度_0点', '温度_C', '氨氮_mg_per_L', 'pH值', '库区水位_m',
          '消耗电_kWh', '供水量_Km3', FLOW_FEAT, '耗用矾量_kg']:
    if f in df.columns:
        df[f] = pd.to_numeric(df[f], errors='coerce')

# 基础 lag
df['lag1'] = df['耗用矾量_kg'].shift(1)
df['lag2'] = df['耗用矾量_kg'].shift(2)
df['lag3'] = df['耗用矾量_kg'].shift(3)
df['lag7'] = df['耗用矾量_kg'].shift(7)   # 周滞后

# 衍生特征（物理意义, 不含水量）
df['浊度_氨氮_比值'] = df['浑浊度_0点'] / df['氨氮_mg_per_L'].clip(lower=0.01)
df['浊度_变化'] = df['浑浊度_0点'] - df['浑浊度_0点'].shift(1)        # 浊度日变化
df['浊度_周均值'] = df['浑浊度_0点'].rolling(7, min_periods=1).mean().shift(1)  # 前7日均浊
df['温度_变化'] = df['温度_C'] - df['温度_C'].shift(1)
df['lag1_浊度'] = df['耗用矾量_kg'].shift(1) * 0  # 占位, 用单位投矾滞后代替

# 单位投矾 lag（更直接的自回归信号）
df['lag1_unit'] = (df['耗用矾量_kg'] / df[FLOW_FEAT].clip(lower=0.01)).shift(1)
df['lag2_unit'] = (df['耗用矾量_kg'] / df[FLOW_FEAT].clip(lower=0.01)).shift(2)

df['unit_alum'] = df['耗用矾量_kg'] / df[FLOW_FEAT].clip(lower=0.01)
df = df[(df['unit_alum'] > 0) & (df['unit_alum'] < 100)]

# 特征集定义
FEAT_SETS = {
    'F0_基线8特征': ['浑浊度_0点', '温度_C', '氨氮_mg_per_L', 'pH值',
                  '库区水位_m', '消耗电_kWh', 'lag1', 'lag2'],
    'F1_加lag3': ['浑浊度_0点', '温度_C', '氨氮_mg_per_L', 'pH值',
               '库区水位_m', '消耗电_kWh', 'lag1', 'lag2', 'lag3'],
    'F2_加lag3lag7': ['浑浊度_0点', '温度_C', '氨氮_mg_per_L', 'pH值',
                    '库区水位_m', '消耗电_kWh', 'lag1', 'lag2', 'lag3', 'lag7'],
    'F3_加浊度比': ['浑浊度_0点', '温度_C', '氨氮_mg_per_L', 'pH值',
                 '库区水位_m', '消耗电_kWh', 'lag1', 'lag2', '浊度_氨氮_比值'],
    'F4_加浊度变化': ['浑浊度_0点', '温度_C', '氨氮_mg_per_L', 'pH值',
                   '库区水位_m', '消耗电_kWh', 'lag1', 'lag2', '浊度_变化', '温度_变化'],
    'F5_加周均值': ['浑浊度_0点', '温度_C', '氨氮_mg_per_L', 'pH值',
                 '库区水位_m', '消耗电_kWh', 'lag1', 'lag2', '浊度_周均值'],
    'F6_单位lag': ['浑浊度_0点', '温度_C', '氨氮_mg_per_L', 'pH值',
                 '库区水位_m', '消耗电_kWh', 'lag1_unit', 'lag2_unit'],
    'F7_综合': ['浑浊度_0点', '温度_C', '氨氮_mg_per_L', 'pH值',
              '库区水位_m', '消耗电_kWh', 'lag1', 'lag2', 'lag3', 'lag7',
              '浊度_氨氮_比值', '浊度_周均值'],
}

from train_config import load_train_params
params = load_train_params('baseline')

N = len(df)
split70 = int(N * 0.7)
flow_all = df[FLOW_FEAT].values
y_unit = df['unit_alum'].values
flow_te = flow_all[split70:]
actual_total = flow_te * y_unit[split70:]

print(f'样本: {N}  70/30切分: {split70}/{N - split70}\n')
results = {}
for name, feats in FEAT_SETS.items():
    # 数值化 + 缺失处理
    X = df[feats].values.astype(float)
    # 前 N 行 lag 为 NaN -> 用列中位数补（与 retrain 一致的做法）
    col_med = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_med, np.where(nan_mask)[1])
    # 标准化
    imp = SimpleImputer(strategy='median')
    Xtr = imp.fit_transform(X[:split70]); Xte = imp.transform(X[split70:])
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    ytr, yte = y_unit[:split70], y_unit[split70:]
    m = XGBRegressor(**params).fit(Xtr, ytr)
    yp = m.predict(Xte)
    r2u = r2_score(yte, yp); maeu = mean_absolute_error(yte, yp)
    r2t = r2_score(actual_total, flow_te * yp)
    maet = mean_absolute_error(actual_total, flow_te * yp)
    results[name] = {'r2_unit': r2u, 'mae_unit': maeu, 'r2_total': r2t, 'mae_total': maet, 'n_feats': len(feats)}
    print(f'{name:14s} 特征数={len(feats):2d} | 单位R²={r2u:.4f} 单位MAE={maeu:.2f} | 总量R²={r2t:.4f} 总量MAE={maet:.0f} kg')

# 目标变换实验: log(unit_alum)
print('\n===== 目标变换实验 (基线特征) =====')
feats = FEAT_SETS['F0_基线8特征']
X = df[feats].values.astype(float)
col_med = np.nanmedian(X, axis=0)
nan_mask = np.isnan(X)
X[nan_mask] = np.take(col_med, np.where(nan_mask)[1])
imp = SimpleImputer(strategy='median')
Xtr = imp.fit_transform(X[:split70]); Xte = imp.transform(X[split70:])
sc = StandardScaler()
Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
ytr, yte = y_unit[:split70], y_unit[split70:]
for tname, transform in [('log目标', np.log), ('sqrt目标', np.sqrt)]:
    ytr_t, yte_t = transform(ytr), transform(yte)
    m = XGBRegressor(**params).fit(Xtr, ytr_t)
    yp_t = m.predict(Xte)
    yp = np.exp(yp_t) if tname == 'log目标' else yp_t ** 2
    r2u = r2_score(yte, yp); maet = mean_absolute_error(actual_total, flow_te * yp)
    r2t = r2_score(actual_total, flow_te * yp)
    print(f'{tname}: 单位R²={r2u:.4f} | 总量R²={r2t:.4f} 总量MAE={maet:.0f} kg')

with open(BASE + r'\_feat_exp.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print('\n✅ 特征实验完成, 结果写入 _feat_exp.json')
