import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

BASE = os.path.dirname(os.path.abspath(__file__))

# ===== 1. 加载数据（含缺失补全）=====
import sys
sys.path.insert(0, BASE)
import data_loader as dl

df = dl.load_raw_data(BASE + r'\data.xlsx')
df, _ = dl.clean_data(df)
df = df.sort_values('日期').reset_index(drop=True)

# ===== 2. 构建特征 + 单位投矾量目标 =====
# 为实现"总量正比"，单位投矾模型不使用水量特征（避免学到高水量→低单位投矾）
# 总量 = 单位投矾 × 原水量（预测时乘回）
feats = ['浑浊度_0点', '温度_C', '氨氮_mg_per_L',
         'pH值', '库区水位_m', '消耗电_kWh', 'lag1', 'lag2']
FLOW_FEAT = '原水量_Km3'  # 乘回用，不进模型

df['lag1'] = df['耗用矾量_kg'].shift(1)
df['lag2'] = df['耗用矾量_kg'].shift(2)

for f in feats + [FLOW_FEAT]:
    df[f] = pd.to_numeric(df[f], errors='coerce')
df[feats] = df[feats].ffill().bfill()
for f in feats:
    df[f] = df[f].fillna(df[f].median())

# 单位投矾量 = 总投矾 / 原水量 (kg/Km3)
df['unit_alum'] = df['耗用矾量_kg'] / df['原水量_Km3'].clip(lower=0.01)
# 过滤异常单位值
df = df[(df['unit_alum'] > 0) & (df['unit_alum'] < 100)]

keep_cols = list(dict.fromkeys(feats + ['unit_alum', '耗用矾量_kg', FLOW_FEAT, '日期']))
df = df[keep_cols].dropna().reset_index(drop=True)

X = df[feats].values
y_unit = df['unit_alum'].values
flow_all = df[FLOW_FEAT].values
N = len(X)
print(f'样本: {N}  日期: {df["日期"].iloc[0].date()} ~ {df["日期"].iloc[-1].date()}')
print(f'单位投矾: mean={y_unit.mean():.2f} std={y_unit.std():.2f} kg/Km3')

# ===== 3. 70/30 时序评估（单位投矾量）=====
split70 = int(N * 0.7)
imp = SimpleImputer(strategy='median')
Xtr = imp.fit_transform(X[:split70]); Xte = imp.transform(X[split70:])
sc = StandardScaler()
Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
ytr, yte = y_unit[:split70], y_unit[split70:]

from train_config import load_train_params
params = load_train_params('baseline')
model = XGBRegressor(**params)
model.fit(Xtr, ytr)
yp_unit = model.predict(Xte)
r2 = r2_score(yte, yp_unit)
print(f'\n=== 单位投矾量模型 (70/30时序) ===')
print(f'单位投矾 R² = {r2:.4f}  MAE = {mean_absolute_error(yte,yp_unit):.2f} kg/Km3')

# 换算回总量评估（单位×水量）
flow_te = flow_all[split70:]
actual_total = flow_te * yte
pred_total = flow_te * yp_unit
r2_total = r2_score(actual_total, pred_total)
mae_total = mean_absolute_error(actual_total, pred_total)
print(f'换算总量: R² = {r2_total:.4f}  MAE = {mae_total:.0f} kg')

# ===== 4. 正比关系验证（水量不进模型, 单位应恒定）=====
print(f'\n=== 正比关系验证 (lag=2000, 浊度=2) ===')
for flow in [120, 160, 200, 240, 280, 320]:
    # 特征顺序: 浑浊度,温度,氨氮,pH,水位,电耗,lag1,lag2
    row = np.array([2.0, 20.0, 0.05, 7.1, 150.0, 45000.0, 8.0, 8.0])
    u = model.predict(sc.transform(imp.transform([row])))[0]
    print(f'  水量={flow:3d} -> 单位投矾={u:5.2f}  总量={flow*u:6.0f}')

# ===== 5. 全量训练并保存 =====
Xall = imp.fit_transform(X)
Xall = sc.transform(Xall)
model_full = XGBRegressor(**params)
model_full.fit(Xall, y_unit)
joblib.dump(model_full, BASE + r'\models\best_model.pkl')
joblib.dump(sc, BASE + r'\models\scaler.pkl')
joblib.dump(imp, BASE + r'\models\imputer.pkl')
joblib.dump(feats, BASE + r'\models\selected_features.pkl')
import json
json.dump({
    'model_type': 'XGBoost', 'version': '12.1_unit_alum_noflow',
    'features': feats, 'target_col': 'unit_alum (耗用矾量kg/原水量Km3)',
    'flow_feature': FLOW_FEAT, 'note': '单位投矾不进水量特征, 总量=单位×水量严格正比',
    'r2_unit': r2, 'r2_total': r2_total, 'mae_total': mae_total,
    'n_samples': N,
}, open(BASE + r'\models\metadata.json', 'w'), indent=2, ensure_ascii=False)
print('\n✅ 模型 v12.1 (单位投矾量, 无水量特征) 已保存')
