import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

BASE = os.path.dirname(os.path.abspath(__file__))

# ===== 1. 用修改后的 data_loader 加载（含缺失补全）=====
import sys
sys.path.insert(0, BASE)
import data_loader as dl

df = dl.load_raw_data(BASE + r'\data.xlsx')
df, avail = dl.clean_data(df)
print(f'清洗后: {len(df)} 行')

# ===== 2. 构建模型实际用的 10 特征 =====
feats = ['浑浊度_0点', '原水量_Km3', '温度_C', '氨氮_mg_per_L',
         'pH值', '库区水位_m', '消耗电_kWh', '供水量_Km3', 'lag1', 'lag2']

df = df.sort_values('日期').reset_index(drop=True)
df['lag1'] = df['耗用矾量_kg'].shift(1)
df['lag2'] = df['耗用矾量_kg'].shift(2)

for f in feats:
    df[f] = pd.to_numeric(df[f], errors='coerce')
df[feats] = df[feats].ffill().bfill()
for f in feats:
    df[f] = df[f].fillna(df[f].median())
# 只检查特征列和目标列，不理会无关的 NaN 列
keep_cols = feats + ['耗用矾量_kg', '日期']
df = df[keep_cols].dropna().reset_index(drop=True)

X = df[feats].values
y = df['耗用矾量_kg'].values
N = len(X)
print(f'训练样本: {N}  日期: {df["日期"].iloc[0].date()} ~ {df["日期"].iloc[-1].date()}')
print(f'目标: mean={y.mean():.0f} std={y.std():.0f}')

# ===== 3. 70/30 时序切分 + 评估 =====
split70 = int(N * 0.7)
imp = SimpleImputer(strategy='median')
Xtr = imp.fit_transform(X[:split70]); Xte = imp.transform(X[split70:])
sc = StandardScaler()
Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
ytr, yte = y[:split70], y[split70:]

from train_config import load_train_params
params = load_train_params('baseline')
model = XGBRegressor(**params)
model.fit(Xtr, ytr)
yp = model.predict(Xte)
r2 = r2_score(yte, yp); mae = mean_absolute_error(yte, yp)
ape = np.abs(yp - yte) / yte * 100
print(f'\n=== 重训后时序评估 ===')
print(f'时序R² = {r2:.4f}  MAE = {mae:.0f} kg')
print(f'±5%: {np.mean(ape<5)*100:.0f}%  ±10%: {np.mean(ape<10)*100:.0f}%  ±20%: {np.mean(ape<20)*100:.0f}%')
yptr = model.predict(Xtr)
print(f'训练R² = {r2_score(ytr,yptr):.4f}  过拟合gap = {r2_score(ytr,yptr)-r2:.4f}')

# ===== 4. 验证水量-投矾响应 =====
print(f'\n=== 水量-投矾响应测试 (固定浊度=2, lag=2000) ===')
base_row = np.array([2.0, 200.0, 20.0, 0.05, 7.1, 150.0, 45000.0, 190.0, 8.0, 8.0])
for flow in [120, 160, 200, 240, 280, 320]:
    row = base_row.copy()
    row[1] = flow
    row[7] = flow * 0.97
    pred = model.predict(sc.transform(imp.transform([row])))[0]
    print(f'  原水量={flow:3d} -> 预测投矾={pred:6.1f} kg')

# ===== 5. 全量重训并保存 =====
print(f'\n=== 全量训练并保存 ===')
Xall = imp.fit_transform(X)
Xall = sc.fit_transform(Xall)
model_full = XGBRegressor(**params)
model_full.fit(Xall, y)
joblib.dump(model_full, BASE + r'\models\best_model.pkl')
joblib.dump(sc, BASE + r'\models\scaler.pkl')
joblib.dump(imp, BASE + r'\models\imputer.pkl')
joblib.dump(feats, BASE + r'\models\selected_features.pkl')
import json
json.dump({
    'model_type': 'XGBoost', 'version': '11.0_fixed_flow',
    'features': feats, 'target_col': '耗用矾量_kg',
    'r2_timeseries': r2, 'mae': mae, 'n_samples': N,
    'fix': '原水量_Km3缺失已用原水量_千m3补全'
}, open(BASE + r'\models\metadata.json', 'w'), indent=2, ensure_ascii=False)
print('✅ 模型 v11.0 已保存 (fix/models/)')
