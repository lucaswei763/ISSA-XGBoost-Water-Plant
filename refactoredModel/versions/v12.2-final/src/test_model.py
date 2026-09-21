import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error

BASE = os.path.dirname(os.path.abspath(__file__))

# ===== 1. 加载模型 =====
model = joblib.load(BASE + r'\models\best_model.pkl')
sc = joblib.load(BASE + r'\models\scaler.pkl')
imp = joblib.load(BASE + r'\models\imputer.pkl')
feats = joblib.load(BASE + r'\models\selected_features.pkl')
print(f'模型特征({len(feats)}): {feats}')
print()

# ===== 2. 用补全后的真实数据测试 =====
import sys
sys.path.insert(0, BASE)
import data_loader as dl
df = dl.load_raw_data(BASE + r'\data.xlsx')
df, _ = dl.clean_data(df)
df = df.sort_values('日期').reset_index(drop=True)
df['lag1'] = df['耗用矾量_kg'].shift(1)
df['lag2'] = df['耗用矾量_kg'].shift(2)
for f in feats:
    df[f] = pd.to_numeric(df[f], errors='coerce')
df[feats] = df[feats].ffill().bfill()
for f in feats:
    df[f] = df[f].fillna(df[f].median())
keep_cols = feats + ['耗用矾量_kg', '日期']
df = df[keep_cols].dropna().reset_index(drop=True)

X = df[feats].values
y = df['耗用矾量_kg'].values
N = len(X)

# 70/30 时序
split70 = int(N * 0.7)
Xtr = imp.transform(X[:split70])
Xte = imp.transform(X[split70:])
Xtr = sc.transform(Xtr); Xte = sc.transform(Xte)
ytr, yte = y[:split70], y[split70:]

yp = model.predict(Xte)
r2 = r2_score(yte, yp); mae = mean_absolute_error(yte, yp)
ape = np.abs(yp - yte) / yte * 100
print('===== 时序测试集 (70%后段) =====')
print(f'测试天数: {len(yte)}')
print(f'时序R² = {r2:.4f}  MAE = {mae:.0f} kg')
print(f'±5%: {np.mean(ape<5)*100:.0f}%  ±10%: {np.mean(ape<10)*100:.0f}%  ±20%: {np.mean(ape<20)*100:.0f}%')
print()

# 最差10天
print('===== 误差最大的10天 =====')
worst = np.argsort(-ape)[:10]
for i in worst:
    d = df['日期'].iloc[split70 + i].strftime('%Y-%m-%d')
    print(f'  {d}: 实际={yte[i]:.0f} 预测={yp[i]:.0f} 误差={ape[i]:.0f}%')
print()

# 误差最好10天
print('===== 误差最小的10天 =====')
best = np.argsort(ape)[:10]
for i in best:
    d = df['日期'].iloc[split70 + i].strftime('%Y-%m-%d')
    print(f'  {d}: 实际={yte[i]:.0f} 预测={yp[i]:.0f} 误差={ape[i]:.1f}%')

# ===== 3. 单日预测演示 =====
print()
print('===== 单日预测演示 =====')
test_cases = [
    ('低浊常温', {'浑浊度_0点':1.5,'原水量_Km3':200,'温度_C':20,'氨氮_mg_per_L':0.05,'pH值':7.0,'库区水位_m':150.0,'消耗电_kWh':45000,'供水量_Km3':195,'lag1':2000,'lag2':2000}),
    ('高浊',     {'浑浊度_0点':30.0,'原水量_Km3':250,'温度_C':25,'氨氮_mg_per_L':0.50,'pH值':7.0,'库区水位_m':160.0,'消耗电_kWh':50000,'供水量_Km3':245,'lag1':2000,'lag2':2000}),
    ('低温',     {'浑浊度_0点':1.0,'原水量_Km3':300,'温度_C':10,'氨氮_mg_per_L':0.05,'pH值':7.3,'库区水位_m':155.0,'消耗电_kWh':60000,'供水量_Km3':295,'lag1':2000,'lag2':2000}),
    ('高水量',   {'浑浊度_0点':1.2,'原水量_Km3':340,'温度_C':26,'氨氮_mg_per_L':0.04,'pH值':7.0,'库区水位_m':158.0,'消耗电_kWh':62000,'供水量_Km3':335,'lag1':2000,'lag2':2000}),
]
for name, td in test_cases:
    vec = [float(td.get(f, 0)) for f in feats]
    row_sc = sc.transform(imp.transform([vec]))
    pred = model.predict(row_sc)[0]
    print(f'  {name}: 预测投矾 = {pred:.0f} kg')

# ===== 4. 水量-投矾响应 =====
print()
print('===== 水量-投矾响应 =====')
base_row = np.array([2.0, 200.0, 20.0, 0.05, 7.1, 150.0, 45000.0, 190.0, 8.0, 8.0])
for flow in [120, 160, 200, 240, 280, 320]:
    row = base_row.copy()
    row[1] = flow
    row[7] = flow * 0.97
    pred = model.predict(sc.transform(imp.transform([row])))[0]
    print(f'  原水量={flow:3d} -> 预测投矾={pred:6.1f} kg')
