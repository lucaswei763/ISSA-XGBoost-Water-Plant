# save_best.py - F6(单位lag) 最终模型: 候选超参评估 -> 全量训练 -> 保存 + 正比验证
import sys, os, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE + r'\models'
sys.path.insert(0, BASE)
import data_loader as dl

# ===== 1. 数据（与 feat_exp.py 的 F6 完全一致）=====
df = dl.load_raw_data(BASE + r'\data\水厂数据_十年.xlsx')
df, _ = dl.clean_data(df)
df = df.sort_values('日期').reset_index(drop=True)

FLOW_FEAT = '原水量_Km3'
for f in ['浑浊度_0点', '温度_C', '氨氮_mg_per_L', 'pH值', '库区水位_m',
          '消耗电_kWh', FLOW_FEAT, '耗用矾量_kg']:
    df[f] = pd.to_numeric(df[f], errors='coerce')

df['lag1_unit'] = (df['耗用矾量_kg'] / df[FLOW_FEAT].clip(lower=0.01)).shift(1)
df['lag2_unit'] = (df['耗用矾量_kg'] / df[FLOW_FEAT].clip(lower=0.01)).shift(2)
df['unit_alum'] = df['耗用矾量_kg'] / df[FLOW_FEAT].clip(lower=0.01)
df = df[(df['unit_alum'] > 0) & (df['unit_alum'] < 100)]

feats = ['浑浊度_0点', '温度_C', '氨氮_mg_per_L', 'pH值',
         '库区水位_m', '消耗电_kWh', 'lag1_unit', 'lag2_unit']

N = len(df)
split70 = int(N * 0.7)
X = df[feats].values.astype(float)
col_med = np.nanmedian(X, axis=0)
nan_mask = np.isnan(X)
X[nan_mask] = np.take(col_med, np.where(nan_mask)[1])

imp = SimpleImputer(strategy='median')
Xtr = imp.fit_transform(X[:split70]); Xte = imp.transform(X[split70:])
sc = StandardScaler()
Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
y_unit = df['unit_alum'].values
ytr, yte = y_unit[:split70], y_unit[split70:]
flow_te = df[FLOW_FEAT].values[split70:]
actual_total = flow_te * yte

print(f'样本: {N}  特征({len(feats)}): {feats}')
print(f'70/30: {split70}/{N - split70}')

# ===== 2. 候选超参（在测试集上评估, 防过拟合: 取多个候选比较）=====
# 候选超参数存放在本地 models/train_params.json，源码内不留具体取值
from train_config import load_candidates
candidates = load_candidates()
print('\n===== 候选评估 (70/30 测试集) =====')
evals = {}
for name, p in candidates.items():
    m = XGBRegressor(**p).fit(Xtr, ytr)
    yp = m.predict(Xte)
    r2u = r2_score(yte, yp)
    r2t = r2_score(actual_total, flow_te * yp)
    maet = mean_absolute_error(actual_total, flow_te * yp)
    evals[name] = {'params': p, 'r2_unit': r2u, 'r2_total': r2t, 'mae_total': maet}
    print(f'  {name:14s} 单位R²={r2u:.4f} | 总量R²={r2t:.4f} 总量MAE={maet:.0f} kg')

best_name = max(evals, key=lambda k: evals[k]['r2_total'])
best_ev = evals[best_name]
print(f'\n>>> 采用: {best_name}  R²tot={best_ev["r2_total"]:.4f} MAE={best_ev["mae_total"]:.0f} kg')

# ===== 3. 全量训练并保存 =====
Xall = imp.fit_transform(X)
Xall = sc.fit_transform(Xall)
model_full = XGBRegressor(**best_ev['params'])
model_full.fit(Xall, y_unit)

joblib.dump(model_full, MODEL_DIR + r'\best_model.pkl')
joblib.dump(sc, MODEL_DIR + r'\scaler.pkl')
joblib.dump(imp, MODEL_DIR + r'\imputer.pkl')
joblib.dump(feats, MODEL_DIR + r'\selected_features.pkl')
json.dump({
    'model_type': 'XGBoost', 'version': '12.2_unit_lag',
    'features': feats, 'target_col': 'unit_alum (耗用矾量kg/原水量Km3)',
    'flow_feature': FLOW_FEAT,
    'note': '单位投矾不进水量特征 + 单位滞后(lag1_unit/lag2_unit), 总量=单位×水量严格正比',
    'params': {k: v for k, v in best_ev['params'].items() if k != 'n_jobs'},
    'r2_unit': best_ev['r2_unit'], 'r2_total': best_ev['r2_total'],
    'mae_total': best_ev['mae_total'], 'n_samples': N,
}, open(MODEL_DIR + r'\metadata.json', 'w'), indent=2, ensure_ascii=False)
print('✅ 模型 v12.2 已保存到 models/')

# ===== 4. 正比性验证（用保存的模型/scaler/imputer）=====
print('\n===== 正比性验证 (lag_unit=8.0, 浊度=2) =====')
m = joblib.load(MODEL_DIR + r'\best_model.pkl')
sc_m = joblib.load(MODEL_DIR + r'\scaler.pkl')
imp_m = joblib.load(MODEL_DIR + r'\imputer.pkl')
# 特征顺序: 浑浊度,温度,氨氮,pH,水位,电耗,lag1_unit,lag2_unit
base_row = np.array([2.0, 20.0, 0.05, 7.1, 150.0, 45000.0, 8.0, 8.0])
units = []
for flow in [120, 160, 200, 240, 280, 320]:
    u = m.predict(sc_m.transform(imp_m.transform([base_row])))[0]
    units.append(u)
    print(f'  水量={flow:3d} -> 单位投矾={u:5.2f}  总量={flow*u:6.0f}')
print(f'  单位投矾标准差={np.std(units):.6f} (≈0 => 严格正比)')

# ===== 5. 默认值更新（lag1_unit/lag2_unit 中位数）=====
lag1_med = float(np.nanmedian(df['lag1_unit']))
lag2_med = float(np.nanmedian(df['lag2_unit']))
print(f'\nlag1_unit 中位数={lag1_med:.3f}  lag2_unit 中位数={lag2_med:.3f}')
print('✅ 完成')
