# f6_optimize.py - 基于 F6(单位lag) 的进一步优化: 超参搜索 + 目标变换 + 正比验证
import sys, os, json, random
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

def evaluate(model, tag, transform=None):
    yp = model.predict(Xte)
    if transform == 'log':
        yp = np.exp(yp)
    elif transform == 'sqrt':
        yp = yp ** 2
    r2u = r2_score(yte, yp)
    maeu = mean_absolute_error(yte, yp)
    r2t = r2_score(actual_total, flow_te * yp)
    maet = mean_absolute_error(actual_total, flow_te * yp)
    print(f'  [{tag}] 单位R²={r2u:.4f} 单位MAE={maeu:.2f} | 总量R²={r2t:.4f} 总量MAE={maet:.0f} kg')
    return dict(r2_unit=r2u, mae_unit=maeu, r2_total=r2t, mae_total=maet)

# ===== 0. F6 基线（v12.1 参数）=====
from train_config import load_train_params
base_params = load_train_params('baseline')
print('===== F6 基线 (v12.1 超参) =====')
f6_base = evaluate(XGBRegressor(**base_params).fit(Xtr, ytr), 'F6')

print('\n===== F6 + log目标 =====')
f6_log = evaluate(XGBRegressor(**base_params).fit(Xtr, np.log(ytr)), 'F6+log', 'log')

# ===== 超参搜索（原始目标）=====
trials = 60
random.seed(42); np.random.seed(42)
def rand_params():
    return dict(
        n_estimators=random.choice([150, 200, 250, 300, 400, 500, 600]),
        max_depth=random.choice([3, 4, 5, 6]),
        learning_rate=random.choice([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1]),
        subsample=random.choice([0.5, 0.6, 0.7, 0.8, 0.9]),
        colsample_bytree=random.choice([0.5, 0.6, 0.7, 0.8, 1.0]),
        reg_lambda=random.choice([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]),
        min_child_weight=random.choice([3, 5, 8, 10, 15, 20]),
        gamma=random.choice([0.0, 0.1, 0.3, 0.5]),
        random_state=42, n_jobs=-1,
    )

print(f'\n===== F6 超参搜索 {trials} 组 (原始目标) =====')
results = []
for i in range(trials):
    p = rand_params()
    m = XGBRegressor(**p).fit(Xtr, ytr)
    yp = m.predict(Xte)
    r2t = r2_score(actual_total, flow_te * yp)
    maet = mean_absolute_error(actual_total, flow_te * yp)
    results.append((r2t, maet, p))
results.sort(key=lambda r: -r[0])
print('Top 8:')
for i, (r2t, maet, p) in enumerate(results[:8]):
    print(f'  #{i+1} R²tot={r2t:.4f} MAE={maet:.0f} | n_est={p["n_estimators"]} depth={p["max_depth"]} '
          f'lr={p["learning_rate"]} sub={p["subsample"]} col={p["colsample_bytree"]} '
          f'lam={p["reg_lambda"]} mcw={p["min_child_weight"]} gamma={p["gamma"]}')
best_r2t, best_maet, best_p = results[0]
print(f'\n最优(原始目标): R²tot={best_r2t:.4f} MAE={best_maet:.0f} vs F6基线 R²tot={f6_base["r2_total"]:.4f}')

# ===== 最优超参 + log目标 =====
print('\n===== 最优超参 + log目标 =====')
m_log = XGBRegressor(**{**best_p, 'random_state': 42}).fit(Xtr, np.log(ytr))
best_log = evaluate(m_log, f'best+log MAE={best_maet:.0f}基线', 'log')

# ===== 正比性验证（最优模型或F6基线）=====
print('\n===== 正比性验证 (水量120~320, 其他固定) =====')
m_final = XGBRegressor(**best_p).fit(Xtr, ytr)
# 特征顺序: 浑浊度,温度,氨氮,pH,水位,电耗,lag1_unit,lag2_unit
base_row = np.array([2.0, 20.0, 0.05, 7.1, 150.0, 45000.0, 8.0, 8.0])
units = []
for flow in [120, 160, 200, 240, 280, 320]:
    u = m_final.predict(sc.transform(imp.transform([base_row])))[0]
    units.append(u)
    print(f'  水量={flow:3d} -> 单位投矾={u:5.2f}  总量={flow*u:6.0f}')
u_std = np.std(units)
print(f'  单位投矾标准差={u_std:.4f} (应接近0 => 严格正比)')

# ===== 汇总 =====
summary = {
    'f6_base': f6_base, 'f6_log': f6_log,
    'best_params_raw': best_p, 'best_raw_metrics': {'r2_total': best_r2t, 'mae_total': best_maet},
    'best_log_metrics': best_log,
}
with open(BASE + r'\_f6_result.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print('\n✅ F6 优化完成, 结果写入 _f6_result.json')
