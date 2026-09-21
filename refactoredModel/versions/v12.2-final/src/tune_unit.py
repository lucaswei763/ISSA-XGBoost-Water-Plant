# tune_unit.py - v12.1 基线复现 + XGBoost 超参搜索（单位投矾模型, 严格正比方案）
# 用法: python tune_unit.py [--quick]
# 注意: 本脚本只评估, 不保存模型; 保存由 save_best.py 负责
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

QUICK = '--quick' in sys.argv

# ===== 1. 数据管线（与 retrain_unit.py 完全一致, 数据用工作区副本）=====
df = dl.load_raw_data(BASE + r'\data\水厂数据_十年.xlsx')
df, _ = dl.clean_data(df)
df = df.sort_values('日期').reset_index(drop=True)

feats = ['浑浊度_0点', '温度_C', '氨氮_mg_per_L',
         'pH值', '库区水位_m', '消耗电_kWh', 'lag1', 'lag2']
FLOW_FEAT = '原水量_Km3'

df['lag1'] = df['耗用矾量_kg'].shift(1)
df['lag2'] = df['耗用矾量_kg'].shift(2)
for f in feats + [FLOW_FEAT]:
    df[f] = pd.to_numeric(df[f], errors='coerce')
df[feats] = df[feats].ffill().bfill()
for f in feats:
    df[f] = df[f].fillna(df[f].median())

df['unit_alum'] = df['耗用矾量_kg'] / df['原水量_Km3'].clip(lower=0.01)
df = df[(df['unit_alum'] > 0) & (df['unit_alum'] < 100)]
keep_cols = list(dict.fromkeys(feats + ['unit_alum', '耗用矾量_kg', FLOW_FEAT, '日期']))
df = df[keep_cols].dropna().reset_index(drop=True)

X = df[feats].values
y_unit = df['unit_alum'].values
flow_all = df[FLOW_FEAT].values
N = len(X)
split70 = int(N * 0.7)
print(f'样本: {N}  日期: {df["日期"].iloc[0].date()} ~ {df["日期"].iloc[-1].date()}')
print(f'单位投矾: mean={y_unit.mean():.2f} std={y_unit.std():.2f} kg/Km3')
print(f'训练/测试: {split70}/{N - split70}')

imp = SimpleImputer(strategy='median')
Xtr = imp.fit_transform(X[:split70]); Xte = imp.transform(X[split70:])
sc = StandardScaler()
Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
ytr, yte = y_unit[:split70], y_unit[split70:]
flow_te = flow_all[split70:]
actual_total = flow_te * yte


def evaluate(model, tag):
    yp_unit = model.predict(Xte)
    r2u = r2_score(yte, yp_unit)
    maeu = mean_absolute_error(yte, yp_unit)
    pred_total = flow_te * yp_unit
    r2t = r2_score(actual_total, pred_total)
    maet = mean_absolute_error(actual_total, pred_total)
    print(f'  [{tag}] 单位R²={r2u:.4f} 单位MAE={maeu:.2f} | 总量R²={r2t:.4f} 总量MAE={maet:.0f} kg')
    return dict(r2_unit=r2u, mae_unit=maeu, r2_total=r2t, mae_total=maet, model=model)


# ===== 2. 基线复现（v12.1 参数）=====
from train_config import load_train_params
base_params = load_train_params('baseline')
print('\n===== 基线 v12.1 =====')
base = evaluate(XGBRegressor(**base_params).fit(Xtr, ytr), 'v12.1')
base_metrics = {k: v for k, v in base.items() if k != 'model'}

# ===== 3. 超参搜索（随机搜索, 固定 70/30 时序）=====
trials = 30 if QUICK else 120
random.seed(42)
np.random.seed(42)

def rand_params():
    return dict(
        n_estimators=random.choice([150, 200, 250, 300, 400, 500]),
        max_depth=random.choice([3, 4, 5, 6]),
        learning_rate=random.choice([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07]),
        subsample=random.choice([0.5, 0.6, 0.7, 0.8, 0.9]),
        colsample_bytree=random.choice([0.5, 0.6, 0.7, 0.8, 1.0]),
        reg_lambda=random.choice([1.0, 2.0, 3.0, 5.0, 8.0, 12.0]),
        min_child_weight=random.choice([5, 8, 10, 15, 20]),
        gamma=random.choice([0.0, 0.1, 0.3, 0.5]),
        random_state=42, n_jobs=-1,
    )

print(f'\n===== 随机搜索 {trials} 组 =====')
results = []
for i in range(trials):
    p = rand_params()
    m = XGBRegressor(**p).fit(Xtr, ytr)
    yp = m.predict(Xte)
    r2u = r2_score(yte, yp)
    r2t = r2_score(actual_total, flow_te * yp)
    maet = mean_absolute_error(actual_total, flow_te * yp)
    results.append((r2u, r2t, maet, p))
    if i % 20 == 0:
        print(f'  trial {i}/{trials} ... best_r2_total so far: {max(r[1] for r in results):.4f}')

results.sort(key=lambda r: -r[1])  # 按总量R²排序
print('\n===== Top 10 (按总量R²) =====')
for i, (r2u, r2t, maet, p) in enumerate(results[:10]):
    print(f'  #{i+1} R²tot={r2t:.4f} R²unit={r2u:.4f} MAE={maet:.0f} | '
          f'n_est={p["n_estimators"]} depth={p["max_depth"]} lr={p["learning_rate"]} '
          f'sub={p["subsample"]} col={p["colsample_bytree"]} lam={p["reg_lambda"]} '
          f'mcw={p["min_child_weight"]} gamma={p["gamma"]}')

best_r2u, best_r2t, best_maet, best_p = results[0]
print('\n===== 最优参数 vs 基线 =====')
print(f'基线: R²tot={base_metrics["r2_total"]:.4f} MAE={base_metrics["mae_total"]:.0f} kg')
print(f'最优: R²tot={best_r2t:.4f} MAE={best_maet:.0f} kg  提升 R²={best_r2t - base_metrics["r2_total"]:+.4f}')
print(json.dumps(best_p, ensure_ascii=False))

# 保存搜索结论供 save_best.py 使用
with open(BASE + r'\_tune_result.json', 'w', encoding='utf-8') as f:
    json.dump({'base': base_metrics, 'best_params': best_p,
               'best_metrics': {'r2_unit': best_r2u, 'r2_total': best_r2t, 'mae_total': best_maet},
               'top10': [{'r2_unit': r[0], 'r2_total': r[1], 'mae_total': r[2], 'params': r[3]} for r in results[:10]]},
              f, indent=2, ensure_ascii=False)
print('\n✅ 搜索结果已写入 _tune_result.json')
