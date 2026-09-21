# verify_v122.py - v12.2 端到端验证
# 1. WaterPredictor 加载新模型
# 2. 真实数据 70/30 评估（用 lag1_unit/lag2_unit 特征）
# 3. 正比性验证（通过 predict 接口, 检查总量=单位×水量）
import sys, os, json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from predictor_service import WaterPredictor
import data_loader as dl

p = WaterPredictor(model_dir=BASE + r'\models')
print(f'模型版本: {getattr(p, "version", "?")}')
print(f'模型特征({len(p.features)}): {p.features}')
print()

# ===== 1. 单日预测 =====
print('===== 单日预测演示 (predict 接口) =====')
# 合成输入，仅用于演示接口，不含真实工况记录
test_cases = [
    ('低浊常温', {'日期': '2026-02-28', '浑浊度_0点':1.5, '原水量_Km3':200, '温度_C':20, '氨氮_mg_per_L':0.05, 'pH值':7.0, '库区水位_m':150.0, '消耗电_kWh':45000}),
    ('高浊',     {'日期': '2026-02-28', '浑浊度_0点':30.0, '原水量_Km3':250, '温度_C':25, '氨氮_mg_per_L':0.50, 'pH值':7.0, '库区水位_m':160.0, '消耗电_kWh':50000}),
    ('低温',     {'日期': '2026-02-28', '浑浊度_0点':1.0, '原水量_Km3':300, '温度_C':10, '氨氮_mg_per_L':0.05, 'pH值':7.3, '库区水位_m':155.0, '消耗电_kWh':60000}),
    ('高水量',   {'日期': '2026-02-28', '浑浊度_0点':1.2, '原水量_Km3':340, '温度_C':26, '氨氮_mg_per_L':0.04, 'pH值':7.0, '库区水位_m':158.0, '消耗电_kWh':62000}),
]
for name, td in test_cases:
    pred, warns = p.predict(td)
    unit = pred / td['原水量_Km3'] if pred else 0
    print(f'  {name}: 预测总量={pred:.0f} kg  单位投矾={unit:.2f} kg/Km3')

# ===== 2. 正比性验证（固定其他条件, 遍历水量）=====
print('\n===== 正比性验证 (其他固定, 水量 120~320) =====')
units = []
for flow in [120, 160, 200, 240, 280, 320]:
    td = {'日期': '2026-02-28', '浑浊度_0点':2.0, '原水量_Km3':float(flow), '温度_C':20.0,
          '氨氮_mg_per_L':0.05, 'pH值':7.1, '库区水位_m':150.0, '消耗电_kWh':45000.0,
          'lag1_unit':8.0, 'lag2_unit':8.0}
    pred, _ = p.predict(td)
    units.append(pred / flow)
    print(f'  水量={flow:3d} -> 投矾总量={pred:6.0f} kg  单位投矾={pred/flow:5.2f}')
print(f'  单位投矾标准差={np.std(units):.6f} (≈0 => 严格正比)')

# ===== 3. 真实数据 70/30 评估 =====
print('\n===== 真实数据 70/30 评估 (lag1_unit 特征) =====')
df = dl.load_raw_data(BASE + r'\data\水厂数据_十年.xlsx')
df, _ = dl.clean_data(df)
df = df.sort_values('日期').reset_index(drop=True)
for f in ['浑浊度_0点', '温度_C', '氨氮_mg_per_L', 'pH值', '库区水位_m', '消耗电_kWh', '原水量_Km3', '耗用矾量_kg']:
    df[f] = pd.to_numeric(df[f], errors='coerce')
df['lag1_unit'] = (df['耗用矾量_kg'] / df['原水量_Km3'].clip(lower=0.01)).shift(1)
df['lag2_unit'] = (df['耗用矾量_kg'] / df['原水量_Km3'].clip(lower=0.01)).shift(2)
df['unit_alum'] = df['耗用矾量_kg'] / df['原水量_Km3'].clip(lower=0.01)
df = df[(df['unit_alum'] > 0) & (df['unit_alum'] < 100)]
feats = list(p.features)
df[feats] = df[feats].ffill().bfill()
for f in feats:
    df[f] = df[f].fillna(df[f].median())
df = df.dropna(subset=feats).reset_index(drop=True)
N = len(df); split70 = int(N * 0.7)
te = df.iloc[split70:].reset_index(drop=True)

import joblib
model = joblib.load(BASE + r'\models\best_model.pkl')
sc = joblib.load(BASE + r'\models\scaler.pkl')
imp = joblib.load(BASE + r'\models\imputer.pkl')
Xte = sc.transform(imp.transform(te[feats].values))
yp_unit = model.predict(Xte)
yte = te['unit_alum'].values
flow_te = te['原水量_Km3'].values
actual_total = flow_te * yte
pred_total = flow_te * yp_unit
r2u = r2_score(yte, yp_unit); r2t = r2_score(actual_total, pred_total)
maet = mean_absolute_error(actual_total, pred_total)
ape = np.abs(pred_total - actual_total) / actual_total * 100
print(f'测试天数: {len(te)}')
print(f'单位R²={r2u:.4f}  总量R²={r2t:.4f}  总量MAE={maet:.0f} kg')
print(f'±5%: {np.mean(ape<5)*100:.0f}%  ±10%: {np.mean(ape<10)*100:.0f}%  ±20%: {np.mean(ape<20)*100:.0f}%')

# 最差10天
print('\n最差10天:')
worst = np.argsort(-ape)[:10]
for i in worst:
    d = te['日期'].iloc[i].strftime('%Y-%m-%d')
    print(f'  {d}: 实际={actual_total[i]:.0f} 预测={pred_total[i]:.0f} 误差={ape[i]:.0f}%')

# ===== 4. SHAP explain 接口 =====
print('\n===== explain 接口 (SHAP) =====')
try:
    td = {'日期': '2026-02-28', '浑浊度_0点':2.0, '原水量_Km3':200.0, '温度_C':20.0,
          '氨氮_mg_per_L':0.05, 'pH值':7.1, '库区水位_m':150.0, '消耗电_kWh':45000.0}
    exp = p.explain(td)
    print(f'  预测: {exp["prediction"]:.0f} kg  base: {exp.get("base_value")}')
    for c in exp.get('contributions', []):
        print(f'    {c["feature"]}: value={c["value"]:.2f} shap={c["shap"]:+.1f} {c["direction"]}')
    print(f'  confidence: {exp.get("confidence", {}).get("level")}')
except Exception as e:
    import traceback; traceback.print_exc()
print('\n✅ v12.2 验证完成')
