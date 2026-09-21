# gen_aux.py - 为 v12.2 重新生成 SHAP 辅助文件与预测区间（新特征顺序, 单位投矾目标）
import sys, os, json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import shap

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE + r'\models'
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
X = df[feats].values.astype(float)
col_med = np.nanmedian(X, axis=0)
nan_mask = np.isnan(X)
X[nan_mask] = np.take(col_med, np.where(nan_mask)[1])
y_unit = df['unit_alum'].values
flow_all = df[FLOW_FEAT].values
N = len(X)
split70 = int(N * 0.7)

# 加载已保存的模型管线（与 save_best.py 一致）
model = joblib.load(MODEL_DIR + r'\best_model.pkl')
sc = joblib.load(MODEL_DIR + r'\scaler.pkl')
imp = joblib.load(MODEL_DIR + r'\imputer.pkl')

# ===== 1. shap_background: 训练集标准化后样本（500行, 覆盖训练分布）=====
Xtr_imp = imp.transform(X[:split70])
Xtr_sc = sc.transform(Xtr_imp)
bg = Xtr_sc[np.random.RandomState(42).choice(len(Xtr_sc), min(500, len(Xtr_sc)), replace=False)]
joblib.dump(bg, MODEL_DIR + r'\shap_background.pkl')
print(f'✅ shap_background.pkl: {bg.shape}')

# ===== 2. shap_expected: TreeExplainer 自身的期望输出（与 shap 贡献严格自洽）=====
explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
exp = float(explainer.expected_value)
joblib.dump(exp, MODEL_DIR + r'\shap_expected.pkl')
print(f'✅ shap_expected.pkl = {exp:.4f} (单位投矾 kg/Km3)')

# ===== 3. prediction_bounds: 测试集总量残差的 5%/95% 分位 =====
Xte_imp = imp.transform(X[split70:])
Xte_sc = sc.transform(Xte_imp)
yp_unit = model.predict(Xte_sc)
flow_te = flow_all[split70:]
actual_total = flow_te * y_unit[split70:]
pred_total = flow_te * yp_unit
resid = actual_total - pred_total
p5, p95 = np.percentile(resid, 5), np.percentile(resid, 95)
joblib.dump({'p5': float(p5), 'p95': float(p95)}, MODEL_DIR + r'\prediction_bounds.pkl')
print(f'✅ prediction_bounds.pkl: p5={p5:.0f} p95={p95:.0f} kg (总量残差)')

# ===== 4. 一致性验证: shap_expected + Σshap ≈ 模型输出(单位) =====
print('\n===== SHAP 一致性验证 =====')
row = Xte_sc[0:1]
sv = explainer.shap_values(row)[0]
model_out = model.predict(row)[0]
sum_check = exp + sv.sum()
print(f'模型输出(单位投矾)={model_out:.4f}')
print(f'expected + Σshap = {sum_check:.4f}')
print(f'差异 = {abs(model_out - sum_check):.6f} (应≈0)')
print('\n✅ 辅助文件生成完成')
