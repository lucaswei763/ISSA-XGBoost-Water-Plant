# verify_fixes.py - 验证 B1(record_actual单位换算) 与 B2(水位/电耗默认值补全)
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor_service import WaterPredictor

BASE = os.path.dirname(os.path.abspath(__file__))
p = WaterPredictor(model_dir=BASE + r'\models')

print('===== B2: UI 输入(无水位/电耗) 特征补全 =====')
td = {'日期': '2026-02-28', '浑浊度_0点': 2.0, '原水量_Km3_hour': 9.5,
      '温度_C': 20.0, '氨氮_mg_per_L': 0.05, 'pH值': 7.1}
aug = p._augment_input(dict(td))
print(f'库区水位_m = {aug.get("库区水位_m")} (应为非 0 的本地中位数)')
print(f'消耗电_kWh = {aug.get("消耗电_kWh")} (应为非 0 的本地中位数)')
print(f'lag1_unit  = {aug.get("lag1_unit"):.3f}')
print(f'lag2_unit  = {aug.get("lag2_unit"):.3f}')
pred, _ = p.predict(td)
print(f'预测投矾 = {pred:.0f} kg')
assert aug.get('库区水位_m') and aug['库区水位_m'] > 0, 'B2 水位未补全!'
assert aug.get('消耗电_kWh') and aug['消耗电_kWh'] > 0, 'B2 电耗未补全!'
print('✅ B2 通过: 水位/电耗已用中位数补全\n')

print('===== B1: record_actual 后 lag 单位换算 =====')
# 录入一个与本地基准同量级的合成总量：次日 lag1_unit 应等于 总量 / 当日水量
flow = 240.0
median_unit = float(p.default_values.get('lag1_unit') or 0.0)
RECORDED_TOTAL = float(round(median_unit * flow)) or flow * 10.0
p.record_actual('2026-02-25', RECORDED_TOTAL)
td2 = {'日期': '2026-02-26', '浑浊度_0点': 2.0, '原水量_Km3': flow,
       '温度_C': 20.0, '氨氮_mg_per_L': 0.05, 'pH值': 7.1}
aug2 = p._augment_input(dict(td2))
lag1_u = aug2.get('lag1_unit')
expected = RECORDED_TOTAL / flow
print(f'lag1_unit = {lag1_u:.3f} (期望 = 录入总量 / 当日水量 = {expected:.2f})')
assert lag1_u and abs(lag1_u - expected) < max(1.0, expected * 0.05), f'B1 单位换算失败: lag1_unit={lag1_u}'
print('✅ B1 通过: runtime 总量已正确换算为单位投矾\n')

print('===== 回归: 正比性仍保持 =====')
units = []
for flow in [120, 200, 280]:
    td3 = {'日期': '2026-02-26', '浑浊度_0点': 2.0, '原水量_Km3': float(flow),
           '温度_C': 20.0, '氨氮_mg_per_L': 0.05, 'pH值': 7.1,
           'lag1_unit': 8.0, 'lag2_unit': 8.0}
    pred3, _ = p.predict(td3)
    units.append(pred3 / flow)
print(f'单位投矾: {[f"{u:.2f}" for u in units]} 标准差={np.std(units):.6f}')
assert np.std(units) < 1e-6, '正比性被破坏!'
print('✅ 正比性回归通过')
print('\n全部修复验证完成')
