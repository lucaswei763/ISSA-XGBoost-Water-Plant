import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor_service import WaterPredictor

print("=" * 50)
print("测试预测功能")
print("=" * 50)

try:
    predictor = WaterPredictor()
    print(f"✅ 模型加载成功")
    print(f"   类型: {predictor.model_type}")
    print(f"   版本: {predictor.version}")
    print(f"   特征: {predictor.features}")

    test_data = {
        '日期': '2026-01-15',
        '浑浊度_0点': 1.0,
        '原水量_Km3_hour': 8.3,
        '温度_C': 15,
        '氨氮_mg_per_L': 0.1,
        'pH值': 7.2,
    }

    print("\n测试输入:", test_data)
    print("\n开始预测...")

    pred, warnings = predictor.predict(test_data)
    print(f"\n预测结果: {pred}")
    print(f"警告: {warnings}")

    if pred is None:
        print("\n❌ 预测返回空值！")
    else:
        print(f"\n✅ 预测成功: {pred:.2f} kg")

except Exception as e:
    print(f"\n❌ 发生错误: {e}")
    import traceback
    traceback.print_exc()