"""
文件名称：refactoredModel/test_predictor.py
所属类别：重构核心生产代码 (Refactored Core Production)

功能描述：
    对底层预测封装服务类 `WaterPredictor` 进行快速功能性自检与集成兼容性测试的脚本。
    它使用一组模拟的水厂监测指标（包括浑浊度、流量、pH、氨氮等）作为输入，
    执行完整的模型前向预测，打印预测结果并验证是否触发范围警告，以保障系统部署或模型重训后可以正常工作。

运行与使用方法：
    直接运行此文件进行测试：
    python test_predictor.py

调用与依赖关系：
    - 导入并测试 `predictor_service.WaterPredictor`。
    - 开发者在重构、更新权重或修改特征计算后用于第一时间的健康检测。
"""
import sys
import os

# Ensure current dir is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predictor_service import WaterPredictor

def main():
    print("🧪 Starting prediction service compatibility check...")
    try:
        predictor = WaterPredictor(model_dir='models')
        print(f"✅ Predictor initialized. Loaded model type: {predictor.model_type}")
        print(f"✅ Feature count: {len(predictor.features)}")
        
        # Mock input dictionary representing user inputs in ui_app.py
        mock_input = {
            "日期": "2026-06-04",
            "浑浊度（NTU）_0点": 21.5,
            "浑浊度（NTU）": 21.5,
            "原水量（Km³）": 240.0,
            "供水量（Km³）": 235.0,
            "温度（℃）_9点": 18.0,
            "温度（℃）": 18.0,
            "氨氮（mg/L）_9点": 0.1,
            "氨氮（mg/L）": 0.1,
            "pH值_9点": 7.2,
            "pH值": 7.2,
            "冲程": 65.0,
            "小时原水量_km3": 10.0,
        }
        
        pred_value, has_warnings = predictor.predict(mock_input)
        print(f"✅ Prediction completed successfully!")
        print(f"   -> Predicted Alum Dosage: {pred_value:.4f} kg/d")
        print(f"   -> Has warnings: {has_warnings}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
