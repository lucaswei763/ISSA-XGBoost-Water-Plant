"""
文件名称：refactoredModel/compare_trained_models.py
所属类别：重构核心生产代码 (Refactored Core Production)

功能描述：
    对已经训练并保存的 XGBoost 与 ResNet 回归模型进行快速读取与离线评估对比的工具脚本。
    执行步骤包括：
    1. 分别从 `models/xgboost` 和 `models/resnet` 目录下反序列化加载 scaler.pkl、imputer.pkl、selected_features.pkl 以及 best_model.pkl；
    2. 加载对应的验证数据集，执行模型预测推断；
    3. 统一测算并输出两种模型在总投加量 (kg/d) 和吨水单位投加量 (kg/m³) 下的 MAE、RMSE、R² 指标并进行控制台表格化输出。

运行与使用方法：
    确保对应的 models 目录中存在已训练的 pkl 权重，然后在终端直接运行：
    python compare_trained_models.py

调用与依赖关系：
    - 被开发或部署人员手动调用，用于快速验证及确认当前生产模型的精度配置。
    - 依赖于 `utils.load_and_preprocess_data` 数据加载函数。
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

# Ensure current dir in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from utils import load_and_preprocess_data

def evaluate_model(model_name, model_dir):
    # Load scaling and model artifacts
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    imputer = joblib.load(os.path.join(model_dir, 'imputer.pkl'))
    features = joblib.load(os.path.join(model_dir, 'selected_features.pkl'))
    model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
    
    # Load dataset using this model's directory (to get the exact preprocess scaling split)
    X_train_scaled, X_test_scaled, y_train, y_test, X_full_scaled, y_full, full_dates, water_test, water_full = load_and_preprocess_data(model_dir=model_dir)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Metrics - Total Dosage
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Metrics - Unit Dosage
    safe_water = np.where(water_test == 0, 1e-5, water_test)
    y_test_unit = y_test / safe_water
    y_pred_unit = y_pred / safe_water
    mae_unit = mean_absolute_error(y_test_unit, y_pred_unit)
    rmse_unit = np.sqrt(mean_squared_error(y_test_unit, y_pred_unit))
    r2_unit = r2_score(y_test_unit, y_pred_unit)
    
    return {
        'MAE': mae, 'RMSE': rmse, 'R2': r2,
        'MAE_Unit': mae_unit, 'RMSE_Unit': rmse_unit, 'R2_Unit': r2_unit
    }

def main():
    xgb_metrics = evaluate_model("ISSA-XGBoost", "models/xgboost")
    resnet_metrics = evaluate_model("ResNet (双头监督残差网络)", "models/resnet")
    
    print("\n" + "=" * 80)
    print("🏆 两种已训练模型性能对比汇总 (测试集表现)")
    print("=" * 80)
    
    summary_df = pd.DataFrame([
        {
            "预测模型": "ISSA-XGBoost",
            "MAE (kg)": f"{xgb_metrics['MAE']:.2f}",
            "RMSE (kg)": f"{xgb_metrics['RMSE']:.2f}",
            "R² 决定系数": f"{xgb_metrics['R2']:.4f}",
            "单位MAE (kg/千吨)": f"{xgb_metrics['MAE_Unit']:.2f}",
            "单位R²": f"{xgb_metrics['R2_Unit']:.4f}"
        },
        {
            "预测模型": "ResNet (残差网络)",
            "MAE (kg)": f"{resnet_metrics['MAE']:.2f}",
            "RMSE (kg)": f"{resnet_metrics['RMSE']:.2f}",
            "R² 决定系数": f"{resnet_metrics['R2']:.4f}",
            "单位MAE (kg/千吨)": f"{resnet_metrics['MAE_Unit']:.2f}",
            "单位R²": f"{resnet_metrics['R2_Unit']:.4f}"
        }
    ])
    
    # print using standard pandas formatting
    print(summary_df.to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    main()
