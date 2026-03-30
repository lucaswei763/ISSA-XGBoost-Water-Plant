import pandas as pd
import sqlite3
import xgboost as xgb
import joblib
import numpy as np
from pathlib import Path

# 1. 路径自适应
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "water_plant_final.db"
MODEL_PATH = BASE_DIR.parent / "Model"

if not DB_PATH.exists() or not (MODEL_PATH / "best_issa_xgboost.json").exists():
    print("❌ 错误：数据库或已存模型不存在，请检查路径！")
else:
    # 2. 加载模型与工具
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH / "best_issa_xgboost.json"))
    scaler_x = joblib.load(str(MODEL_PATH / "scaler_x.pkl"))
    scaler_y = joblib.load(str(MODEL_PATH / "scaler_y.pkl"))

    # 3. 读取补全后的全量数据
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql('SELECT * FROM filled_data', conn)
    conn.close()

    # 4. 列名对齐 (确保与训练时一致)
    column_mapping = {
        '原水量\n（Km³）': 'flow', '浑浊度\n（NTU）': 'turbidity',
        '温度（℃）': 'temp', 'pH值': 'ph',
        '高锰酸盐指数（mg/L）': 'cod', '耗用矾量\n（kg）': 'target_dosage'
    }
    df = df.rename(columns=column_mapping)
    features = ['flow', 'turbidity', 'temp', 'ph', 'cod']

    # 5. 执行全量预测
    X_scaled = scaler_x.transform(df[features].values)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = df['target_dosage'].values

    # 6. 计算绝对误差并找茬
    df['Abs_Error'] = np.abs(y_true - y_pred)

    # 按误差降序排列，取前 20 名“犯罪嫌疑人”
    suspects = df.sort_values(by='Abs_Error', ascending=False).head(20)

    print("\n" + "!" * 20 + " 全量数据排雷报告 " + "!" * 20)
    print(f"当前平均 RMSE 为 {np.sqrt(np.mean((y_true - y_pred) ** 2)):.4f}")
    print("\n以下 20 天的数据极度异常，建议回 Excel 查看原始记录：")

    # 打印关键列：日期、原始药量、预测药量、绝对误差、浊度
    display_cols = ['date', 'target_dosage', 'flow', 'turbidity', 'Abs_Error']
    print(suspects[display_cols].to_string(index=False))

    print("\n" + "!" * 60)
    print("💡 请检查这些日期的 'target_dosage' 是否多写了一个零，或者单位错了。")