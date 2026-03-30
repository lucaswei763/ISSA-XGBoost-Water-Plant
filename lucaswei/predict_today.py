import os
import sqlite3
import pandas as pd
import xgboost as xgb

# --- 1. 路径自动定位 ---
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义模型路径 (固定)
model_path = os.path.join(current_dir, 'DataBase', 'Model', 'best_issa_xgboost.json')

# 数据库路径逻辑：优先寻找脱敏后的 sample.db
sample_path = os.path.join(current_dir, 'DataBase', 'sample.db')
original_path = os.path.join(current_dir, 'DataBase', 'water_plant_final.db')

# 自动切换数据库：如果 sample.db 存在，则使用它；否则回退到原始数据库
db_path = sample_path if os.path.exists(sample_path) else original_path



def get_user_input(prompt):
    """安全获取用户输入的数字"""
    while True:
        try:
            val = input(prompt)
            return float(val)
        except ValueError:
            print("⚠️ 输入无效，请输入数字（例如: 2.5）")


def run_interactive_predict():
    # --- 2. 自动提取“昨日记忆” ---
    sql = """
    SELECT "矾\n（kg/Km³）" as dosage, "浑浊度\n（NTU）" as turbidity 
    FROM filled_data ORDER BY "date" DESC LIMIT 1
    """
    try:
        conn = sqlite3.connect(db_path)
        last_record = pd.read_sql(sql, conn)
        conn.close()
        last_dosage = float(last_record['dosage'].iloc[0])
        last_turbidity = float(last_record['turbidity'].iloc[0])
        print(f"🧠 已自动加载昨日记忆：投药量 {last_dosage:.2f}, 浊度 {last_turbidity:.2f}")
    except Exception as e:
        print(f"❌ 数据库读取失败: {e}")
        return

    # --- 3. 开启交互式输入 ---
    print("\n--- 📝 请输入当前实时监测数据 ---")
    turb = get_user_input("📍 当前原水浊度 (NTU): ")
    ph = get_user_input("📍 当前 pH 值: ")
    temp = get_user_input("📍 当前水温 (℃): ")
    flow = get_user_input("📍 当前原水量 (Km³): ")
    ammo = get_user_input("📍 当前氨氮含量 (mg/L): ")

    # --- 4. 构建 8D 特征向量 ---
    features_data = {
        'turbidity': [turb],
        'ph': [ph],
        'temp': [temp],
        'flow': [flow],
        'ammonia': [ammo],
        'last_turbidity': [last_turbidity],
        'last_dosage': [last_dosage],
        'flow_turbidity_inter': [flow * turb]
    }
    X_input = pd.DataFrame(features_data)

    # --- 5. 加载模型并推理 ---
    try:
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        suggested_dosage = model.predict(X_input)[0]

        print("\n" + "=" * 40)
        print(f"🚀 ISSA-XGBoost 建议投药量：{suggested_dosage:.3f} kg/Km³")
        print("=" * 40 + "\n")
    except Exception as e:
        print(f"❌ 模型推理失败: {e}")


if __name__ == "__main__":
    run_interactive_predict()