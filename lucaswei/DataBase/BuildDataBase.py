import sqlite3
import pandas as pd
import numpy as np


def rebuild_clean_data(db_path):
    conn = sqlite3.connect(db_path)
    # 1. 从已经填补完空缺值的 filled_data 读取
    df = pd.read_sql_query("SELECT * FROM filled_data", conn)

    # 2. 核心映射（处理换行符）
    mapping = {
        '浑浊度\n（NTU）': 'turbidity',
        'pH值': 'ph',
        '温度（℃）': 'temp',
        '高锰酸盐指数（mg/L）': 'cod',
        '原水量\n（Km³）': 'flow',
        '矾\n（kg/Km³）': 'target_dosage'
    }
    df = df.rename(columns=mapping)

    # 3. 仅保留需要的 5D 特征 + 1 标签
    cols_to_keep = ['date', 'turbidity', 'ph', 'temp', 'cod', 'flow', 'target_dosage']
    df = df[cols_to_keep]

    # 4. 采用更温和的离群点处理 (Z-score > 4)
    # 只有当数据偏离均值 4 倍标准差时才认为是录入错误，否则视为极端工况
    before_count = len(df)
    for col in ['turbidity', 'ph', 'temp', 'cod', 'flow']:
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        df = df[np.abs(z_scores) < 4.0]  # 阈值设为 4，保留更多极端但真实的数据

    after_count = len(df)
    print(f"数据清洗完成：{before_count} -> {after_count} (保留率: {after_count / before_count:.1%})")

    # 5. 写入数据库覆盖原有的 clean_data
    df.to_sql('clean_data', conn, if_exists='replace', index=False)
    conn.close()


if __name__ == "__main__":
    rebuild_clean_data('water_plant_final.db')