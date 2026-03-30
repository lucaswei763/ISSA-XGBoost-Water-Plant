import json
import os
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import re  # 引入正则
from sklearn.metrics import r2_score
from xgboost import plot_importance

# --- 0. 环境配置 ---
plt.rcParams['font.sans-serif'] = ['Heiti TC']  # Mac
plt.rcParams['axes.unicode_minus'] = False
current_dir = os.path.dirname(os.path.abspath(__file__))


# 强力字符串清洗函数，把 "0.5mg/L" 变成 0.5
def clean_string_to_float(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int, float)): return float(val)
    # 正则提取数字
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
    return float(nums[0]) if nums else np.nan


def load_data_8d(db_path="water_plant_final.db"):
    print("📂 正在读取数据库并进行深度清洗...")
    conn = sqlite3.connect(db_path)
    sql = 'SELECT "date", "pH值" as ph, "浑浊度\n（NTU）" as turbidity, "温度（℃）" as temp, "原水量\n（Km³）" as flow, "氨氮\n（mg/L）" as ammonia, "矾\n（kg/Km³）" as dosage FROM filled_data'
    df = pd.read_sql_query(sql, conn)
    conn.close()

    # --- 清洗函数 ---
    # 先处理投药量（目标值）
    df['dosage'] = df['dosage'].apply(clean_string_to_float)
    # 手动去除异常点，保持和 train.py 一致
    df = df[df['dosage'] < 30].dropna(subset=['dosage'])

    # 再处理氨氮和其他特征列
    df['ammonia'] = df['ammonia'].apply(clean_string_to_float)
    df['ammonia'] = df['ammonia'].fillna(df['ammonia'].median())  # 补齐缺失值

    for col in ['ph', 'turbidity', 'temp', 'flow']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').dropna()

    # 8D 特征构建
    df['last_turbidity'] = df['turbidity'].shift(1)
    df['last_dosage'] = df['dosage'].shift(1)
    df['flow_turbidity_inter'] = df['flow'] * df['turbidity']
    df = df.dropna().reset_index(drop=True)

    features = ['turbidity', 'ph', 'temp', 'flow', 'ammonia', 'last_turbidity', 'last_dosage', 'flow_turbidity_inter']
    return df[features], df['dosage'], df


def run_visualization():
    output_dir = os.path.join(current_dir, 'Analysis_Results')
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # 1. 加载参数 (由 train.py 产出)
    params_path = os.path.join(current_dir, 'Model', 'best_issa_params.json')
    with open(params_path, 'r') as f:
        config = json.load(f)

    # 2. 获取干净数据
    X, y, df = load_data_8d()

    # 3. 训练展示模型并预测
    print(f"🤖 正在使用 8D 特征加载模型大脑...")
    model = xgb.XGBRegressor(**config)
    model.fit(X, y)
    df['predict'] = model.predict(X)

    # 4. 生成年度图 (循环逻辑)
    df['year'] = df['date'].dt.year
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        plt.figure(figsize=(16, 7))
        plt.plot(year_df['date'], year_df['dosage'], label='实际值', color='#1f77b4', alpha=0.8)
        plt.plot(year_df['date'], year_df['predict'], label='预测值', color='#ff7f0e', linestyle='--', alpha=0.9)
        plt.title(f'{year}年度每日拟合 (R2: {r2_score(year_df["dosage"], year_df["predict"]):.4f})')
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig(os.path.join(output_dir, f'1_daily_fit_{year}.png'), dpi=300)
        plt.close()
        print(f"   ✅ 已更新年度图: {year}")

    # 5. 全局图
    plt.figure(figsize=(10, 6))
    model.get_booster().feature_names = X.columns.tolist()
    plot_importance(model, importance_type='gain', title='8D 特征贡献度 (Gain)')
    plt.savefig(os.path.join(output_dir, '2_feature_importance.png'), dpi=300)
    plt.close()

    print(f"✨ 修正成功！图片已存入: {output_dir}")


# 在 visible.py 中添加该函数定义
def generate_statistical_evidence(df, model, features, output_dir):
    # 1. 计算 Spearman 相关性矩阵
    # 使用 Spearman 是因为水质与药耗往往是非线性关系
    corr_matrix = df[features + ['dosage']].corr(method='spearman')
    dosage_corr = corr_matrix['dosage'].sort_values(ascending=False)

    print("\n--- 统计学证据：Spearman 相关系数排行 ---")
    print(dosage_corr)

    # 2. 绘制相关性热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("各维度与投药量统计相关性热力图")
    plt.savefig(os.path.join(output_dir, '5_spearman_correlation.png'))
    plt.close()
    return dosage_corr


if __name__ == "__main__":
    run_visualization()