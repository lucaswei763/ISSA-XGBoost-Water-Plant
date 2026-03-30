import json
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import time
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import csv
from datetime import datetime

# 基础配置
plt.rcParams['font.sans-serif'] = ['Heiti TC']  # Mac使用，Windows建议换成 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from ISSA_Module import ISSA_XGBoost_Optimizer


# --- 工具函数 ---
def clean_string_to_float(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int, float)): return float(val)
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
    return float(nums[0]) if nums else np.nan


def generate_statistical_evidence(df, features, output_dir):
    """
    统计学证据：计算各维度与投药量的 Spearman 相关性
    """
    print("\n🔍 正在分析统计学证据...")
    corr_matrix = df[features + ['target_dosage']].corr(method='spearman')
    dosage_corr = corr_matrix['target_dosage'].sort_values(ascending=False)

    print("--- Spearman 相关系数排行 ---")
    print(dosage_corr)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("各维度与投药量统计相关性热力图")
    plt.savefig(os.path.join(output_dir, '5_spearman_correlation.png'))
    plt.close()
    return dosage_corr


def add_advanced_features(df):
    """注入 8D 金牌特征"""
    df = df.sort_values('date')
    if '氨氮\n（mg/L）' in df.columns:
        df['ammonia'] = df['氨氮\n（mg/L）'].apply(clean_string_to_float)
        df['ammonia'] = df['ammonia'].fillna(df['ammonia'].median())

    df['last_turbidity'] = df['turbidity'].shift(1)
    df['last_dosage'] = df['target_dosage'].shift(1)
    df['flow_turbidity_inter'] = df['flow'] * df['turbidity']
    return df.dropna()


def log_training_history(metrics_dict, params_dict, file_path='history.csv'):
    """记录每一次训练的成绩，方便观察进化过程"""
    file_exists = os.path.isfile(file_path)
    log_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        **metrics_dict, **params_dict
    }
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(log_data)
    print(f"✅ 进化历程已记录至: {file_path}")


def compare_params(new_params, params_path):
    """对比新旧参数，特别是监控 gamma 是否在变大以抵御噪声"""
    if not os.path.exists(params_path): return
    with open(params_path, 'r') as f:
        old = json.load(f)
    print("\n" + "🔄" * 10 + " 参数进化对比 " + "🔄" * 10)
    for k in ['n_estimators', 'learning_rate', 'max_depth', 'gamma']:
        old_v, new_v = old.get(k, 0), new_params.get(k, 0)
        diff = new_v - old_v
        symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"   - {k:15}: {old_v:.4f} -> {new_v:.4f} ({symbol} {abs(diff):.4f})")
    print("=" * 35 + "\n")


def main():
    # 1. 数据加载与清洗
    conn = sqlite3.connect('water_plant_final.db')
    df = pd.read_sql_query("SELECT * FROM filled_data", conn)
    conn.close()

    mapping = {'矾\n（kg/Km³）': 'target_dosage', '浑浊度\n（NTU）': 'turbidity',
               'pH值': 'ph', '温度（℃）': 'temp', '原水量\n（Km³）': 'flow'}
    df.rename(columns=mapping, inplace=True)
    df['date'] = pd.to_datetime(df['date'])

    # 数据清洗：手动异常 + IQR 统计过滤
    df = df[df['target_dosage'] < 30]
    Q1, Q3 = df['target_dosage'].quantile(0.25), df['target_dosage'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['target_dosage'] < (Q1 - 3 * IQR)) | (df['target_dosage'] > (Q3 + 3 * IQR)))]

    df = add_advanced_features(df)
    features = ['turbidity', 'ph', 'temp', 'flow', 'ammonia',
                'last_turbidity', 'last_dosage', 'flow_turbidity_inter']

    # 2. 产出统计学证据
    output_dir = os.path.join(current_dir, 'Analysis_Results')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    generate_statistical_evidence(df, features, output_dir)

    # 3. 准备模型数据 (核心修复区：关闭 shuffle 保证时序，增加独立验证集)
    X = df[features].values
    y = df['target_dosage'].values

    # 第一次切分：留出最后 15% 作为盲测集 (Test)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

    # 第二次切分：将前面的 85% 再次切分出 15% 给 ISSA 做验证集 (Val)
    # 0.176 * 0.85 ≈ 0.15
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, shuffle=False)

    # 4. 启动 ISSA 寻优 (传入验证集而非测试集)
    print(f"🚀 启动 ISSA 寻优...")
    optimizer = ISSA_XGBoost_Optimizer(X_train, y_train, X_val, y_val, pop_size=32, max_iter=200)
    best_params, _ = optimizer.optimize()

    # 5. 训练最终模型 (核心修复区：对齐 subsample 参数，使用全量非测试数据)
    final_model = xgb.XGBRegressor(
        n_estimators=int(best_params[0]),
        learning_rate=best_params[1],
        max_depth=int(best_params[2]),
        gamma=best_params[3],
        subsample=0.8,  # 必须与 ISSA 中保持一致
        colsample_bytree=0.8,  # 必须与 ISSA 中保持一致
        random_state=42
    )

    # 最终模型可以使用 (训练集 + 验证集) 合并训练，以喂给模型尽可能多的数据
    final_model.fit(X_temp, y_temp)

    # 防过拟合诊断报告
    train_r2 = r2_score(y_temp, final_model.predict(X_temp))
    # 这里的测试集是从未参与过任何寻优和训练的纯净盲测集
    test_r2 = r2_score(y_test, final_model.predict(X_test))
    gap = abs(train_r2 - test_r2)

    # 核心修复区：使用时序交叉验证代替随机 KFold
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(final_model, X, y, cv=tscv, scoring='r2')

    print("\n" + "=" * 50)
    print(f"🛡️  模型泛化能力评估报告 (应对水质变化敏感度分析)")
    print(f"   1. 训练&验证集 R²: {train_r2:.4f}")
    print(f"   2. 盲测集 R²:      {test_r2:.4f}")
    print(f"   3. 泛化间隙:       {gap:.4f} (差异越小，泛化越强)")
    print(f"   4. 5折时序验证 R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    if gap < 0.05 and cv_scores.std() < 0.05:  # 时序数据的 std 波动往往略大，可适当放宽至 0.05
        print("✅ 统计学判定：模型不存在显著过拟合，具备工程应用价值。")
    else:
        print("⚠️ 统计学判定：模型存在过拟合风险，或受时序突变影响显著。")
    print("=" * 50)

    metrics = {
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'gap': float(gap),
        'cv_mean': float(cv_scores.mean())
    }

    new_params_dict = {
        'n_estimators': int(best_params[0]),
        'learning_rate': float(best_params[1]),
        'max_depth': int(best_params[2]),
        'gamma': float(best_params[3]),
        'subsample': 0.8,  # 同步记录进配置文件
        'colsample_bytree': 0.8,  # 同步记录进配置文件
        'objective': 'reg:squarederror'
    }

    params_path = os.path.join(current_dir, 'Model', 'best_issa_params.json')

    compare_params(new_params_dict, params_path)
    log_training_history(metrics, new_params_dict)

    # 6. 固化资产
    model_dir = os.path.join(current_dir, 'Model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(new_params_dict, f, indent=4)

    final_model.save_model(os.path.join(model_dir, 'best_issa_xgboost.json'))
    print(f"📂 模型资产已固化入库。")


if __name__ == "__main__":
    main()