import os
import sys
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from data_loader import (
    load_raw_data, clean_data, compute_derived_features,
    prepare_features, split_train_test, preprocess_pipeline,
    ALL_FEATURES, TARGET_COL, DATE_COL, BASE_FEATURES_MAP, UI_FEATURES,
)

# 设定中文字体，确保绘图时不乱码
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Songti SC', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 特征配置（从 data_loader 继承） ====================
FEATURE_COLS = ALL_FEATURES  # 19 维


def setup_logger_and_dir(model_prefix):
    """
    建立按日期命名的目录，例如 outputs/xgb_train-4.11
    并将 sys.stdout 重定向到该目录下以当前时间命名的 txt 记录文件中
    """
    now = datetime.datetime.now()
    date_str = f"{now.month}.{now.day}"
    run_dir = os.path.join("outputs", f"{model_prefix}-{date_str}")
    os.makedirs(run_dir, exist_ok=True)

    time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(run_dir, f"{time_str}.txt")

    class DualLogger:
        def __init__(self, log_path):
            self.terminal = sys.stdout
            self.log = open(log_path, "a", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = DualLogger(log_file)
    print(f"📄 开始记录日志至: {log_file}")
    
    return run_dir


def load_and_preprocess_data(test_size=0.2, excel_path=None, random_split=True):
    """使用新 data_loader 管线：Excel → 清洗 → 衍生特征 → 划分 → 插值 → 标准化
    random_split=True: 随机划分（推荐用于日预测场景）"""
    # 1. 加载原始数据
    df = load_raw_data(excel_path)
    df, available_features = clean_data(df)

    # 2. 计算衍生特征（生成 19 维完整特征）
    df = compute_derived_features(df)

    # 3. 使用完整 19 维特征
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    used = len(feature_cols)
    skipped = len(ALL_FEATURES) - used
    if skipped > 0:
        missing = [c for c in ALL_FEATURES if c not in df.columns]
        print(f"⚠️ {skipped}/{len(ALL_FEATURES)} 个特征不可用: {missing}")
    print(f"   使用特征: {used}/{len(ALL_FEATURES)}")

    # 4. 预处理管线
    X_train_s, X_test_s, y_train, y_test, scaler, imputer, cols_used = \
        preprocess_pipeline(df, feature_cols, test_size=test_size, random_split=random_split)

    # 5. 全量数据（用于绘图）
    X_full, y_full, _ = prepare_features(df, feature_cols)
    X_full_imputed = imputer.transform(X_full)
    X_full_scaled = scaler.transform(X_full_imputed)
    full_dates = df[DATE_COL].values if DATE_COL in df.columns else None

    # 原水量列索引（用于单位投矾量计算）
    water_idx = None
    for i, c in enumerate(cols_used):
        if '原水量' in c:
            water_idx = i
            break
    water_test = X_test_s[:, water_idx] if water_idx is not None else None
    water_full = X_full_scaled[:, water_idx] if water_idx is not None else None
    # 需要原始原水量（非标准化）用于单位计算
    if water_idx is not None:
        X_test_raw = imputer.transform(prepare_features(df.iloc[-len(y_test):], feature_cols)[0])
        water_test_raw = X_test_raw[:, water_idx]
        water_full_raw = X_full[:, water_idx]
    else:
        water_test_raw = None
        water_full_raw = None

    print(f"✅ 数据管线完成: {used} 特征, 训练集 {len(X_train_s)}, 测试集 {len(X_test_s)}")
    if '原水量_Km3_hour' in df.columns:
        print(f"   原水量统计: mean={df['原水量_Km3_hour'].mean():.1f}, median={df['原水量_Km3_hour'].median():.1f}")

    return X_train_s, X_test_s, y_train, y_test, X_full_scaled, y_full, full_dates, water_test_raw, water_full_raw


def evaluate_and_plot(y_test, y_pred, y_full, y_full_pred, full_dates, model_name, plot_dir="outputs", water_test=None,
                      water_full=None):
    """
    计算指标（仅依靠测试集），并根据 full_dates 将全量预测结果按年份分组并分别绘制实际-预测图。
    这样保证图表是逐天连续的。
    """
    os.makedirs(plot_dir, exist_ok=True)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 {model_name} 测试集评估结果 (总投矾量):")
    print(f" -> MAE  (平均绝对误差): {mae:.2f} kg")
    print(f" -> RMSE (均方根误差):   {rmse:.2f} kg")
    print(f" -> R²   (决定系数):     {r2:.4f}")

    if water_test is not None:
        safe_water_test = np.where(water_test == 0, 1e-5, water_test)
        y_test_unit = y_test / safe_water_test
        y_pred_unit = y_pred / safe_water_test

        mae_unit = mean_absolute_error(y_test_unit, y_pred_unit)
        rmse_unit = np.sqrt(mean_squared_error(y_test_unit, y_pred_unit))
        r2_unit = r2_score(y_test_unit, y_pred_unit)

        print(f"\n📊 {model_name} 测试集评估结果 (投矾量/千吨水):")
        print(f" -> MAE  (平均绝对误差): {mae_unit:.2f} kg/千吨水")
        print(f" -> RMSE (均方根误差):   {rmse_unit:.2f} kg/千吨水")
        print(f" -> R²   (决定系数):     {r2_unit:.4f}")

    if full_dates is None:
        print("⚠️ 警告: 未找到日期数据，无法按年份连续绘图。")
        return mae, rmse, r2

    results_df = pd.DataFrame({
        'Date': pd.to_datetime(full_dates),
        'Actual': y_full,
        'Predicted': y_full_pred
    })

    results_df.dropna(subset=['Date'], inplace=True)
    results_df['Year'] = results_df['Date'].dt.year
    years = results_df['Year'].unique()

    if water_full is not None:
        safe_water_full = np.where(water_full == 0, 1e-5, water_full)
        results_df['Actual_Unit'] = results_df['Actual'] / safe_water_full
        results_df['Predicted_Unit'] = results_df['Predicted'] / safe_water_full

    for year in sorted(years):
        # 取该年的全量并排序
        year_df = results_df[results_df['Year'] == year].sort_values(by='Date')
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 逐天绘制连续折线：总投矾量
        axes[0].plot(year_df['Date'], year_df['Actual'], label='实际总耗用量', marker='o', markersize=3, alpha=0.8)
        axes[0].plot(year_df['Date'], year_df['Predicted'], label=f'{model_name}预测总耗用量', marker='x', markersize=3, alpha=0.8)
        axes[0].set_title(f'{model_name} 预测总投矾量表现 - {year}年 (全景视角)')
        axes[0].set_ylabel('耗用矾量 (kg)')
        axes[0].legend()
        axes[0].grid(True)
        
        # 逐天绘制连续折线：单位投矾量
        if 'Actual_Unit' in year_df.columns:
            axes[1].plot(year_df['Date'], year_df['Actual_Unit'], label='实际投矾量/千吨水', marker='o', markersize=3, alpha=0.8, color='green')
            axes[1].plot(year_df['Date'], year_df['Predicted_Unit'], label=f'{model_name}预测投矾量/千吨水', marker='x', markersize=3, alpha=0.8, color='orange')
            axes[1].set_title(f'{model_name} 预测投矾量/千吨水表现 - {year}年 (全景视角)')
            axes[1].set_xlabel('日期')
            axes[1].set_ylabel('投矾量 (kg/Km³)')
            axes[1].legend()
            axes[1].grid(True)
            
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        
        plot_path = os.path.join(plot_dir, f'{model_name.lower().replace("-", "_")}_predict_{year}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ {year}年预测对比图已保存至: {plot_path}")

    return mae, rmse, r2
