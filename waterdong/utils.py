import os
import sys
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from build_database import WaterDataLoader

# 设定中文字体
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Songti SC', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 重要：定义特征列和目标列（匹配数据库实际列名） ====================
TARGET_COL = '耗用矾量（kg）'
FEATURE_COLS = [
    '浑浊度（NTU）_0点',  # 注意：使用 _0点 匹配数据库
    '原水量（Km³）',
    '温度（℃）_9点',
    '氨氮（mg/L）_9点',
    'pH值_9点'
]


def setup_logger_and_dir(model_prefix):
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


def load_and_preprocess_data():
    loader = WaterDataLoader()
    try:
        df = loader.get_all_data()
    except Exception as e:
        raise RuntimeError(f"数据库读取失败: {e}")

    df.replace(['/', '\\', '', ' '], np.nan, inplace=True)

    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        print(f"⚠️ 警告: 数据找不到对应的列 {missing_cols}")

    df = df.dropna(subset=[TARGET_COL]).copy()

    date_col = '日期' if '日期' in df.columns else None
    cols_to_convert = FEATURE_COLS + [TARGET_COL]
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

    if date_col:
        mask_df = df[FEATURE_COLS + [TARGET_COL] + [date_col]]
    else:
        mask_df = df[FEATURE_COLS + [TARGET_COL]]

    idx = mask_df.index
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    train_data = mask_df.loc[train_idx].copy()
    test_data = mask_df.loc[test_idx].copy()

    imputer = SimpleImputer(strategy='median')
    train_data[FEATURE_COLS] = imputer.fit_transform(train_data[FEATURE_COLS])
    test_data[FEATURE_COLS] = imputer.transform(test_data[FEATURE_COLS])

    full_data = mask_df.copy()
    if date_col:
        full_data = full_data.sort_values(by=date_col)
    full_data[FEATURE_COLS] = imputer.transform(full_data[FEATURE_COLS])

    X_train = train_data[FEATURE_COLS].values
    y_train = train_data[TARGET_COL].values
    X_test = test_data[FEATURE_COLS].values
    y_test = test_data[TARGET_COL].values

    X_full = full_data[FEATURE_COLS].values
    y_full = full_data[TARGET_COL].values
    full_dates = full_data[date_col].values if date_col else None

    water_test = X_test[:, 1] if X_test.shape[1] > 1 else None
    water_full = X_full[:, 1] if X_full.shape[1] > 1 else None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_full_scaled = scaler.transform(X_full)

    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(FEATURE_COLS, 'models/selected_features.pkl')
    joblib.dump(imputer, 'models/imputer.pkl')

    print(f"✅ 有效数据量: {len(mask_df)} 条 (训练集: {len(X_train)}, 测试集: {len(X_test)})。")
    print("原水量（Km³）描述统计：")
    print(df['原水量（Km³）'].describe())
    return X_train_scaled, X_test_scaled, y_train, y_test, X_full_scaled, y_full, full_dates, water_test, water_full


def evaluate_and_plot(y_test, y_pred, y_full, y_full_pred, full_dates, model_name, plot_dir="outputs", water_test=None,
                      water_full=None):
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
        return

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
        year_df = results_df[results_df['Year'] == year].sort_values(by='Date')
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        axes[0].plot(year_df['Date'], year_df['Actual'], label='实际总耗用量', marker='o', markersize=3, alpha=0.8)
        axes[0].plot(year_df['Date'], year_df['Predicted'], label=f'{model_name}预测总耗用量', marker='x', markersize=3,
                     alpha=0.8)
        axes[0].set_title(f'{model_name} 预测总投矾量表现 - {year}年')
        axes[0].set_ylabel('耗用矾量 (kg)')
        axes[0].legend()
        axes[0].grid(True)
        if 'Actual_Unit' in year_df.columns:
            axes[1].plot(year_df['Date'], year_df['Actual_Unit'], label='实际投矾量/千吨水', marker='o', markersize=3,
                         alpha=0.8, color='green')
            axes[1].plot(year_df['Date'], year_df['Predicted_Unit'], label=f'{model_name}预测投矾量/千吨水', marker='x',
                         markersize=3, alpha=0.8, color='orange')
            axes[1].set_title(f'{model_name} 预测投矾量/千吨水表现 - {year}年')
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