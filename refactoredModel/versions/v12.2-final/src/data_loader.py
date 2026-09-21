# data_loader.py - 统一数据管线
# 支持从 data/ 目录自动发现 Excel 文件，定义 12 基础 + 7 衍生 = 19 维特征

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# ==================== 特征配置 ====================

# 5 基础特征：Excel 列名 → 标准特征名
BASE_FEATURES_MAP = {
    '浑浊度_0点':               '浑浊度_0点',
    '原水量_Km3_hour':         '原水量_Km3_hour',
    '温度_C':                  '温度_C',
    '氨氮_mg_per_L':           '氨氮_mg_per_L',
    'pH值':                    'pH值',
}

# 3 衍生特征：从基础列计算
DERIVED_FEATURES = {
    '浊度_氨氮_比值':     lambda df: df['浑浊度_0点'] / df['氨氮_mg_per_L'].clip(lower=0.01),
    '浊度_异常_幅度':     lambda df: np.maximum(
        0, df['浑浊度_0点'] - df['浑浊度_0点'].rolling(7, min_periods=1).mean().shift(1) * 1.3
    ),
    '浊度_氨氮_偏差':     lambda df: df['浑浊度_0点'] - df['氨氮_mg_per_L'] * (
        (df['浑浊度_0点'] / df['氨氮_mg_per_L'].clip(lower=0.01)).median()
    ),
}

DATE_COL = '日期'
TARGET_COL = '耗用矾量_kg'

# 完整 19 维特征列表（基础 + 衍生，按顺序）
ALL_FEATURES = list(BASE_FEATURES_MAP.keys()) + list(DERIVED_FEATURES.keys())

# 预测时 UI 可提供的 5 个特征
UI_FEATURES = [
    '浑浊度_0点', '原水量_Km3_hour', '温度_C', '氨氮_mg_per_L', 'pH值',
]


# ==================== Excel 自动发现 ====================

def find_data_file(data_dir='data'):
    """在 data/ 目录自动发现 Excel 数据文件"""
    os.makedirs(data_dir, exist_ok=True)
    patterns = ['*.xlsx', '*.xls']
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(os.path.join(data_dir, pattern)))
    if not candidates:
        raise FileNotFoundError(
            f"data/ 目录下未找到 Excel 文件。请将数据文件（.xlsx 或 .xls）放入 {os.path.abspath(data_dir)}/ 目录。"
        )
    # 优先选最新的文件
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


# ==================== 数据加载 ====================

def load_raw_data(excel_path=None):
    """加载原始 Excel 数据，返回 DataFrame"""
    if excel_path is None:
        excel_path = find_data_file()
    print(f"📂 读取数据: {excel_path}")
    df = pd.read_excel(excel_path, engine='openpyxl')
    print(f"   原始: {len(df)} 行 × {len(df.columns)} 列")
    return df


def clean_data(df, fill_strategy='ffill_bfill_median'):
    """
    清洗数据：日期解析 → 数值化 → 缺失值填充 → 过滤异常目标值
    """
    df = df.copy()

    # 1. 日期解析
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
        df = df.dropna(subset=[DATE_COL])
        df = df.sort_values(DATE_COL).reset_index(drop=True)

    # 2. 确保目标列存在
    if TARGET_COL not in df.columns:
        raise KeyError(f"Excel 中缺少目标列 '{TARGET_COL}'，请检查数据")

    # 3. 原水量缺失补全：2017-2020年原水量_Km3缺失，用完整的原水量_千m3补全
    #    （两列数值相同，相关系数1.0，是同一数据的不同命名）
    if '原水量_千m3' in df.columns and '原水量_Km3' in df.columns:
        missing_mask = df['原水量_Km3'].isna()
        if missing_mask.any():
            df.loc[missing_mask, '原水量_Km3'] = df.loc[missing_mask, '原水量_千m3']
            print(f"   ✅ 已用原水量_千m3 补全原水量_Km3 缺失 {missing_mask.sum()} 行")

    # 4. 原水量单位转换：从日值(Km3)转换为小时值(Km3/hour)
    if '原水量_Km3' in df.columns and '原水量_Km3_hour' not in df.columns:
        df['原水量_Km3_hour'] = df['原水量_Km3'] / 24.0
        print(f"   ✅ 原水量已转换为小时值 (原水量_Km3 → 原水量_Km3_hour)")

    # 4. 基础特征列检查
    available_features = {}
    for col, name in BASE_FEATURES_MAP.items():
        if col in df.columns:
            available_features[col] = name
        else:
            print(f"⚠️ Excel 中未找到基础特征列: {col}")
    if not available_features:
        raise KeyError("没有任何基础特征列匹配成功，请检查列名是否一致")

    # 4. 数值化所有特征列和目标列
    cols_to_numeric = list(available_features.keys()) + [TARGET_COL]
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 5. 缺失值填充
    numeric_cols = list(available_features.keys()) + [TARGET_COL]
    if fill_strategy == 'ffill_bfill_median':
        # 先用前向/后向填充（适合时序数据）
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        # 残余 NaN 用中位数填充
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)

    # 6. 过滤异常目标值
    df = df[df[TARGET_COL] > 0]
    df = df[df[TARGET_COL] < 50000]

    print(f"   清洗后: {len(df)} 行, 特征列: {len(available_features)}")
    print(f"   日期范围: {df[DATE_COL].min().date()} ~ {df[DATE_COL].max().date()}")

    return df, list(available_features.keys())


def compute_derived_features(df):
    """计算 7 个衍生特征，追加到 DataFrame"""
    df = df.copy()
    for name, func in DERIVED_FEATURES.items():
        df[name] = func(df)
        # 填充可能产生的 NaN（如滚动计算开头几行）
        if df[name].isnull().any():
            df[name] = df[name].fillna(df[name].median() if not df[name].median() != df[name].median() else 0)
            df[name] = df[name].fillna(0)
    return df


def prepare_features(df, feature_cols):
    """
    提取特征矩阵 X 和目标 y，填充 NaN
    """
    # 确保所有特征列存在
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        # 不是所有特征都可用，只使用可用的
        use_cols = [c for c in feature_cols if c in df.columns]
        print(f"⚠️ 缺少特征列: {missing}，使用 {len(use_cols)}/{len(feature_cols)} 个特征")
    else:
        use_cols = feature_cols

    X = df[use_cols].values.astype(np.float64)
    y = df[TARGET_COL].values.astype(np.float64)

    # NaN 填 0（不应发生，但安全起见）
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    return X, y, use_cols


# ==================== 训练/测试划分 ====================

def split_train_test(df, test_size=0.2, random_state=42):
    """时间序列顺序划分（保留时序，不打乱）"""
    n = len(df)
    split_idx = int(n * (1 - test_size))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"   训练集: {len(train_df)} 行, 测试集: {len(test_df)} 行")
    return train_df, test_df


# ==================== 预处理管线 ====================

def preprocess_pipeline(df, feature_cols, test_size=0.2, model_dir='models', random_split=True):
    """
    完整预处理管线：划分 → 插值 → 标准化 → 保存 scaler/imputer/features
    random_split=True: 随机划分（推荐，适合日预测场景）
    random_split=False: 时序划分（评估对未来数据的泛化能力）
    """
    os.makedirs(model_dir, exist_ok=True)

    # 1. 划分
    if random_split:
        n = len(df)
        indices = np.arange(n)
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
    else:
        train_df, test_df = split_train_test(df, test_size=test_size)

    # 2. 提取特征
    X_train, y_train, use_cols = prepare_features(train_df, feature_cols)
    X_test, y_test, _ = prepare_features(test_df, feature_cols)

    # 3. 插值
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # 4. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # 5. 保存
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(imputer, os.path.join(model_dir, 'imputer.pkl'))
    joblib.dump(use_cols, os.path.join(model_dir, 'selected_features.pkl'))

    print(f"✅ 预处理完成: {len(use_cols)} 个特征")
    print(f"   训练集: X={X_train_scaled.shape}, y={y_train.shape}")
    print(f"   测试集: X={X_test_scaled.shape}, y={y_test.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, imputer, use_cols


# ==================== 预测时用的默认值 ====================

def compute_default_values(df_clean):
    """
    从训练数据计算每个特征的中位数，供预测时补全缺失特征
    返回 dict: {特征名: 中位数值}
    """
    defaults = {}
    # 仅对 12 基础特征计算默认值（衍生特征由公式计算）
    for col in BASE_FEATURES_MAP.keys():
        if col in df_clean.columns:
            val = df_clean[col].median()
            defaults[col] = float(val) if not pd.isna(val) else 0.0
        else:
            defaults[col] = 0.0
    return defaults


if __name__ == '__main__':
    # 自检
    print("=" * 50)
    print("data_loader 自检")
    print("=" * 50)
    print(f"基础特征: {len(BASE_FEATURES_MAP)} 个")
    print(f"衍生特征: {len(DERIVED_FEATURES)} 个")
    print(f"总特征:   {len(ALL_FEATURES)} 个")
    print(f"UI特征:   {len(UI_FEATURES)} 个")
    try:
        path = find_data_file()
        print(f"✅ 找到数据文件: {path}")
        df = load_raw_data(path)
        df, available = clean_data(df)
        print(f"✅ 可用特征列: {available}")
        df = compute_derived_features(df)
        print(f"✅ 衍生特征计算完成")
        defaults = compute_default_values(df)
        print(f"✅ 默认值: { {k: f'{v:.2f}' for k, v in list(defaults.items())[:8]} }...")
    except Exception as e:
        print(f"⚠ 自检警告: {e}")
