#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
水厂投矾量数据分析与建模系统
================================
功能：
1. 数据加载与合并
2. 特征工程与数据清洗
3. 相关性分析与特征选择
4. 多模型训练与对比
5. 模型诊断与可视化
6. 模型保存与部署
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# ==================== 自动创建目录 ====================
required_dirs = ['data', 'models', 'outputs']
for dir_name in required_dirs:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"✓ 创建目录: {dir_name}")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (12, 8)

# 设置随机种子
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

print("=" * 80)
print("水厂投矾量数据分析与建模系统")
print("=" * 80)

# ==================== 第一部分：数据加载 ====================
print("\n【步骤1/6】数据加载与合并...")


def find_date_column(df):
    """智能查找日期列"""
    date_keywords = ['日期', 'date', '时间', 'time', '日', 'day', 'Date', 'Time']
    for col in df.columns:
        col_lower = col.lower()
        for keyword in date_keywords:
            if keyword in col_lower:
                return col
    return None


def load_data():
    """加载并合并数据库中的数据"""
    db_path = 'data/water_data.db'

    if not os.path.exists(db_path):
        print(f"错误：数据库文件不存在 - {db_path}")
        print("请确保 water_data.db 文件位于 data/ 目录下")
        sys.exit(1)

    conn = sqlite3.connect(db_path)

    # 查询所有表
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    print(f"✓ 数据库表: {tables}")

    # 尝试加载合并后的数据
    if 'merged_data' in tables:
        df = pd.read_sql_query("SELECT * FROM merged_data", conn)
        print(f"✓ 成功加载合并数据表，形状: {df.shape}")

        # 查找日期列
        date_col = find_date_column(df)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.rename(columns={date_col: '日期'}, inplace=True)
            print(f"✓ 找到日期列: {date_col}")
        else:
            print("⚠ 警告: 未找到日期列，将使用索引作为日期")
            df['日期'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
    else:
        # 分别加载药耗和水质数据
        consumption_table = None
        quality_table = None

        for table in tables:
            if 'consumption' in table.lower() or '药耗' in table:
                consumption_table = table
            elif 'quality' in table.lower() or '水质' in table:
                quality_table = table

        if consumption_table is None:
            consumption_table = tables[0]
        if quality_table is None:
            quality_table = tables[1] if len(tables) > 1 else tables[0]

        print(f"使用药耗数据表: {consumption_table}")
        print(f"使用水质数据表: {quality_table}")

        consumption_df = pd.read_sql_query(f"SELECT * FROM {consumption_table}", conn)
        quality_df = pd.read_sql_query(f"SELECT * FROM {quality_table}", conn)

        # 查找日期列
        date_col_c = find_date_column(consumption_df)
        date_col_q = find_date_column(quality_df)

        if date_col_c is None:
            print(f"⚠ 警告: 药耗表未找到日期列，使用第一列")
            date_col_c = consumption_df.columns[0]
        if date_col_q is None:
            print(f"⚠ 警告: 水质表未找到日期列，使用第一列")
            date_col_q = quality_df.columns[0]

        print(f"药耗表日期列: {date_col_c}")
        print(f"水质表日期列: {date_col_q}")

        consumption_df[date_col_c] = pd.to_datetime(consumption_df[date_col_c])
        quality_df[date_col_q] = pd.to_datetime(quality_df[date_col_q])

        df = pd.merge(consumption_df, quality_df,
                      left_on=date_col_c, right_on=date_col_q, how='inner')

        # 统一日期列名
        df.rename(columns={date_col_c: '日期'}, inplace=True)
        if date_col_q != date_col_c and date_col_q in df.columns:
            df.drop(columns=[date_col_q], inplace=True)

        print(f"✓ 数据合并完成，形状: {df.shape}")

    conn.close()
    return df


df = load_data()

# 识别目标变量
target_col = None
target_keywords = ['矾', 'alum', '投矾', '药耗', '矾耗', 'dosage']

for col in df.columns:
    col_lower = col.lower()
    for kw in target_keywords:
        if kw in col_lower and 'lag' not in col_lower:
            target_col = col
            break
    if target_col:
        break

if target_col is None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    target_col = numeric_cols[0] if len(numeric_cols) > 0 else None

if target_col is None:
    print("错误：无法识别目标变量")
    sys.exit(1)

print(f"✓ 目标变量: {target_col}")
print(f"✓ 数据时间范围: {df['日期'].min()} 至 {df['日期'].max()}")
print(f"✓ 数据样本数: {len(df)} 条")
print(f"✓ 数据列数: {len(df.columns)} 列")

# ==================== 第二部分：数据清洗 ====================
print("\n【步骤2/6】数据清洗与异常值处理...")

df_clean = df.copy()
initial_shape = df_clean.shape

# 1. 处理缺失值
print("\n▶ 处理缺失值...")
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
if '日期' in numeric_cols:
    numeric_cols.remove('日期')

# 计算缺失率
if len(numeric_cols) > 0:
    missing_rate = df_clean[numeric_cols].isnull().sum() / len(df_clean) * 100
    high_missing = missing_rate[missing_rate > 30].index.tolist()

    if high_missing:
        df_clean = df_clean.drop(columns=high_missing)
        print(f"  ✓ 删除缺失率>30%的列: {high_missing}")
        numeric_cols = [c for c in numeric_cols if c not in high_missing]

# 填充缺失值
if len(numeric_cols) > 0:
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill')
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='bfill')
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)

print(f"  ✓ 缺失值处理完成，剩余缺失值: {df_clean.isnull().sum().sum()}")

# 2. 异常值处理（IQR方法）
print("\n▶ 处理异常值...")
outlier_count = 0
for col in numeric_cols:
    if col != target_col:
        try:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
                if outliers > 0:
                    df_clean[col] = df_clean[col].clip(lower, upper)
                    outlier_count += outliers
        except Exception as e:
            print(f"  处理 {col} 时出错: {e}")

print(f"  ✓ 异常值处理完成，共处理 {outlier_count} 个异常值")

print(f"✓ 数据清洗完成，形状: {df_clean.shape} (原始: {initial_shape})")

# ==================== 第三部分：特征工程 ====================
print("\n【步骤3/6】特征工程...")

# 1. 时间特征
print("\n▶ 构造时间特征...")
df_clean['年'] = df_clean['日期'].dt.year
df_clean['月'] = df_clean['日期'].dt.month
df_clean['日'] = df_clean['日期'].dt.day
df_clean['星期几'] = df_clean['日期'].dt.dayofweek
df_clean['是否为周末'] = (df_clean['星期几'] >= 5).astype(int)


# 2. 季节特征
def get_season(month):
    if month in [3, 4, 5]:
        return '春'
    elif month in [6, 7, 8]:
        return '夏'
    elif month in [9, 10, 11]:
        return '秋'
    else:
        return '冬'


df_clean['季节'] = df_clean['月'].apply(get_season)

# 3. 滞后特征
print("▶ 构造滞后特征...")
for lag in [1, 2, 3, 7]:
    df_clean[f'{target_col}_lag_{lag}天'] = df_clean[target_col].shift(lag)

# 4. 滚动统计特征
print("▶ 构造滚动统计特征...")
# 查找浊度和流量列
turbidity_col = None
flow_col = None
for col in df_clean.columns:
    if '浊度' in col or 'turbidity' in col.lower():
        turbidity_col = col
    if '流量' in col or 'flow' in col.lower():
        flow_col = col

if turbidity_col:
    df_clean[f'{turbidity_col}_7天均值'] = df_clean[turbidity_col].rolling(7, min_periods=1).mean()
    df_clean[f'{turbidity_col}_7天标准差'] = df_clean[turbidity_col].rolling(7, min_periods=1).std()
    print(f"  ✓ 基于 {turbidity_col} 创建滚动特征")

if flow_col:
    df_clean[f'{flow_col}_7天均值'] = df_clean[flow_col].rolling(7, min_periods=1).mean()
    df_clean[f'{flow_col}_7天标准差'] = df_clean[flow_col].rolling(7, min_periods=1).std()
    print(f"  ✓ 基于 {flow_col} 创建滚动特征")

# 5. 交互特征
print("▶ 构造交互特征...")
if turbidity_col and flow_col:
    df_clean['浊度_流量_交互'] = df_clean[turbidity_col] * df_clean[flow_col]
    print(f"  ✓ 创建交互特征: 浊度×流量")

# 6. 填充新特征缺失值
new_cols = [c for c in df_clean.columns if c not in df.columns]
if new_cols:
    df_clean[new_cols] = df_clean[new_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)

# 7. 季节独热编码
if '季节' in df_clean.columns:
    season_dummies = pd.get_dummies(df_clean['季节'], prefix='季节')
    df_clean = pd.concat([df_clean, season_dummies], axis=1)
    df_clean = df_clean.drop('季节', axis=1)

# 保存特征工程数据
df_clean.to_csv('outputs/feature_engineered_data.csv', index=False, encoding='utf-8-sig')
print(f"✓ 特征工程完成，最终特征数: {len(df_clean.columns) - 1}")
print(f"✓ 数据已保存: outputs/feature_engineered_data.csv")

# ==================== 第四部分：相关性分析 ====================
print("\n【步骤4/6】相关性分析与特征选择...")

# 排除时间特征和构造特征
exclude_patterns = ['日期', '年', '月', '日', '星期', 'lag_', '均值', '标准差', '季节', '滚动']
feature_candidates = [c for c in df_clean.columns if c != target_col and c != '日期'
                      and df_clean[c].dtype in ['int64', 'float64']
                      and not any(p in c for p in exclude_patterns)]

print(f"▶ 候选特征数: {len(feature_candidates)}")

# 计算相关性
corr_results = []
for col in feature_candidates:
    valid = df_clean[[target_col, col]].dropna()
    if len(valid) > 10:
        try:
            pearson_r, pearson_p = pearsonr(valid[target_col], valid[col])
            spearman_rho, spearman_p = spearmanr(valid[target_col], valid[col])
            corr_results.append({
                '特征': col,
                '皮尔逊_r': pearson_r,
                '皮尔逊_p': pearson_p,
                '斯皮尔曼_ρ': spearman_rho,
                '斯皮尔曼_p': spearman_p
            })
        except Exception as e:
            print(f"  计算 {col} 时出错: {e}")

if corr_results:
    corr_df = pd.DataFrame(corr_results).sort_values('斯皮尔曼_ρ', key=abs, ascending=False)

    # 选择显著特征
    significant = corr_df[(corr_df['斯皮尔曼_p'] < 0.05) & (abs(corr_df['斯皮尔曼_ρ']) >= 0.3)]
    print(f"✓ 显著相关特征: {len(significant)} 个")

    # 保存相关性结果
    corr_df.to_csv('outputs/correlation_results.csv', index=False, encoding='utf-8-sig')

    # 绘制热力图
    if len(significant) > 1:
        try:
            plt.figure(figsize=(12, 10))
            top_features = significant.head(15)['特征'].tolist()
            if target_col not in top_features:
                top_features = [target_col] + top_features
            corr_matrix = df_clean[top_features].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                        cmap='RdBu_r', center=0, square=True,
                        linewidths=0.5, cbar_kws={"shrink": 0.8})
            plt.title('显著特征相关性热力图', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('outputs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ 热力图已保存: outputs/correlation_heatmap.png")
        except Exception as e:
            print(f"⚠ 绘制热力图时出错: {e}")

    # 处理多重共线性
    selected_features = significant['特征'].tolist()
    if len(selected_features) > 1:
        try:
            feature_corr = df_clean[selected_features].corr().abs()
            to_remove = set()
            for i in range(len(feature_corr.columns)):
                for j in range(i + 1, len(feature_corr.columns)):
                    if feature_corr.iloc[i, j] > 0.8:
                        feat1, feat2 = feature_corr.columns[i], feature_corr.columns[j]
                        corr1 = significant[significant['特征'] == feat1]['斯皮尔曼_ρ'].values[0]
                        corr2 = significant[significant['特征'] == feat2]['斯皮尔曼_ρ'].values[0]
                        if abs(corr1) >= abs(corr2):
                            to_remove.add(feat2)
                        else:
                            to_remove.add(feat1)
            final_features = [f for f in selected_features if f not in to_remove]
        except Exception as e:
            print(f"⚠ 处理多重共线性时出错: {e}")
            final_features = selected_features
    else:
        final_features = selected_features
else:
    print("⚠ 警告: 无法计算相关性，使用所有数值型特征")
    final_features = feature_candidates

print(f"✓ 最终选择特征: {len(final_features)} 个")
if final_features:
    print(f"  特征列表: {final_features[:5]}..." if len(final_features) > 5 else f"  特征列表: {final_features}")

# ==================== 第五部分：数据准备 ====================
print("\n【步骤5/6】数据准备与模型训练...")

if not final_features:
    print("错误：没有可用的特征，无法继续")
    sys.exit(1)

# 准备数据
X = df_clean[final_features].copy()
y = df_clean[target_col].copy()

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
)

print(f"▶ 训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
print(f"▶ 测试集: {X_test.shape[0]} 样本, {X_test.shape[1]} 特征")

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 保存标准化器和特征列表
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(final_features, 'models/selected_features.pkl')
print("✓ 标准化器和特征列表已保存")

# ==================== 第六部分：模型训练与评估 ====================
print("\n【步骤6/6】模型训练与评估...")


def evaluate_model(y_true, y_pred, model_name):
    """计算并打印模型评估指标"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  {model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


# 存储所有模型结果
models_results = {}

# 1. 线性回归模型
print("\n▶ 训练线性回归模型...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_train_pred = lr.predict(X_train_scaled)
y_test_pred = lr.predict(X_test_scaled)
models_results['线性回归'] = {
    'train': evaluate_model(y_train, y_train_pred, "训练集"),
    'test': evaluate_model(y_test, y_test_pred, "测试集"),
    'model': lr
}

# 2. XGBoost模型
print("\n▶ 训练XGBoost模型...")
xgb = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
}
print("  超参数调优中...")
grid = GridSearchCV(xgb, param_grid, cv=min(CV_FOLDS, len(X_train)),
                    scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
grid.fit(X_train_scaled, y_train)
best_xgb = grid.best_estimator_
print(f"  最佳参数: {grid.best_params_}")

y_train_pred = best_xgb.predict(X_train_scaled)
y_test_pred = best_xgb.predict(X_test_scaled)
models_results['XGBoost'] = {
    'train': evaluate_model(y_train, y_train_pred, "训练集"),
    'test': evaluate_model(y_test, y_test_pred, "测试集"),
    'model': best_xgb
}

# 3. 随机森林模型
print("\n▶ 训练随机森林模型...")
rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
}
print("  超参数调优中...")
rf_grid = GridSearchCV(rf, rf_param_grid, cv=min(CV_FOLDS, len(X_train)),
                       scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)
best_rf = rf_grid.best_estimator_
print(f"  最佳参数: {rf_grid.best_params_}")

y_train_pred = best_rf.predict(X_train_scaled)
y_test_pred = best_rf.predict(X_test_scaled)
models_results['随机森林'] = {
    'train': evaluate_model(y_train, y_train_pred, "训练集"),
    'test': evaluate_model(y_test, y_test_pred, "测试集"),
    'model': best_rf
}

# 选择最佳模型
best_model_name = max(models_results, key=lambda x: models_results[x]['test']['R2'])
best_model = models_results[best_model_name]['model']
best_r2 = models_results[best_model_name]['test']['R2']

print(f"\n{'=' * 60}")
print(f"✓ 最佳模型: {best_model_name}")
print(f"✓ 测试集R²: {best_r2:.4f}")
print(f"{'=' * 60}")

# 保存最佳模型
joblib.dump(best_model, 'models/best_model.pkl')
print("✓ 最佳模型已保存: models/best_model.pkl")
# ==================== 添加预测值与实际值对比折线图 ====================
print("\n【生成预测值与实际值对比图】")

# 准备数据用于绘图
# 获取所有模型的测试集预测结果
y_pred_lr = models_results['线性回归']['model'].predict(X_test_scaled)
y_pred_xgb = models_results['XGBoost']['model'].predict(X_test_scaled)
y_pred_rf = models_results['随机森林']['model'].predict(X_test_scaled)

# 创建测试集的索引（按日期排序）
test_indices = y_test.index
test_dates = df_clean.loc[test_indices, '日期'].values

# 按日期排序
sorted_idx = np.argsort(test_dates)
test_dates_sorted = test_dates[sorted_idx]
y_test_sorted = y_test.values[sorted_idx]
y_pred_lr_sorted = y_pred_lr[sorted_idx]
y_pred_xgb_sorted = y_pred_xgb[sorted_idx]
y_pred_rf_sorted = y_pred_rf[sorted_idx]

# 图1: 单个模型预测值与实际值对比（XGBoost，因为是最佳模型）
plt.figure(figsize=(14, 7))
plt.plot(test_dates_sorted, y_test_sorted, 'b-', linewidth=1.5, label='实际值', alpha=0.8)
plt.plot(test_dates_sorted, y_pred_xgb_sorted, 'r-', linewidth=1.5, label='预测值 (XGBoost)', alpha=0.8)
plt.xlabel('日期', fontsize=12, fontweight='bold')
plt.ylabel('投矾量 (kg)', fontsize=12, fontweight='bold')
plt.title(f'XGBoost模型预测值与实际值对比 (测试集)\nR² = {models_results["XGBoost"]["test"]["R2"]:.4f}',
          fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/prediction_vs_actual_xgboost.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 已保存: outputs/prediction_vs_actual_xgboost.png")

# 图2: 三个模型预测值与实际值对比（在同一张图上）
plt.figure(figsize=(16, 8))
plt.plot(test_dates_sorted, y_test_sorted, 'b-', linewidth=2, label='实际值', alpha=0.9, marker='o', markersize=3)
plt.plot(test_dates_sorted, y_pred_lr_sorted, 'g--', linewidth=1.5,
         label=f'线性回归预测 (R²={models_results["线性回归"]["test"]["R2"]:.3f})', alpha=0.7)
plt.plot(test_dates_sorted, y_pred_xgb_sorted, 'r-', linewidth=1.5,
         label=f'XGBoost预测 (R²={models_results["XGBoost"]["test"]["R2"]:.3f})', alpha=0.7)
plt.plot(test_dates_sorted, y_pred_rf_sorted, 'orange', linewidth=1.5,
         label=f'随机森林预测 (R²={models_results["随机森林"]["test"]["R2"]:.3f})', alpha=0.7)
plt.xlabel('日期', fontsize=12, fontweight='bold')
plt.ylabel('投矾量 (kg)', fontsize=12, fontweight='bold')
plt.title('三种模型预测值与实际值对比 (测试集)', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/prediction_vs_actual_all_models.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 已保存: outputs/prediction_vs_actual_all_models.png")

# 图3: 预测误差分析图
plt.figure(figsize=(14, 10))

# 子图1: 预测误差随时间变化
plt.subplot(2, 2, 1)
errors_xgb = y_test_sorted - y_pred_xgb_sorted
plt.plot(test_dates_sorted, errors_xgb, 'r-', linewidth=1, alpha=0.7, label='XGBoost误差')
plt.plot(test_dates_sorted, [0] * len(errors_xgb), 'k--', linewidth=1, alpha=0.5)
plt.fill_between(test_dates_sorted, errors_xgb, 0, alpha=0.3, color='red')
plt.xlabel('日期', fontsize=10)
plt.ylabel('预测误差', fontsize=10)
plt.title('XGBoost预测误差随时间变化', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# 子图2: 预测误差分布直方图
plt.subplot(2, 2, 2)
plt.hist(errors_xgb, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('预测误差', fontsize=10)
plt.ylabel('频数', fontsize=10)
plt.title('预测误差分布', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# 子图3: 绝对误差分布
plt.subplot(2, 2, 3)
abs_errors = np.abs(errors_xgb)
plt.bar(range(len(abs_errors[:50])), abs_errors[:50], color='coral', alpha=0.7)
plt.xlabel('测试样本序号', fontsize=10)
plt.ylabel('绝对误差', fontsize=10)
plt.title('绝对误差分布 (前50个样本)', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# 子图4: 误差统计箱线图
plt.subplot(2, 2, 4)
error_data = [errors_xgb, errors_xgb[errors_xgb >= 0], errors_xgb[errors_xgb < 0]]
bp = plt.boxplot(error_data, labels=['全部误差', '正误差', '负误差'], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)
plt.ylabel('误差值', fontsize=10)
plt.title('误差统计箱线图', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/error_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 已保存: outputs/error_analysis.png")

# 图4: 局部放大图（前50个样本）
plt.figure(figsize=(14, 6))
sample_range = min(50, len(test_dates_sorted))
plt.plot(range(sample_range), y_test_sorted[:sample_range], 'b-', linewidth=2, label='实际值', marker='o', markersize=4)
plt.plot(range(sample_range), y_pred_xgb_sorted[:sample_range], 'r--', linewidth=2, label='XGBoost预测值', marker='s',
         markersize=3)
plt.fill_between(range(sample_range), y_test_sorted[:sample_range], y_pred_xgb_sorted[:sample_range],
                 alpha=0.3, color='gray', label='误差区域')
plt.xlabel('测试样本序号', fontsize=12, fontweight='bold')
plt.ylabel('投矾量 (kg)', fontsize=12, fontweight='bold')
plt.title(f'XGBoost模型预测效果局部放大图 (前{sample_range}个测试样本)', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/prediction_zoom_in.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 已保存: outputs/prediction_zoom_in.png")

# 图5: 散点图加趋势线（增强版）
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models_plot = [
    ('线性回归', y_pred_lr, models_results['线性回归']['test']['R2']),
    ('XGBoost', y_pred_xgb, models_results['XGBoost']['test']['R2']),
    ('随机森林', y_pred_rf, models_results['随机森林']['test']['R2'])
]

for idx, (name, y_pred, r2) in enumerate(models_plot):
    axes[idx].scatter(y_test, y_pred, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

    # 添加趋势线
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    x_range = np.linspace(y_test.min(), y_test.max(), 100)
    axes[idx].plot(x_range, p(x_range), "r-", linewidth=2, label=f'趋势线 (y={z[0]:.2f}x+{z[1]:.2f})')

    # 添加理想线
    axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'g--', linewidth=1.5, label='理想线')

    axes[idx].set_xlabel('实际值', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('预测值', fontsize=12, fontweight='bold')
    axes[idx].set_title(f'{name}\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
    axes[idx].legend(loc='lower right', fontsize=8)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/scatter_with_trendline.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 已保存: outputs/scatter_with_trendline.png")

# 图6: 累计误差分析
plt.figure(figsize=(12, 6))

# 计算累计误差
cumulative_errors = np.cumsum(errors_xgb)
plt.plot(test_dates_sorted, cumulative_errors, 'b-', linewidth=2, marker='o', markersize=3)
plt.fill_between(test_dates_sorted, 0, cumulative_errors, alpha=0.3, color='blue')
plt.xlabel('日期', fontsize=12, fontweight='bold')
plt.ylabel('累计预测误差', fontsize=12, fontweight='bold')
plt.title('XGBoost模型累计预测误差趋势', fontsize=14, fontweight='bold')
plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/cumulative_error.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 已保存: outputs/cumulative_error.png")

# 打印统计信息
print("\n" + "=" * 60)
print("预测效果统计:")
print("=" * 60)
print(f"XGBoost模型预测误差统计:")
print(f"  平均误差: {np.mean(errors_xgb):.4f}")
print(f"  平均绝对误差 (MAE): {np.mean(np.abs(errors_xgb)):.4f}")
print(f"  误差标准差: {np.std(errors_xgb):.4f}")
print(f"  最大正误差: {np.max(errors_xgb):.4f}")
print(f"  最大负误差: {np.min(errors_xgb):.4f}")
print(f"  误差在±10%内的比例: {np.mean(np.abs(errors_xgb / y_test_sorted) < 0.1) * 100:.1f}%")
print(f"  误差在±20%内的比例: {np.mean(np.abs(errors_xgb / y_test_sorted) < 0.2) * 100:.1f}%")
print("=" * 60)
# ==================== 生成报告 ====================
print("\n【生成分析报告】")

# 生成报告
report = f"""# 水厂投矾量预测模型分析报告

## 1. 项目概述
本报告基于水厂历史运行数据，通过系统性的数据分析和机器学习建模，建立了投矾量预测模型。

**分析日期**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 2. 数据概况
- **数据样本数**: {len(df)} 条记录
- **时间范围**: {df['日期'].min().strftime('%Y-%m-%d')} 至 {df['日期'].max().strftime('%Y-%m-%d')}
- **目标变量**: {target_col}
- **原始特征数**: {len(df.columns)} 个
- **特征工程后特征数**: {len(df_clean.columns) - 1} 个
- **最终选择特征数**: {len(final_features)} 个

## 3. 关键影响因素分析

"""

if 'significant' in locals() and len(significant) > 0:
    report += "通过相关性分析，发现以下因素与投矾量显著相关：\n\n"
    report += "| 特征 | 斯皮尔曼相关系数 | p值 | 相关方向 |\n"
    report += "|------|-----------------|-----|----------|\n"
    for i, row in significant.head(10).iterrows():
        direction = "正相关" if row['斯皮尔曼_ρ'] > 0 else "负相关"
        report += f"| {row['特征']} | {row['斯皮尔曼_ρ']:.4f} | {row['斯皮尔曼_p']:.4e} | {direction} |\n"

report += f"""
## 4. 模型性能对比

| 模型 | RMSE (测试集) | MAE (测试集) | R² (测试集) | 训练集R² |
|------|--------------|-------------|------------|----------|
"""

for name, results in models_results.items():
    report += f"| {name} | {results['test']['RMSE']:.4f} | {results['test']['MAE']:.4f} | {results['test']['R2']:.4f} | {results['train']['R2']:.4f} |\n"

report += f"""
## 5. 最佳模型

- **模型名称**: {best_model_name}
- **测试集R²**: {best_r2:.4f}
- **测试集RMSE**: {models_results[best_model_name]['test']['RMSE']:.4f}
- **测试集MAE**: {models_results[best_model_name]['test']['MAE']:.4f}

## 6. 文件输出清单

| 文件 | 说明 |
|------|------|
| models/best_model.pkl | 最佳模型文件 |
| models/scaler.pkl | 标准化器 |
| models/selected_features.pkl | 选择的特征列表 |
| outputs/feature_engineered_data.csv | 特征工程后的数据 |
| outputs/correlation_results.csv | 相关性分析结果 |
| outputs/correlation_heatmap.png | 相关性热力图 |

---

*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

# 保存报告
with open('outputs/model_evaluation_report.md', 'w', encoding='utf-8') as f:
    f.write(report)

print("✓ 分析报告已保存: outputs/model_evaluation_report.md")

# ==================== 完成 ====================
print("\n" + "=" * 80)
print("✅ 分析完成！")
print("=" * 80)
print("\n输出文件:")
print("  📁 models/")
print("     ├── best_model.pkl        # 最佳预测模型")
print("     ├── scaler.pkl            # 数据标准化器")
print("     └── selected_features.pkl # 选择的特征列表")
print("\n  📁 outputs/")
print("     ├── feature_engineered_data.csv  # 特征工程数据")
print("     ├── correlation_results.csv      # 相关性结果")
print("     ├── correlation_heatmap.png      # 相关性热力图")
print("     └── model_evaluation_report.md   # 详细报告")
print("\n" + "=" * 80)