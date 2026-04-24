#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
水厂投矾量数据分析与建模 - 模块化优化版
功能：数据加载、清洗、特征工程、多模型对比评估及自动化报告生成
"""

import os
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
from typing import Tuple, List, Dict, Any, Optional

# 环境配置
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置中心 ====================
CONFIG = {
    'db_candidates': [
        'water_data.db',
        '../refactoredModel/data/water_data.db',
        'refactoredModel/data/water_data.db'
    ],
    'output_dir': 'outputs',
    'model_dir': 'models',
    'target_candidates': ['alum_kg', '矾耗(kg)', '矾', 'alum', '矾耗', '矾用量'],
    'date_candidates': ['日期', 'date', '时间', 'time'],
    'test_size': 0.2,
    'random_state': 42
}


def setup_directories():
    """创建必要的输出目录"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    os.makedirs(CONFIG['model_dir'], exist_ok=True)


def find_database() -> str:
    """寻找并返回有效的数据库路径"""
    for candidate in CONFIG['db_candidates']:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("找不到数据库文件 water_data.db，请检查配置路径。")


def load_data(db_path: str) -> pd.DataFrame:
    """连接数据库并加载合并后的数据"""
    print(f"\n[1/6] 正在从 {db_path} 加载数据...")
    conn = sqlite3.connect(db_path)
    try:
        # 优先从 merged_data 加载
        df = pd.read_sql_query("SELECT * FROM merged_data", conn)
        print(f"✓ 成功加载 merged_data 表，记录数: {len(df)}")
    except Exception:
        print("! merged_data 表不存在，尝试手动合并 consumption 和 water_quality...")
        cons_df = pd.read_sql_query("SELECT * FROM consumption", conn)
        qual_df = pd.read_sql_query("SELECT * FROM water_quality", conn)
        
        # 自动识别日期列
        d1 = next((c for c in cons_df.columns if any(p in c.lower() for p in CONFIG['date_candidates'])), cons_df.columns[0])
        d2 = next((c for c in qual_df.columns if any(p in c.lower() for p in CONFIG['date_candidates'])), qual_df.columns[0])
        
        cons_df[d1] = pd.to_datetime(cons_df[d1])
        qual_df[d2] = pd.to_datetime(qual_df[d2])
        
        df = pd.merge(cons_df, qual_df, left_on=d1, right_on=d2, how='inner')
        df.rename(columns={d1: '日期'}, inplace=True)
        if d2 != d1:
            df.drop(columns=[d2], inplace=True)
    finally:
        conn.close()
    
    if '日期' in df.columns:
        df['日期'] = pd.to_datetime(df['日期'])
    return df


def identify_target(df: pd.DataFrame) -> str:
    """自动识别目标变量列名"""
    for candidate in CONFIG['target_candidates']:
        for col in df.columns:
            if candidate.lower() in col.lower():
                print(f"✓ 识别到目标变量: {col}")
                return col
    
    # 备选：包含“耗”字的第一个数值列
    for col in df.columns:
        if '耗' in col and pd.api.types.is_numeric_dtype(df[col]):
            return col
            
    raise ValueError("无法识别目标变量（投矾量），请手动指定。")


def clean_and_engineer(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """数据清洗与特征工程"""
    print("\n[2/6] 数据清洗与特征工程...")
    df_new = df.copy()
    
    # 1. 处理缺失值 (前向/后向填充)
    num_cols = df_new.select_dtypes(include=[np.number]).columns
    df_new[num_cols] = df_new[num_cols].ffill().bfill().fillna(0)
    
    # 2. 处理异常值 (IQR 截断)
    for col in num_cols:
        if col != '日期' and col != target_col:
            Q1, Q3 = df_new[col].quantile(0.25), df_new[col].quantile(0.75)
            IQR = Q3 - Q1
            df_new[col] = df_new[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # 3. 时间特征
    if '日期' in df_new.columns:
        df_new['月'] = df_new['日期'].dt.month
        df_new['星期'] = df_new['日期'].dt.dayofweek
        df_new['季节'] = df_new['月'].apply(lambda m: (m%12 + 3)//3) # 1:冬, 2:春, 3:夏, 4:秋
        
    # 4. 滞后特征 (Lag)
    for lag in [1, 2, 3, 7]:
        df_new[f'{target_col}_lag_{lag}'] = df_new[target_col].shift(lag)
    
    # 5. 滚动均值 (Rolling)
    for col in [c for c in num_cols if 'turbidity' in c.lower() or c == target_col][:3]:
        df_new[f'{col}_roll7_mean'] = df_new[col].rolling(7, min_periods=1).mean()
        
    df_new = df_new.ffill().bfill().fillna(0)
    df_new.to_csv(os.path.join(CONFIG['output_dir'], 'feature_engineered_data.csv'), index=False, encoding='utf-8-sig')
    return df_new


def correlation_analysis(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """相关性分析并生成热力图"""
    print("\n[3/6] 相关性分析...")
    exclude = ['日期', '月', '星期', '季节']
    candidates = [c for c in df.columns if c != target_col and c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    
    results = []
    for col in candidates:
        valid = df[[target_col, col]].dropna()
        if len(valid) > 10:
            rho, p = spearmanr(valid[target_col], valid[col])
            results.append({'特征': col, '斯皮尔曼_ρ': rho, 'p值': p})
            
    corr_df = pd.DataFrame(results).sort_values('斯皮尔曼_ρ', key=abs, ascending=False)
    corr_df.to_csv(os.path.join(CONFIG['output_dir'], 'correlation_results.csv'), index=False, encoding='utf-8-sig')
    
    # 绘制显著特征热力图
    significant = corr_df[corr_df['p值'] < 0.05].head(10)
    if len(significant) > 1:
        plt.figure(figsize=(10, 8))
        top_feats = significant['特征'].tolist() + [target_col]
        sns.heatmap(df[top_feats].corr(), annot=True, cmap='RdBu_r', fmt=".2f")
        plt.title("显著特征相关性热力图")
        plt.savefig(os.path.join(CONFIG['output_dir'], 'correlation_heatmap.png'))
        plt.close()
        
    return corr_df


def train_models(df: pd.DataFrame, target_col: str, corr_df: pd.DataFrame) -> Dict[str, Any]:
    """模型训练、调优与评估"""
    print("\n[4/6] 模型训练与超参数调优...")
    
    # 特征选择：选择显著且不高度共线的特征
    significant_cols = corr_df[corr_df['p值'] < 0.05]['特征'].tolist()
    final_features = []
    if significant_cols:
        # 简单去共线逻辑
        corr_matrix = df[significant_cols].corr().abs()
        to_drop = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.85:
                    to_drop.add(corr_matrix.columns[j])
        final_features = [c for c in significant_cols if c not in to_drop]
    
    if not final_features:
        final_features = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])][:15]

    X, y = df[final_features], df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'])
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 保存预处理器
    joblib.dump(scaler, os.path.join(CONFIG['model_dir'], 'scaler.pkl'))
    joblib.dump(final_features, os.path.join(CONFIG['model_dir'], 'selected_features.pkl'))
    
    models = {
        '线性回归': LinearRegression(),
        'XGBoost': GridSearchCV(XGBRegressor(random_state=42), 
                               {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.1]}, 
                               cv=3, scoring='r2').fit(X_train_scaled, y_train).best_estimator_,
        '随机森林': GridSearchCV(RandomForestRegressor(random_state=42),
                               {'n_estimators': [100], 'max_depth': [5, 10]},
                               cv=3, scoring='r2').fit(X_train_scaled, y_train).best_estimator_
    }
    
    results = {}
    for name, model in models.items():
        if name == '线性回归': model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results[name] = {
            'model': model,
            'metrics': {
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R2': r2_score(y_test, y_pred)
            },
            'predictions': y_pred
        }
        print(f"  {name}: R²={results[name]['metrics']['R2']:.4f}")
        
    return {'results': results, 'X_test': X_test, 'y_test': y_test, 'features': final_features}


def visualize_and_report(train_results: Dict[str, Any], target_col: str):
    """生成图表和最终报告"""
    print("\n[5/6] 生成可视化图表与报告...")
    results = train_results['results']
    y_test = train_results['y_test']
    
    # 1. 性能对比图
    plt.figure(figsize=(12, 4))
    names = list(results.keys())
    r2_scores = [r['metrics']['R2'] for r in results.values()]
    plt.bar(names, r2_scores, color=['#3498db', '#e67e22', '#2ecc71'])
    plt.title("各模型 R² 评分对比")
    plt.ylabel("R² Score")
    plt.savefig(os.path.join(CONFIG['output_dir'], 'model_performance.png'))
    plt.close()
    
    # 2. 预测散点图 (以最佳模型为例)
    best_name = max(results, key=lambda k: results[k]['metrics']['R2'])
    best_pred = results[best_name]['predictions']
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("实际值")
    plt.ylabel("预测值")
    plt.title(f"最佳模型 ({best_name}) 预测效果")
    plt.savefig(os.path.join(CONFIG['output_dir'], 'prediction_scatter.png'))
    plt.close()
    
    # 3. 保存最佳模型
    joblib.dump(results[best_name]['model'], os.path.join(CONFIG['model_dir'], 'best_model.pkl'))
    
    # 4. 生成 Markdown 报告
    report = f"""# 水厂投矾量预测模型评估报告

## 1. 任务摘要
- **目标变量**: {target_col}
- **特征总数**: {len(train_results['features'])}
- **最佳模型**: {best_name} (R² = {results[best_name]['metrics']['R2']:.4f})

## 2. 模型性能对比
| 模型 | RMSE | MAE | R² |
|------|------|-----|-----|
"""
    for name, data in results.items():
        m = data['metrics']
        report += f"| {name} | {m['RMSE']:.4f} | {m['MAE']:.4f} | {m['R2']:.4f} |\n"
        
    report += f"\n## 3. 关键特征\n{', '.join(train_results['features'][:10])}\n"
    
    with open(os.path.join(CONFIG['output_dir'], 'model_evaluation_report.md'), 'w', encoding='utf-8') as f:
        f.write(report)


def main():
    try:
        setup_directories()
        db_path = find_database()
        df = load_data(db_path)
        target = identify_target(df)
        df_processed = clean_and_engineer(df, target)
        corr_df = correlation_analysis(df_processed, target)
        train_res = train_models(df_processed, target, corr_df)
        visualize_and_report(train_res, target)
        
        print("\n" + "="*50)
        print("✅ 所有流程已顺利完成！")
        print(f"📊 结果已保存至 {CONFIG['output_dir']}/ 目录")
        print(f"🤖 模型已保存至 {CONFIG['model_dir']}/ 目录")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()