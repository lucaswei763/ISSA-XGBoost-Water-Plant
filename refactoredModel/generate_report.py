#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型拟合能力自动化评估与报告生成脚本 (年度汇总版)
生成: 2021-2025年，每年1个文件夹，内含1张全年拟合图 + 1份年度可信度评价MD报告
"""

import os
import sys
import json
import sqlite3
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# UI和画图设置
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200  # 提高分辨率以适应全年数据展示

REPORT_DIR = 'Model_Evaluation_Report_Annual'


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if len(y_true[non_zero]) == 0:
        return 0.0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


def load_and_preprocess_data():
    """清洗与特征重构逻辑（已修复7天平滑截断和空值遗漏问题）"""
    print("正在加载数据与模型...")
    db_path = 'data/water_data.db'
    if not os.path.exists(db_path):
        sys.exit("错误：数据库文件不存在。请确保在项目根目录运行。")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]

    if 'merged_data' in tables:
        df = pd.read_sql_query("SELECT * FROM merged_data", conn)
        date_col = next((c for c in df.columns if '日期' in c or 'date' in c.lower()), df.columns[0])
        df['日期'] = pd.to_datetime(df[date_col])
    else:
        consumption_table = next((t for t in tables if 'consumption' in t.lower() or '药耗' in t), tables[0])
        quality_table = next((t for t in tables if 'quality' in t.lower() or '水质' in t),
                             tables[1] if len(tables) > 1 else tables[0])
        df_c = pd.read_sql_query(f"SELECT * FROM {consumption_table}", conn)
        df_q = pd.read_sql_query(f"SELECT * FROM {quality_table}", conn)
        dc_c = next((c for c in df_c.columns if '日期' in c or 'date' in c.lower()), df_c.columns[0])
        dc_q = next((c for c in df_q.columns if '日期' in c or 'date' in c.lower()), df_q.columns[0])
        df_c[dc_c] = pd.to_datetime(df_c[dc_c])
        df_q[dc_q] = pd.to_datetime(df_q[dc_q])
        df = pd.merge(df_c, df_q, left_on=dc_c, right_on=dc_q, how='inner')
        df['日期'] = df[dc_c]

    conn.close()
    df = df.sort_values(by='日期').reset_index(drop=True)

    with open('models/metadata.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    target_col = meta['target_col']
    features = meta['features']

    # 物理极限异常清洗
    df_clean = df[df[target_col] <= 5000].reset_index(drop=True)

    num_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c != '日期']
    if num_cols:
        df_clean[num_cols] = df_clean[num_cols].fillna(method='ffill').fillna(df_clean[num_cols].median())

    for lag in [1, 2, 3]:
        if f'{target_col}_lag_{lag}天' in features:
            df_clean[f'{target_col}_lag_{lag}天'] = df_clean[target_col].shift(lag)

    turb_col = next((c for c in df_clean.columns if '浊度' in c or 'turbidity' in c.lower()), None)
    flow_col = next((c for c in df_clean.columns if '流量' in c or 'flow' in c.lower() or 'supply' in c.lower()), None)

    if turb_col:
        df_clean[f'{turb_col}_3天平滑'] = df_clean[turb_col].rolling(3, min_periods=1).mean()
        df_clean[f'{turb_col}_7天平滑'] = df_clean[turb_col].rolling(7, min_periods=1).mean()
    if flow_col:
        df_clean[f'{flow_col}_3天平滑'] = df_clean[flow_col].rolling(3, min_periods=1).mean()
        df_clean[f'{flow_col}_7天平滑'] = df_clean[flow_col].rolling(7, min_periods=1).mean()
    if turb_col and flow_col:
        df_clean['平滑后浊度_流量交互'] = df_clean[f'{turb_col}_3天平滑'] * df_clean[f'{flow_col}_3天平滑']

    new_cols = [c for c in df_clean.columns if '平滑' in c or 'lag' in c or '交互' in c]
    if new_cols:
        df_clean[new_cols] = df_clean[new_cols].fillna(method='ffill').fillna(0)

    df_clean = df_clean.dropna(subset=features + [target_col]).reset_index(drop=True)
    return df_clean, features, target_col


def evaluate_reliability(r2, mape, acc_10):
    """基于硬性指标生成客观的文字评价结论"""
    if r2 >= 0.85 and mape < 10 and acc_10 > 80:
        return "🟢 **高度可信**：模型对该年度的数据特征捕捉极其精准，预测趋势与实际情况高度吻合，日常运作中完全可作为加药指导的核心依据。"
    elif r2 >= 0.70 and mape < 15 and acc_10 > 60:
        return "🟡 **基本可信**：模型能追踪大部分年度趋势，误差在可控范围内。在原水水质极度恶劣或发生设备突变时，可能存在一定的预测滞后，需结合人工经验适度微调。"
    else:
        return "🔴 **可信度不足**：该年度数据波动可能超出了模型当前特征空间的泛化能力，或者存在较多未被记录的外部干扰因素（如设备停机维修等），预测结果仅供宏观参考。"


def generate_report():
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    df_clean, features, target_col = load_and_preprocess_data()

    scaler = joblib.load('models/scaler.pkl')
    model = joblib.load('models/best_model.pkl')

    X = df_clean[features]
    X_scaled = scaler.transform(X)
    df_clean['预测投矾量'] = model.predict(X_scaled)

    target_years = [2021, 2022, 2023, 2024, 2025]
    print(f"开始生成专属绘图与报告，评估年份锁定: {target_years}...")

    for year in target_years:
        year_data = df_clean[df_clean['日期'].dt.year == year].reset_index(drop=True)

        year_dir = os.path.join(REPORT_DIR, f'{year}年')
        os.makedirs(year_dir, exist_ok=True)

        if year_data.empty:
            print(f"⚠️ 警告: 数据库中未找到 {year} 年的数据，跳过该年份。")
            with open(os.path.join(year_dir, f'{year}_无数据.md'), 'w', encoding='utf-8') as f:
                f.write(f"# {year} 年无数据\n当前数据库未包含本年度的有效记录。")
            continue

        # ------------------------------------
        # 1. 计算本年度评价指标
        # ------------------------------------
        y_true = year_data[target_col]
        y_pred = year_data['预测投矾量']

        r2_yr = r2_score(y_true, y_pred)
        mae_yr = mean_absolute_error(y_true, y_pred)
        mape_yr = mean_absolute_percentage_error(y_true, y_pred)

        error_pct = np.abs((y_true - y_pred) / y_true)
        acc_5_pct = (error_pct <= 0.05).mean() * 100
        acc_10_pct = (error_pct <= 0.10).mean() * 100

        reliability_conclusion = evaluate_reliability(r2_yr, mape_yr, acc_10_pct)

        # ------------------------------------
        # 2. 绘制本年度全景趋势图 (加宽长图)
        # ------------------------------------
        plt.figure(figsize=(20, 6))  # 加宽尺寸以适应365天的数据
        plt.plot(year_data['日期'], y_true, label='实际投矾量', color='#1f77b4', linewidth=1.5, alpha=0.9)
        plt.plot(year_data['日期'], y_pred, label='ISSA-XGB预测量', color='#d62728', linestyle='--', linewidth=1.5,
                 alpha=0.9)

        plt.title(f'{year}年度 水厂投矾量预测拟合全景图', fontsize=18, fontweight='bold', pad=15)
        plt.xlabel('日期', fontsize=14)
        plt.ylabel('投矾量 (kg)', fontsize=14)
        plt.legend(fontsize=14, loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)

        # X轴按月显示
        ax = plt.gca()
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_path = os.path.join(year_dir, f'{year}年_预测拟合全景图.png')
        plt.savefig(plot_path)
        plt.close()

        # ------------------------------------
        # 3. 编写年度可信度报告
        # ------------------------------------
        md_path = os.path.join(year_dir, f'{year}年_模型预测可信度评价.md')
        md_content = f"""# 📈 {year}年度 ISSA-XGBoost 模型预测可信度评价报告

本报告基于 {year} 年全年历史加药数据，对比模型回溯预测结果生成。旨在客观评估模型在该自然年度内的泛化能力与实际指导价值。

## 一、 核心量化指标

| 指标维度 | 指标名称 | 本年度实测值 | 行业优秀基准 | 评估释义 |
| :--- | :--- | :--- | :--- | :--- |
| **方差解释力** | 决定系数 (R²) | **{r2_yr:.4f}** | > 0.85 | 反映模型对全年趋势起伏（如夏秋高浊度期）的追踪程度。 |
| **绝对偏差量** | 均方误差 (MAE) | **{mae_yr:.2f} kg** | - | 本年度平均每天预测值与真实值的绝对千克数差异。 |
| **相对偏差率** | 平均绝对百分比误差 (MAPE)| **{mape_yr:.2f}%** | < 10% | 核心业务指标：衡量误差规模相对于实际加药量的比重。 |

## 二、 预测稳定性分布

在加药生产中，容错率极低。以下是 {year} 全年每一天预测精度的分布情况：

* **🔥 高精度控制期 (误差 ≤ 5%)**: 
  全年有 **{acc_5_pct:.2f}%** 的天数，预测误差控制在极严苛的 5% 以内。
* **🛡️ 安全运行期 (误差 ≤ 10%)**: 
  全年有 **{acc_10_pct:.2f}%** 的天数，预测值完全落在业务允许的安全微调区间内。

## 三、 综合可信度定性评价

基于上述量化数据的交叉验证，系统对 {year} 年度模型表现的最终定级为：

> {reliability_conclusion}

**💡 专家审阅建议：**
请结合同级目录下的 `{year}年_预测拟合全景图.png`，重点排查图中出现**巨大尖峰**的日期（通常对应暴雨或管网冲洗导致的浊度剧增）。观察红色虚线（预测值）是否能及时、等幅地跟随蓝色实线（真实值）。这将是最直观反映该模型泛化抗压能力的证据。
"""
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

    print(f"\n✅ 全部评估执行完毕！\n请查看 {REPORT_DIR} 文件夹，获取 2021-2025 年各年度的可信度报告与全景对比图。")


if __name__ == "__main__":
    generate_report()