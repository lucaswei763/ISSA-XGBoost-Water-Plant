#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文件名称：refactoredModel/generate_report.py
所属类别：重构核心生产代码 (Refactored Core Production)

功能描述：
    水厂投药模型拟合能力年度评估报告自动化生成脚本。
    该程序自动读取水厂历史数据库并根据当前激活的预测模型 (ResNet 或 XGBoost) 执行 2021-2025 各年份的回溯推断，
    针对每个自然年度执行：
    1. 计算量化评价指标：包括方差解释力 R² 决定系数、日平均绝对偏差 MAE、相对偏差百分比 MAPE 等；
    2. 统计高精度区间的概率分布 (误差在 <=5% 和 <=10% 以内的天数占比)；
    3. 自动生成各年度 365 天日加矾量的实际值-预测值拟合对比全景图 (PNG 长图)；
    4. 自动生成年度模型可信度分析 Markdown 报告文档。

运行与使用方法：
    直接在控制台执行：
    python generate_report.py
    
    结果文件将被保存在新生成的 `Model_Evaluation_Report_Annual` 目录下，按年份文件夹（如 `2021年`）分类。

调用与依赖关系：
    - 运行依赖于 `models/metadata.json` 所指定的输入特征名与目标字段。
    - 依赖 `joblib` 读取当前已打包的最佳模型和归一化参数。
    - 输出物用于水厂工艺审核与模型拟合水平的专家组报告汇报。
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


def load_and_preprocess_data(model_dir='models'):
    """使用重构后的通用数据加载与预处理机制（对接 19 维特征）"""
    print("正在加载数据与模型...")
    from utils import load_and_preprocess_data as utils_load_data
    
    # 自动调用 utils 的方法加载清洗并对齐好的数据
    _, _, _, _, X_full_scaled, y_full, full_dates, _, _ = utils_load_data(model_dir)
    
    df_clean = pd.DataFrame({
        '日期': pd.to_datetime(full_dates),
        '实际投矾量': y_full
    })
    
    return df_clean, X_full_scaled, '实际投矾量'


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

    model_dir = 'models/resnet' if os.path.exists('models/resnet/best_model.pkl') else ('models/xgboost' if os.path.exists('models/xgboost/best_model.pkl') else 'models')
    df_clean, X_scaled, target_col = load_and_preprocess_data(model_dir=model_dir)

    model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
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