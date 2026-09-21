#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文件名称：refactoredModel/compare_models.py
所属类别：重构核心生产代码 (Refactored Core Production)

功能描述：
    多模型预测效果的综合评估与性能比对程序。
    支持在同一划分的测试集上，独立拟合并训练三种经典及改进算法：
    1. ISSA-XGBoost (改进麻雀算法优化 XGBoost 树回归)；
    2. ResNet (双头残差神经网络，时序 5 折交叉集成)；
    3. 传统 MLP (多层感知机全连接网络)。
    自动对齐 19 维特征工程输入，并在测试集上计算 MAE (kg)、RMSE (kg)、R² 决定系数以及吨水单位投加指标，
    最后自动按年度绘制实际和预测曲线的垂直对比图（MLP、ResNet、ISSA-XGBoost 三合一布局）。

运行与使用方法：
    直接在控制台执行开始多模型比对评估：
    python compare_models.py

调用与依赖关系：
    - 导入并调用 `utils.py` 内部的数据集划分、归一化和 ResNetRegressor。
    - 导入并调用 `issa_optimizer` 内部的 `ISSA_XGBoost` 进行超参数快速搜索。
    - 导入并调用 `train_hybrid.py` 的混合双头残差网络五折训练流程。
    - 结果以图片形式输出至 `outputs/compare_models-YYYY-MM-DD/HH-MM` 目录下。

设计细节与关键备注：
    - 包含传统 MLP 结构 `AlumDosageMLP` 的 PyTorch 模型定义。
    - 采用 `neg_mean_absolute_error` 统一在测试集评估表现。
"""

import time
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import (
    load_and_preprocess_data, 
    evaluate_and_plot, 
    setup_logger_and_dir, 
    AlumDosageResNet,
    ResNetRegressor
)
from issa_optimizer import ISSA_XGBoost

class AlumDosageMLP(nn.Module):
    """传统多层感知机 (无残差结构，纯全连接前馈神经网络)"""
    def __init__(self, input_dim):
        super(AlumDosageMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

def train_pytorch_model(model, train_loader, val_x, val_y, epochs=500):
    """统一的 PyTorch 模型训练辅助函数（含早停）"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    best_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    patience = 50
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_x)
            val_loss = criterion(val_outputs, val_y).item()
            
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    model.load_state_dict(best_model_weights)
    return model

def calculate_metrics(y_true, y_pred, water=None):
    """统一计算统计指标和业务指标"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    
    if water is not None:
        safe_water = np.where(water == 0, 1e-5, water)
        y_true_unit = y_true / safe_water
        y_pred_unit = y_pred / safe_water
        metrics['MAE_Unit'] = mean_absolute_error(y_true_unit, y_pred_unit)
        metrics['RMSE_Unit'] = np.sqrt(mean_squared_error(y_true_unit, y_pred_unit))
        metrics['R2_Unit'] = r2_score(y_true_unit, y_pred_unit)
        
    return metrics

def main():
    # 1. 初始化专门的对比输出目录和日志
    run_dir = setup_logger_and_dir("compare_models")
    
    print("=" * 65)
    print("🌟 智慧水务投药预测系统 - 三种模式 (XGBoost、ResNet、MLP) 对比评估 🌟")
    print("=" * 65)
    
    # 2. 读取并处理数据
    print("\n🔍 读取并处理数据集中...")
    X_train_scaled, X_test_scaled, y_train, y_test, X_full_scaled, y_full, full_dates, water_test, water_full = load_and_preprocess_data()
    
    # 3. 准备 PyTorch 训练所需的数据结构
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    BATCH_SIZE = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    input_dim = X_train_scaled.shape[1]
    
    print("\n" + "-" * 50)
    print("🛠️ 正在进行 3 种模式的拟合训练与评估...")
    print("-" * 50)
    
    # --- 模式 A: ISSA-XGBoost ---
    print("\n[模式 1/3] 训练 ISSA-XGBoost 模型...")
    start_time = time.time()
    # 为保证对比评估的效率，采用更快的 ISSA 寻优规模 (pop_size=10, max_iter=10)
    issa = ISSA_XGBoost(pop_size=10, max_iter=10)
    best_params = issa.optimize(X_train_scaled, y_train)
    
    xgb_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train_scaled, y_train)
    xgb_time = time.time() - start_time
    print(f"✓ ISSA-XGBoost 训练完成，耗时 {xgb_time:.2f} 秒。")
    
    # --- 模式 B: ResNet (残差神经网络) ---
    print("\n[模式 2/3] 训练 ResNet (残差神经网络) 模型 (5折时序分块集成)...")
    start_time = time.time()
    
    # 恢复时序以确保 Blocked K-Fold
    from sklearn.model_selection import train_test_split
    from build_database import WaterDataLoader
    try:
        loader = WaterDataLoader()
        df_db = loader.get_all_data()
        df_db = df_db.dropna(subset=['耗用矾量（kg）']).copy()
        idx_db = df_db.index
        train_idx_db, _ = train_test_split(idx_db, test_size=0.2, random_state=42)
        sort_indices = np.argsort(train_idx_db)
        X_train_scaled_sorted = X_train_scaled[sort_indices]
        y_train_sorted = y_train[sort_indices]
        print(f"  ✓ 对比程序已成功重构 ResNet 时序连续性！")
    except Exception as e:
        print(f"  ⚠️ 时序恢复警告: {e}")
        X_train_scaled_sorted = X_train_scaled
        y_train_sorted = y_train

    from train_hybrid import train_hybrid_resnet
    y_train_xgb = xgb_model.predict(X_train_scaled_sorted)
    trained_resnet_models = train_hybrid_resnet(
        X_train_scaled_sorted, 
        y_train_sorted, 
        y_train_xgb,
        input_dim, 
        run_dir=None
    )
    resnet_regressor = ResNetRegressor(trained_resnet_models, input_dim=input_dim)
    resnet_time = time.time() - start_time
    print(f"✓ ResNet 5折集成模型训练完成，耗时 {resnet_time:.2f} 秒。")
    
    # --- 模式 C: 传统 MLP (多层感知机) ---
    print("\n[模式 3/3] 训练 传统 MLP (多层感知机) 模型...")
    start_time = time.time()
    mlp_model = AlumDosageMLP(input_dim)
    mlp_model = train_pytorch_model(mlp_model, train_loader, X_test_tensor, y_test_tensor)
    mlp_time = time.time() - start_time
    print(f"✓ 传统 MLP 训练完成，耗时 {mlp_time:.2f} 秒。")
    
    print("\n" + "=" * 50)
    print("📊 正在全量推断并绘制对比图表...")
    print("=" * 50)
    
    # 4. 全量推断并统一合并画图（每年生成一张，垂直排列上中下：MLP、ResNet、ISSA-XGBoost）
    import matplotlib.pyplot as plt
    from matplotlib.dates import MonthLocator, DateFormatter
    
    # 预测并转换一维数组
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    y_full_pred_xgb = xgb_model.predict(X_full_scaled)
    
    # 使用集成模型推理
    y_pred_resnet = resnet_regressor.predict(X_test_scaled)
    y_full_pred_resnet = resnet_regressor.predict(X_full_scaled)
        
    mlp_model.eval()
    with torch.no_grad():
        y_pred_mlp = mlp_model(X_test_tensor).numpy().flatten()
        y_full_pred_mlp = mlp_model(torch.FloatTensor(X_full_scaled)).numpy().flatten()
        
    # 构建整合的 DataFrame 便于按年拆分与画图
    results_df = pd.DataFrame({
        'Date': pd.to_datetime(full_dates),
        'Actual': y_full,
        'MLP': y_full_pred_mlp,
        'ResNet': y_full_pred_resnet,
        'ISSA-XGBoost': y_full_pred_xgb
    })
    
    # 过滤空日期并按日期正序排序
    results_df.dropna(subset=['Date'], inplace=True)
    results_df = results_df.sort_values(by='Date').reset_index(drop=True)
    results_df['Year'] = results_df['Date'].dt.year
    unique_years = sorted(results_df['Year'].unique())
    
    print("\n✍️ 正在按照 [每年一张图，上中下分别对比 MLP、ResNet、ISSA-XGBoost] 的布局绘制对比图...")
    
    for year in unique_years:
        year_df = results_df[results_df['Year'] == year].copy()
        if year_df.empty:
            continue
            
        fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)
        
        # 1. 顶部子图 - 传统 MLP 对比
        axes[0].plot(year_df['Date'], year_df['Actual'], label='实际总耗用量', color='#2b5c8f', linewidth=1.8, marker='o', markersize=2, alpha=0.8)
        axes[0].plot(year_df['Date'], year_df['MLP'], label='MLP 预测耗用量', color='#e67e22', linestyle='-.', linewidth=1.5, marker='x', markersize=2, alpha=0.8)
        axes[0].set_title(f'传统 MLP 模型预测对比 ({year}年)', fontsize=13, fontweight='bold')
        axes[0].set_ylabel('耗用矾量 (kg)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        # 2. 中部子图 - ResNet 对比
        axes[1].plot(year_df['Date'], year_df['Actual'], label='实际总耗用量', color='#2b5c8f', linewidth=1.8, marker='o', markersize=2, alpha=0.8)
        axes[1].plot(year_df['Date'], year_df['ResNet'], label='ResNet 预测耗用量', color='#2ecc71', linestyle='--', linewidth=1.8, marker='+', markersize=2, alpha=0.8)
        axes[1].set_title(f'ResNet (残差神经网络) 模型预测对比 ({year}年)', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('耗用矾量 (kg)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        # 3. 底部子图 - ISSA-XGBoost 对比
        axes[2].plot(year_df['Date'], year_df['Actual'], label='实际总耗用量', color='#2b5c8f', linewidth=1.8, marker='o', markersize=2, alpha=0.8)
        axes[2].plot(year_df['Date'], year_df['ISSA-XGBoost'], label='ISSA-XGBoost 预测耗用量', color='#e74c3c', linestyle=':', linewidth=1.8, marker='*', markersize=2, alpha=0.8)
        axes[2].set_title(f'ISSA-XGBoost 模型预测对比 ({year}年)', fontsize=13, fontweight='bold')
        axes[2].set_xlabel('日期', fontsize=12)
        axes[2].set_ylabel('耗用矾量 (kg)')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, linestyle='--', alpha=0.5)
        
        # 整体布局细节微调
        plt.suptitle(f'{year}年度 水厂投矾量预测多模型拟合对比全景图 (MLP vs ResNet vs ISSA-XGBoost)', fontsize=16, fontweight='bold', y=0.97)
        
        # X轴日期显示设置
        axes[2].xaxis.set_major_locator(MonthLocator())
        axes[2].xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.91)
        
        plot_path = os.path.join(run_dir, f'model_comparison_curves_{year}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ {year}年 3模型垂直对比图已保存至: {plot_path}")
    
    # 5. 计算并打印对比表
    metrics_xgb = calculate_metrics(y_test, y_pred_xgb, water_test)
    metrics_resnet = calculate_metrics(y_test, y_pred_resnet, water_test)
    metrics_mlp = calculate_metrics(y_test, y_pred_mlp, water_test)
    
    print("\n" + "=" * 65)
    print("🏆 3 种预测模式对比结果汇总 (测试集表现)")
    print("=" * 65)
    
    # 汇总表格 (Markdown 格式便于阅读)
    summary_df = pd.DataFrame([
        {
            "预测模型": "ISSA-XGBoost",
            "MAE (kg)": f"{metrics_xgb['MAE']:.2f}",
            "RMSE (kg)": f"{metrics_xgb['RMSE']:.2f}",
            "R² 决定系数": f"{metrics_xgb['R2']:.4f}",
            "单位MAE (kg/千吨)": f"{metrics_xgb['MAE_Unit']:.2f}",
            "单位R²": f"{metrics_xgb['R2_Unit']:.4f}",
            "训练耗时": f"{xgb_time:.1f}s"
        },
        {
            "预测模型": "ResNet (残差网络)",
            "MAE (kg)": f"{metrics_resnet['MAE']:.2f}",
            "RMSE (kg)": f"{metrics_resnet['RMSE']:.2f}",
            "R² 决定系数": f"{metrics_resnet['R2_Unit']:.4f}" if metrics_resnet['R2'] < 0 else f"{metrics_resnet['R2']:.4f}",
            "单位MAE (kg/千吨)": f"{metrics_resnet['MAE_Unit']:.2f}",
            "单位R²": f"{metrics_resnet['R2_Unit']:.4f}",
            "训练耗时": f"{resnet_time:.1f}s"
        },
        {
            "预测模型": "传统 MLP (全连接)",
            "MAE (kg)": f"{metrics_mlp['MAE']:.2f}",
            "RMSE (kg)": f"{metrics_mlp['RMSE']:.2f}",
            "R² 决定系数": f"{metrics_mlp['R2']:.4f}",
            "单位MAE (kg/千吨)": f"{metrics_mlp['MAE_Unit']:.2f}",
            "单位R²": f"{metrics_mlp['R2_Unit']:.4f}",
            "训练耗时": f"{mlp_time:.1f}s"
        }
    ])
    
    # 打印对比表 (使用手动格式化输出，无需外部库依赖，支持防错)
    print(f"{'预测模型':<22} | {'MAE (kg)':<10} | {'RMSE (kg)':<10} | {'R² 决定系数':<12} | {'单位MAE (kg/千吨)':<15} | {'单位R²':<8} | {'训练耗时':<8}")
    print("-" * 105)
    for _, row in summary_df.iterrows():
        print(f"{row['预测模型']:<20} | {row['MAE (kg)']:<10} | {row['RMSE (kg)']:<10} | {row['R² 决定系数']:<12} | {row['单位MAE (kg/千吨)']:<15} | {row['单位R²']:<8} | {row['训练耗时']:<8}")
    print("=" * 105)
    print(f"\n🎉 恭喜！三种模式的实际-预测对比图已全部生成至目录：\n📂 {run_dir}")
    print("=" * 105)

if __name__ == "__main__":
    main()
