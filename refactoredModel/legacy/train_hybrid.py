"""
文件名称：refactoredModel/train_hybrid.py
所属类别：重构核心生产代码 (Refactored Core Production)

功能描述：
    本项目最主要的深度集成模型训练管线脚本。
    执行“双头残差网络 (SupervisedAlumDosageResNet)”的渐进式混合监督训练，步骤如下：
    1. 统一加载水厂数据库数据，并对训练集执行时序恢复，以实施时间分块五折交叉验证 (Blocked K-Fold)；
    2. 训练一个 ISSA-XGBoost 教师模型，生成训练集上的基准预测拟合标签；
    3. 运行渐进式两阶段训练：
       - 阶段一 (Phase 1)：冻结残差优化层，仅让初始特征提取层与辅助输出头训练逼近 XGBoost 基线规律；
       - 阶段二 (Phase 2)：解冻全网，以真实加药量为主损失、XGBoost预测值为辅助损失，联合微调修正并完全训练残差微调网络；
    4. 将五折训练出来的模型打包为 `ResNetRegressor` 集成类，输出评估指标、绘制年度表现对比图并保存至 models/resnet/best_model.pkl。

运行与使用方法：
    直接在终端中启动进行模型重训：
    python train_hybrid.py

调用与依赖关系：
    - 导入并调用 `utils` 中的数据处理、绘图、残差类架构。
    - 导入并调用 `issa_optimizer` 中的 `ISSA_XGBoost` 类来优化 XGBoost 超参数。
    - 运行后会输出最优权重到 `models/resnet/` 目录，供 `ui_app.py`、`app.py`、`cli_app.py` 进行在线部署推理。

设计细节与关键备注：
    - 两阶段优化器均使用 AdamW，并配合 CosineAnnealingLR (余弦退火学习率衰减) 与 Early Stopping (提前终止) 策略防范过拟合。
    - 日志和过程损失曲线图会保存在 `outputs/hybrid_train-YYYY-MM-DD/HH-MM` 临时目录下以供开发比对。
"""
import time
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from utils import (
    load_and_preprocess_data, 
    evaluate_and_plot, 
    setup_logger_and_dir, 
    SupervisedAlumDosageResNet, 
    ResNetRegressor
)
from issa_optimizer import ISSA_XGBoost

# 设定中文字体
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Songti SC', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def train_hybrid_resnet(train_x_all, train_y_all, train_y_xgb, input_dim, run_dir=None):
    """时间分块 5 折交叉验证联合训练：Phase 1 逼近 XGBoost -> Phase 2 联合优化真实值+残差修正"""
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=5, shuffle=False)
    trained_models = []
    
    criterion = nn.MSELoss()
    
    # 5折交叉验证
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_x_all)):
        print(f"\n==================== 训练第 {fold + 1} 折 (Fold {fold + 1}/5) ====================")
        
        # 1. 提取当前折的训练集和验证集
        X_tr, y_tr_actual, y_tr_xgb = train_x_all[train_idx], train_y_all[train_idx], train_y_xgb[train_idx]
        X_va, y_va_actual, y_va_xgb = train_x_all[val_idx], train_y_all[val_idx], train_y_xgb[val_idx]
        
        # 2. 转换为张量
        X_tr_tensor = torch.FloatTensor(X_tr)
        y_tr_actual_tensor = torch.FloatTensor(y_tr_actual).view(-1, 1)
        y_tr_xgb_tensor = torch.FloatTensor(y_tr_xgb).view(-1, 1)
        
        X_va_tensor = torch.FloatTensor(X_va)
        y_va_actual_tensor = torch.FloatTensor(y_va_actual).view(-1, 1)
        y_va_xgb_tensor = torch.FloatTensor(y_va_xgb).view(-1, 1)
        
        # 3. 实例化当折的双头残差模型
        model = SupervisedAlumDosageResNet(input_dim)
        
        # ------------------ 【Phase 1: 拟合 XGBoost 决策边界】 ------------------
        print("  ⚡ 阶段一：冻结残差层，训练前几层逼近 XGBoost 基线规律...")
        
        # 仅让前几层及辅助头更新梯度，冻结残差优化层及最终输出头
        for name, param in model.named_parameters():
            if "res_blocks" in name or "out_block" in name:
                param.requires_grad = False
                
        # 阶段一优化器
        optimizer_p1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, weight_decay=1e-4)
        
        # 封装阶段一 DataLoader
        dataset_p1 = TensorDataset(X_tr_tensor, y_tr_xgb_tensor)
        loader_p1 = DataLoader(dataset_p1, batch_size=64, shuffle=True)
        
        p1_epochs = 300
        best_p1_loss = float('inf')
        best_p1_weights = copy.deepcopy(model.state_dict())
        p1_patience = 40
        p1_patience_counter = 0
        
        for epoch in range(p1_epochs):
            model.train()
            for batch_x, batch_y_xgb in loader_p1:
                # 前向传播，得到辅助输出
                _, y_base = model(batch_x)
                loss = criterion(y_base, batch_y_xgb)
                
                optimizer_p1.zero_grad()
                loss.backward()
                optimizer_p1.step()
                
            # 在验证折评估
            model.eval()
            with torch.no_grad():
                _, val_y_base = model(X_va_tensor)
                val_loss_p1 = criterion(val_y_base, y_va_xgb_tensor).item()
                
            if val_loss_p1 < best_p1_loss:
                best_p1_loss = val_loss_p1
                best_p1_weights = copy.deepcopy(model.state_dict())
                p1_patience_counter = 0
            else:
                p1_patience_counter += 1
                
            if p1_patience_counter >= p1_patience:
                break
                
        # 加载阶段一最优权重
        model.load_state_dict(best_p1_weights)
        print(f"  ✓ 阶段一对齐完成！最优验证集对齐 MSE: {best_p1_loss:.2f}")
        
        # ------------------ 【Phase 2: 联合微调纠错】 ------------------
        print("  ⚡ 阶段二：解冻所有网络层，通过残差对投矾量进行联合纠错优化...")
        
        # 解冻所有层，使其在微调时自适应微调前部基准并完全训练后部残差
        for param in model.parameters():
            param.requires_grad = True
            
        # 阶段二优化器与余弦退火
        p2_epochs = 2000
        optimizer_p2 = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4) # 稍微降低学习率，实现微调
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=p2_epochs, eta_min=1e-5)
        
        # 封装多任务 DataLoader (包含输入特征、真实加药量、XGBoost辅助基线)
        dataset_p2 = TensorDataset(X_tr_tensor, y_tr_actual_tensor, y_tr_xgb_tensor)
        loader_p2 = DataLoader(dataset_p2, batch_size=64, shuffle=True)
        
        best_p2_loss = float('inf')
        best_p2_weights = copy.deepcopy(model.state_dict())
        p2_patience = 150
        p2_patience_counter = 0
        
        fold_train_losses = []
        fold_val_losses = []
        
        start_time = time.time()
        for epoch in range(p2_epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch_x, batch_y_actual, batch_y_xgb in loader_p2:
                # 前向传播，同时得到最终输出和辅助基准输出
                y_final, y_base = model(batch_x)
                
                # 联合损失函数：真实值主损失 + 0.2 * XGBoost 辅助损失
                loss_main = criterion(y_final, batch_y_actual)
                loss_aux = criterion(y_base, batch_y_xgb)
                loss = loss_main + 0.2 * loss_aux
                
                optimizer_p2.zero_grad()
                loss.backward()
                optimizer_p2.step()
                
                epoch_loss += loss_main.item() * batch_x.size(0)
                
            epoch_loss /= len(loader_p2.dataset)
            fold_train_losses.append(epoch_loss)
            
            scheduler.step()
            
            # 在验证折评估最终输出的 MSE 表现
            model.eval()
            with torch.no_grad():
                val_y_final, _ = model(X_va_tensor)
                val_loss_p2 = criterion(val_y_final, y_va_actual_tensor).item()
                fold_val_losses.append(val_loss_p2)
                
            if val_loss_p2 < best_p2_loss:
                best_p2_loss = val_loss_p2
                best_p2_weights = copy.deepcopy(model.state_dict())
                p2_patience_counter = 0
            else:
                p2_patience_counter += 1
                
            if (epoch + 1) % 100 == 0:
                current_lr = optimizer_p2.param_groups[0]['lr']
                print(f"    [Epoch {epoch+1}/{p2_epochs}] Train MSE: {epoch_loss:.2f} | Val MSE: {val_loss_p2:.2f} | LR: {current_lr:.6f}")
                
            if p2_patience_counter >= p2_patience:
                print(f"    [Early Stopping] 第 {epoch+1} 轮提前停止。")
                break
                
        print(f"  ⏱️ Fold {fold + 1} 训练总耗时: {(time.time() - start_time):.2f} 秒 | 最优验证集真实 MSE: {best_p2_loss:.2f}")
        
        # 保存当折最优模型权重
        model.load_state_dict(best_p2_weights)
        trained_models.append(model)
        
        # 绘制该折的损失曲线
        if run_dir:
            plt.figure(figsize=(10, 5))
            plt.plot(fold_train_losses, label='Train Loss')
            plt.plot(fold_val_losses, label='Validation Loss')
            plt.title(f'Fold {fold + 1} Joint Refinement Loss')
            plt.xlabel('Epochs')
            plt.ylabel('MSE Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(run_dir, f'hybrid_resnet_loss_fold_{fold + 1}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
    return trained_models

def main():
    run_dir = setup_logger_and_dir("hybrid_train")
    
    print("=" * 70)
    print("🌟 PyTorch Dual-Head ResNet (XGBoost辅助监督残差修正) 混合集成模型 🌟")
    print("=" * 70)
    
    # 1. 统一加载数据
    print("\n🔍 读取并处理数据集中...")
    try:
        X_train_scaled, X_test_scaled, y_train, y_test, X_full_scaled, y_full, full_dates, water_test, water_full = load_and_preprocess_data(model_dir='models/resnet')
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 2. 重新进行时序恢复，以确保 Blocked K-Fold 能够切分出连续的时间块
    print("🕰️ 正在重建数据的时序顺序以确保分块时间连续性 (Blocked K-Fold)...")
    from sklearn.model_selection import train_test_split
    from build_database import WaterDataLoader
    
    try:
        loader = WaterDataLoader()
        df = loader.get_all_data()
        df = df.dropna(subset=['耗用矾量（kg）']).copy()
        idx = df.index
        train_idx, _ = train_test_split(idx, test_size=0.2, random_state=42)
        
        sort_indices = np.argsort(train_idx)
        X_train_scaled = X_train_scaled[sort_indices]
        y_train = y_train[sort_indices]
        print(f"✅ 时序恢复成功！数据行数: {len(X_train_scaled)}")
    except Exception as e:
        print(f"⚠️ 时序恢复警告 (回退到原顺序): {e}")

    # 3. 训练 ISSA-XGBoost 模型以 provide 基线监督标签
    print("\n🚀 [第一步] 正在训练基础监督教师模型 ISSA-XGBoost...")
    start_time = time.time()
    issa = ISSA_XGBoost(pop_size=10, max_iter=10)
    best_params = issa.optimize(X_train_scaled, y_train)
    
    xgb_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train_scaled, y_train)
    print(f"✅ XGBoost 教师模型拟合完成，耗时 {(time.time() - start_time):.2f} 秒。")
    
    # 为训练集生成 XGBoost 监督基准值
    y_train_xgb = xgb_model.predict(X_train_scaled)

    # 4. 开启 5 折渐进式混合 ResNet 训练
    input_dim = X_train_scaled.shape[1]
    print(f"\n🚀 [第二步] 开始进行 5 折时间分块双头残差训练...")
    
    trained_models = train_hybrid_resnet(
        X_train_scaled, 
        y_train, 
        y_train_xgb, 
        input_dim, 
        run_dir=run_dir
    )

    # 5. 创建 5 折集成的 Regressor 包装器
    regressor = ResNetRegressor(trained_models, input_dim=input_dim)

    # 6. 验证模型并全量生成绘图预测
    print("\n📊 正在使用 5 折双头集成模型进行测试集评估与全量推断...")
    y_pred = regressor.predict(X_test_scaled)
    y_full_pred = regressor.predict(X_full_scaled)
        
    evaluate_and_plot(
        y_test, 
        y_pred, 
        y_full, 
        y_full_pred, 
        full_dates, 
        model_name="ResNet", 
        plot_dir=run_dir, 
        water_test=water_test, 
        water_full=water_full
    )

    # 7. 序列化并打包保存模型至 models/resnet 目录，以便 ui_app 调用
    import joblib
    os.makedirs('models/resnet', exist_ok=True)
    joblib.dump(regressor, 'models/resnet/best_model.pkl')
    print("\n✅ 双头监督残差集成模型已成功保存至 models/resnet/best_model.pkl，可以用于 ui_app 部署！")

if __name__ == "__main__":
    main()
