"""
文件名称：refactoredModel/mlp_gd_train.py
所属类别：重构核心生产代码 (Refactored Core Production)

功能描述：
    使用 PyTorch 训练五折时序交叉验证 (Blocked K-Fold) 集成残差网络 (ResNetRegressor) 的脚本。
    执行步骤包括：
    1. 统一加载水厂工况数据集，并进行时序重排列，以进行分块时序交叉验证；
    2. 训练 5 折的 `AlumDosageResNet` 回归器；
    3. 将 5 折的模型打包为 Scikit-Learn 兼容的 `ResNetRegressor`，并在测试集上测试指标，生成折线图；
    4. 将模型权重及结构序列化为 `models/resnet/best_model.pkl`。

运行与使用方法：
    直接在控制台执行重训：
    python mlp_gd_train.py

调用与依赖关系：
    - 依赖于 `utils.py` 内部的 `AlumDosageResNet`、`ResNetRegressor` 结构及绘图、数据函数。
    - 保存的文件直接用于 GUI `ui_app.py` 运行时加载。

设计细节与关键备注：
    - 尽管文件名包含 `mlp_gd`，但该模块经重构后已改为标准的 PyTorch 神经网络残差学习 (ResNet) 模式，以 AdamW 优化器配合余弦退火和提前终止进行权重拟合。
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
from utils import (
    load_and_preprocess_data, 
    evaluate_and_plot, 
    setup_logger_and_dir, 
    ResidualBlock, 
    AlumDosageResNet, 
    ResNetRegressor
)

# 设定中文字体
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Songti SC', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def train_resnet(train_x_all, train_y_all, input_dim, epochs=2000, run_dir=None):
    """时间分块 5 折交叉验证集成训练，使用 AdamW + 余弦退火学习率调度"""
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=5, shuffle=False)
    trained_models = []
    
    criterion = nn.MSELoss()
    
    # 5折交叉验证
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_x_all)):
        print(f"\n==================== 训练第 {fold + 1} 折 (Fold {fold + 1}/5) ====================")
        
        # 提取当前折的训练集和验证集（连续时序分块）
        X_tr, y_tr = train_x_all[train_idx], train_y_all[train_idx]
        X_va, y_va = train_x_all[val_idx], train_y_all[val_idx]
        
        # 转换为张量
        X_tr_tensor = torch.FloatTensor(X_tr)
        y_tr_tensor = torch.FloatTensor(y_tr).view(-1, 1)
        X_va_tensor = torch.FloatTensor(X_va)
        y_va_tensor = torch.FloatTensor(y_va).view(-1, 1)
        
        # 封装 DataLoader
        BATCH_SIZE = 64
        train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # 实例化当折的模型
        model = AlumDosageResNet(input_dim)
        
        # 使用 AdamW 优化器与余弦退火学习率调度器
        optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        best_loss = float('inf')
        best_model_weights = copy.deepcopy(model.state_dict())
        patience = 150
        patience_counter = 0
        
        fold_train_losses = []
        fold_val_losses = []
        
        start_time = time.time()
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_x.size(0)
                
            epoch_loss /= len(train_loader.dataset)
            fold_train_losses.append(epoch_loss)
            
            # 学习率更新
            scheduler.step()
            
            # 验证集评估
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_va_tensor)
                val_loss = criterion(val_outputs, y_va_tensor).item()
                fold_val_losses.append(val_loss)
                
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  [Epoch {epoch+1}/{epochs}] Train MSE: {epoch_loss:.2f} | Val MSE: {val_loss:.2f} | LR: {current_lr:.6f}")
                
            if patience_counter >= patience:
                print(f"  [Early Stopping] 第 {epoch+1} 轮提前停止。")
                break
                
        print(f"⏱️ Fold {fold + 1} 训练耗时: {(time.time() - start_time):.2f} 秒 | 最优验证集 MSE: {best_loss:.2f}")
        
        # 加载当折最优参数并保存
        model.load_state_dict(best_model_weights)
        trained_models.append(model)
        
        # 绘制该折的损失曲线
        if run_dir:
            plt.figure(figsize=(10, 5))
            plt.plot(fold_train_losses, label='Train Loss')
            plt.plot(fold_val_losses, label='Validation Loss')
            plt.title(f'Fold {fold + 1} Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('MSE Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(run_dir, f'resnet_loss_curve_fold_{fold + 1}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
    return trained_models

def main():
    run_dir = setup_logger_and_dir("resnet_train")
    
    print("=" * 60)
    print("🌟 PyTorch ResNet (残差神经网络) Blocked 5-Fold 集成模型 🌟")
    print("=" * 60)
    
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

    # 3. 开启 5 折交叉验证集成训练
    input_dim = X_train_scaled.shape[1]
    epochs = 2000
    print(f"\n🚀 开始进行 5 折时间分块交叉验证训练 (最大 Epoch = {epochs})...")
    
    trained_models = train_resnet(
        X_train_scaled, 
        y_train, 
        input_dim, 
        epochs=epochs, 
        run_dir=run_dir
    )

    # 4. 创建 5 折集成的 Regressor 包装器
    regressor = ResNetRegressor(trained_models, input_dim=input_dim)

    # 5. 验证模型并全量生成绘图预测
    print("\n📊 正在使用 5 折集成模型进行测试集评估与全量推断...")
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

    # 6. 序列化并打包保存模型至 models/resnet 目录，以便 ui_app 调用
    import joblib
    os.makedirs('models/resnet', exist_ok=True)
    joblib.dump(regressor, 'models/resnet/best_model.pkl')
    print("\n✅ ResNet 5折集成模型已成功保存至 models/resnet/best_model.pkl，可以用于 ui_app 部署！")

if __name__ == "__main__":
    main()
