import time
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from utils import load_and_preprocess_data, evaluate_and_plot, setup_logger_and_dir

# 设定中文字体
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Songti SC', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class AlumDosageMLP(nn.Module):
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

def train_mlp(model, train_loader, val_x, val_y, criterion, optimizer, epochs=500, run_dir=None):
    """小批量梯度下降 + 验证集早停和损失曲线绘制"""
    train_losses = []
    val_losses = []
    
    best_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    patience = 50
    patience_counter = 0
    
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
        train_losses.append(epoch_loss)
        
        # 验证集评估
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_x)
            val_loss = criterion(val_outputs, val_y).item()
            val_losses.append(val_loss)
            
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch+1) % 50 == 0:
            print(f"  [Epoch {epoch+1}/{epochs}] Train MSE: {epoch_loss:.4f} | Val MSE: {val_loss:.4f}")
            
        if patience_counter >= patience:
            print(f"  [Early Stopping] 在第 {epoch+1} 轮提前停止，验证集损失不再下降。")
            break
            
    print(f"\n⏳ 训练耗时: {(time.time() - start_time):.2f} 秒")
    
    # 恢复最优参数
    model.load_state_dict(best_model_weights)
    
    # 绘制损失曲线
    if run_dir:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('MLP Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(run_dir, 'mlp_loss_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    return train_losses, val_losses

def main():
    run_dir = setup_logger_and_dir("mlp_gd_train")
    
    print("=" * 40)
    print("🌟 PyTorch MLP 小批量梯度下降预测模型 🌟")
    print("=" * 40)
    
    # 1. 统一加载数据
    print("\n🔍 读取并处理数据集中...")
    try:
        X_train_scaled, X_test_scaled, y_train, y_test, X_full_scaled, y_full, full_dates, water_test, water_full = load_and_preprocess_data()
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 转换为 PyTorch 张量
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

    # 2. 封装进 DataLoader（批量处理能够避免陷入局部最优，加速收敛）
    BATCH_SIZE = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"\n🚀 开始使用梯度下降(Adam)训练神经网络模型 (Batch Size = {BATCH_SIZE})...")
    model = AlumDosageMLP(input_dim=X_train_scaled.shape[1])
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4) 
    
    epochs = 500
    train_mlp(model, train_loader, X_test_tensor, y_test_tensor, criterion, optimizer, epochs=epochs, run_dir=run_dir)

    # 3. 验证模型并全量生成绘图预测
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy().flatten()
        y_full_pred = model(torch.FloatTensor(X_full_scaled)).numpy().flatten()
        
    evaluate_and_plot(y_test, y_pred, y_full, y_full_pred, full_dates, model_name="MLP", plot_dir=run_dir, water_test=water_test, water_full=water_full)

if __name__ == "__main__":
    main()
