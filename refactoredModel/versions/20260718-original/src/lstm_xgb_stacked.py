# lstm_xgb_stacked.py
"""
LSTM + XGBoost 堆叠模型
架构：
  1. LSTM 编码器：提取前 N 天时序数据的深度特征 (32维)
  2. 特征拼接：深度特征 + 当日原始特征
  3. XGBoost：最终预测投矾量
"""

import os
import time
import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

from utils import load_and_preprocess_data, evaluate_and_plot, setup_logger_and_dir
from build_database import WaterDataLoader

# ==================== 配置 ====================
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Songti SC', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# LSTM 超参数
LSTM_CONFIG = {
    'seq_length': 7,  # 使用前7天数据
    'input_size': 6,  # 时序特征数量（浊度、流量、pH、温度、氨氮、耗矾量）
    'hidden_size': 64,  # LSTM 隐藏层维度
    'num_layers': 2,  # LSTM 层数
    'output_size': 32,  # 深度特征向量维度
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'dropout': 0.2,
}

# XGBoost 超参数（可通过 ISSA 优化）
XGB_CONFIG = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.08,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 2.0,
    'reg_alpha': 0.1,
    'random_state': 42,
}


# ==================== LSTM 编码器 ====================
class LSTMEncoder(nn.Module):
    """
    LSTM 时序特征编码器
    输入: [batch, seq_len, input_size]
    输出: [batch, output_size] 深度特征向量
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # 双向LSTM，捕捉更多时序信息
        )

        # 将 LSTM 输出映射到固定长度的深度特征向量
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, output_size * 2),  # 双向所以 *2
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_size * 2, output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 取最后时刻的隐藏状态 (双向拼接)
        # h_n: [num_layers * 2, batch, hidden_size]
        # 取最后一层双向的隐藏状态
        last_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        # last_hidden: [batch, hidden_size * 2]
        deep_features = self.fc(last_hidden)
        # deep_features: [batch, output_size]
        return deep_features


# ==================== 数据准备器 ====================
class TimeSeriesDataPreparer:
    """为 LSTM+XGBoost 准备时序数据"""

    def __init__(self, seq_length=7, feature_cols=None, target_col='耗用矾量（kg）'):
        self.seq_length = seq_length
        self.feature_cols = feature_cols or [
            '浑浊度（NTU）_0点',
            '原水量（Km³）',
            '温度（℃）_9点',
            '氨氮（mg/L）_9点',
            'pH值_9点'
        ]
        self.target_col = target_col
        self.scaler_X = None
        self.scaler_y = None

    def load_data_from_db(self, db_path='data/water_data.db'):
        """从数据库加载数据"""
        loader = WaterDataLoader(db_path)
        df = loader.get_all_data()

        # 统一列名
        available_cols = []
        for col in self.feature_cols:
            if col in df.columns:
                available_cols.append(col)
            else:
                # 尝试模糊匹配
                for c in df.columns:
                    if '浊度' in c and '浊度' in col:
                        available_cols.append(c)
                        break
                    elif '原水' in c and '原水' in col:
                        available_cols.append(c)
                        break
                    elif '温度' in c and '温度' in col:
                        available_cols.append(c)
                        break
                    elif '氨氮' in c and '氨氮' in col:
                        available_cols.append(c)
                        break
                    elif 'pH' in c and 'pH' in col:
                        available_cols.append(c)
                        break

        self.feature_cols = available_cols

        # 检查目标列
        if self.target_col not in df.columns:
            # 尝试找到耗用矾量列
            for col in df.columns:
                if '矾' in col or 'alum' in col.lower():
                    self.target_col = col
                    break

        # 提取数据
        cols = self.feature_cols + [self.target_col, '日期']
        df_clean = df[cols].copy()

        # 处理缺失值
        numeric_cols = self.feature_cols + [self.target_col]
        df_clean[numeric_cols] = df_clean[numeric_cols].astype(float)
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill').fillna(method='bfill')

        # 过滤异常值
        df_clean = df_clean[df_clean[self.target_col] <= 5000]
        df_clean = df_clean[df_clean[self.target_col] > 0]

        # 按日期排序
        df_clean['日期'] = pd.to_datetime(df_clean['日期'])
        df_clean = df_clean.sort_values('日期').reset_index(drop=True)

        print(f"✅ 加载数据: {len(df_clean)} 条记录")
        print(f"   - 特征列: {self.feature_cols}")
        print(f"   - 目标列: {self.target_col}")

        return df_clean

    def create_sequences(self, df):
        """
        创建时序序列
        返回:
          - X_seq: 时序输入 [samples, seq_length, input_size]
          - X_current: 当日特征 [samples, current_features]
          - y: 目标值 [samples]
        """
        data = df[self.feature_cols + [self.target_col]].values

        X_seq_list = []
        X_current_list = []
        y_list = []
        dates_list = []

        for i in range(self.seq_length, len(data)):
            # 前 seq_length 天的时序数据
            seq_data = data[i - self.seq_length:i, :len(self.feature_cols)]
            X_seq_list.append(seq_data)

            # 当日特征（只取当日原始特征，不包含lag特征）
            current_data = data[i, :len(self.feature_cols)]
            X_current_list.append(current_data)

            # 目标值
            y_list.append(data[i, -1])

            # 日期
            dates_list.append(df.iloc[i]['日期'])

        X_seq = np.array(X_seq_list, dtype=np.float32)
        X_current = np.array(X_current_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        dates = np.array(dates_list)

        print(f"📊 生成时序数据: {len(X_seq)} 个样本")
        print(f"   - 时序序列形状: {X_seq.shape}")
        print(f"   - 当日特征形状: {X_current.shape}")

        return X_seq, X_current, y, dates

    def prepare_train_test(self, df, test_size=0.2):
        """准备训练集和测试集"""
        X_seq, X_current, y, dates = self.create_sequences(df)

        # 划分时间序列数据（按时间顺序，不打乱）
        split_idx = int(len(X_seq) * (1 - test_size))

        X_seq_train = X_seq[:split_idx]
        X_seq_test = X_seq[split_idx:]
        X_current_train = X_current[:split_idx]
        X_current_test = X_current[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        dates_train = dates[:split_idx]
        dates_test = dates[split_idx:]

        print(f"\n📁 数据集划分:")
        print(f"   - 训练集: {len(X_seq_train)} 样本")
        print(f"   - 测试集: {len(X_seq_test)} 样本")

        return {
            'X_seq_train': X_seq_train,
            'X_seq_test': X_seq_test,
            'X_current_train': X_current_train,
            'X_current_test': X_current_test,
            'y_train': y_train,
            'y_test': y_test,
            'dates_train': dates_train,
            'dates_test': dates_test,
        }


# ==================== LSTM 训练器 ====================
class LSTMTrainer:
    """训练 LSTM 编码器"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler_X = None
        self.scaler_y = None

    def train(self, X_seq, X_current, y, val_split=0.2):
        """
        训练 LSTM 编码器
        使用重建损失 + 预测损失 联合训练
        """
        # 标准化
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        # 重塑 X_seq 进行标准化: [samples, seq_len, features]
        n_samples, seq_len, n_features = X_seq.shape
        X_seq_flat = X_seq.reshape(-1, n_features)
        X_seq_scaled = self.scaler_X.fit_transform(X_seq_flat).reshape(n_samples, seq_len, n_features)

        X_current_scaled = self.scaler_X.transform(X_current)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # 划分训练/验证集
        split_idx = int(len(X_seq_scaled) * (1 - val_split))

        X_seq_train = X_seq_scaled[:split_idx]
        X_seq_val = X_seq_scaled[split_idx:]
        X_current_train = X_current_scaled[:split_idx]
        X_current_val = X_current_scaled[split_idx:]
        y_train = y_scaled[:split_idx]
        y_val = y_scaled[split_idx:]

        # 转换为 PyTorch 张量
        X_seq_train_t = torch.FloatTensor(X_seq_train)
        X_seq_val_t = torch.FloatTensor(X_seq_val)
        X_current_train_t = torch.FloatTensor(X_current_train)
        X_current_val_t = torch.FloatTensor(X_current_val)
        y_train_t = torch.FloatTensor(y_train).view(-1, 1)
        y_val_t = torch.FloatTensor(y_val).view(-1, 1)

        # 创建 DataLoader
        train_dataset = TensorDataset(X_seq_train_t, X_current_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)

        # 初始化 LSTM 模型
        self.model = LSTMEncoder(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            output_size=self.config['output_size'],
            dropout=self.config['dropout']
        )

        # 定义优化器和损失函数
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

        # 训练循环
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0

        print(f"\n🚀 开始训练 LSTM 编码器...")
        print(f"   - 时序输入维度: {self.config['input_size']}")
        print(f"   - 隐藏层维度: {self.config['hidden_size']}")
        print(f"   - 深度特征维度: {self.config['output_size']}")
        print(f"   - 训练轮数: {self.config['epochs']}")

        for epoch in range(self.config['epochs']):
            self.model.train()
            epoch_loss = 0.0

            for batch_X_seq, batch_X_current, batch_y in train_loader:
                # 前向传播
                deep_features = self.model(batch_X_seq)
                # 拼接深度特征和当前特征用于预测
                combined = torch.cat([deep_features, batch_X_current], dim=1)
                # 使用一个简单的线性层进行预测（辅助任务）
                pred = torch.nn.Linear(
                    self.config['output_size'] + self.config['input_size'], 1
                ).to(deep_features.device)(combined) if not hasattr(self, '_pred_layer') else self._pred_layer(combined)

                # 这里简化：直接用 LSTM 输出的深度特征预测目标
                # 实际上我们可以加一个小的预测头
                if not hasattr(self, '_pred_layer'):
                    self._pred_layer = nn.Linear(
                        self.config['output_size'] + self.config['input_size'], 1
                    )

                pred = self._pred_layer(combined)
                loss = criterion(pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X_seq.size(0)

            epoch_loss /= len(train_loader.dataset)
            train_losses.append(epoch_loss)

            # 验证
            self.model.eval()
            with torch.no_grad():
                val_deep = self.model(X_seq_val_t)
                val_combined = torch.cat([val_deep, X_current_val_t], dim=1)
                val_pred = self._pred_layer(val_combined)
                val_loss = criterion(val_pred, y_val_t).item()
                val_losses.append(val_loss)

            if (epoch + 1) % 20 == 0:
                print(
                    f"  [Epoch {epoch + 1}/{self.config['epochs']}] Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f}")

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  [Early Stopping] 在第 {epoch + 1} 轮提前停止")
                    break

        # 加载最佳模型
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)

        print(f"✅ LSTM 训练完成，最佳验证损失: {best_val_loss:.6f}")

        return train_losses, val_losses

    def extract_features(self, X_seq):
        """提取深度特征"""
        self.model.eval()
        X_seq_scaled = self.scaler_X.transform(
            X_seq.reshape(-1, X_seq.shape[-1])
        ).reshape(X_seq.shape)

        X_seq_t = torch.FloatTensor(X_seq_scaled)
        with torch.no_grad():
            deep_features = self.model(X_seq_t).numpy()

        return deep_features

    def save(self, path_prefix='models/lstm_encoder'):
        """保存 LSTM 模型和 Scaler"""
        os.makedirs('models', exist_ok=True)

        # 保存模型
        torch.save(self.model.state_dict(), f'{path_prefix}.pth')

        # 保存 scaler
        joblib.dump(self.scaler_X, f'{path_prefix}_scaler_X.pkl')
        joblib.dump(self.scaler_y, f'{path_prefix}_scaler_y.pkl')

        # 保存配置
        with open(f'{path_prefix}_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"✅ LSTM 模型已保存至: {path_prefix}")

    def load(self, path_prefix='models/lstm_encoder'):
        """加载 LSTM 模型和 Scaler"""
        # 加载配置
        with open(f'{path_prefix}_config.json', 'r') as f:
            self.config = json.load(f)

        # 初始化模型
        self.model = LSTMEncoder(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            output_size=self.config['output_size'],
            dropout=self.config['dropout']
        )

        # 加载权重
        self.model.load_state_dict(torch.load(f'{path_prefix}.pth', map_location='cpu'))
        self.model.eval()

        # 加载 scaler
        self.scaler_X = joblib.load(f'{path_prefix}_scaler_X.pkl')
        self.scaler_y = joblib.load(f'{path_prefix}_scaler_y.pkl')

        print(f"✅ LSTM 模型已加载: {path_prefix}")
        return self


# ==================== LSTM + XGBoost 堆叠模型 ====================
class LSTMXGBStacked:
    """
    LSTM + XGBoost 堆叠预测模型
    """

    def __init__(self, lstm_config=None, xgb_config=None, seq_length=7):
        self.lstm_config = lstm_config or LSTM_CONFIG
        self.xgb_config = xgb_config or XGB_CONFIG
        self.seq_length = seq_length

        self.lstm_trainer = None
        self.xgb_model = None
        self.feature_names = None

    def fit(self, df, test_size=0.2):
        """
        训练堆叠模型
        """
        print("\n" + "=" * 60)
        print("🌟 LSTM + XGBoost 堆叠模型训练")
        print("=" * 60)

        # 1. 准备时序数据
        preparer = TimeSeriesDataPreparer(seq_length=self.seq_length)
        data = preparer.prepare_train_test(df, test_size=test_size)

        # 保存特征名称
        self.feature_names = preparer.feature_cols

        # 2. 训练 LSTM 编码器
        print("\n" + "-" * 40)
        print("阶段 1: 训练 LSTM 编码器")
        print("-" * 40)

        self.lstm_trainer = LSTMTrainer(self.lstm_config)
        train_losses, val_losses = self.lstm_trainer.train(
            data['X_seq_train'],
            data['X_current_train'],
            data['y_train']
        )

        # 绘制 LSTM 训练曲线
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('LSTM Encoder Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('outputs/lstm_training_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ LSTM 损失曲线已保存: outputs/lstm_training_loss.png")

        # 3. 提取深度特征
        print("\n" + "-" * 40)
        print("阶段 2: 提取深度特征")
        print("-" * 40)

        X_seq_combined = np.concatenate([data['X_seq_train'], data['X_seq_test']], axis=0)
        deep_features = self.lstm_trainer.extract_features(X_seq_combined)

        # 拆分训练/测试
        n_train = len(data['X_seq_train'])
        deep_train = deep_features[:n_train]
        deep_test = deep_features[n_train:]

        # 拼接深度特征 + 当前特征
        X_train_combined = np.concatenate([deep_train, data['X_current_train']], axis=1)
        X_test_combined = np.concatenate([deep_test, data['X_current_test']], axis=1)

        print(f"   - 拼接特征维度: {X_train_combined.shape[1]}")
        print(f"     (深度特征 {deep_train.shape[1]} 维 + 当日特征 {data['X_current_train'].shape[1]} 维)")

        # 4. 训练 XGBoost
        print("\n" + "-" * 40)
        print("阶段 3: 训练 XGBoost 模型")
        print("-" * 40)

        self.xgb_model = XGBRegressor(
            **self.xgb_config,
            n_jobs=-1,
            verbosity=0
        )

        self.xgb_model.fit(
            X_train_combined,
            data['y_train'],
            eval_set=[(X_train_combined, data['y_train']), (X_test_combined, data['y_test'])],
            verbose=False
        )

        # 5. 评估
        print("\n" + "-" * 40)
        print("阶段 4: 模型评估")
        print("-" * 40)

        y_pred_train = self.xgb_model.predict(X_train_combined)
        y_pred_test = self.xgb_model.predict(X_test_combined)

        train_rmse = np.sqrt(mean_squared_error(data['y_train'], y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(data['y_test'], y_pred_test))
        test_mae = mean_absolute_error(data['y_test'], y_pred_test)
        test_r2 = r2_score(data['y_test'], y_pred_test)

        print(f"  训练集 RMSE: {train_rmse:.2f}")
        print(f"  测试集 RMSE: {test_rmse:.2f}")
        print(f"  测试集 MAE: {test_mae:.2f}")
        print(f"  测试集 R²: {test_r2:.4f}")

        # 保存结果
        self.training_results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'y_test': data['y_test'],
            'y_pred_test': y_pred_test,
            'dates_test': data['dates_test'],
        }

        print("\n✅ 堆叠模型训练完成!")

        return self

    def predict(self, X_seq, X_current):
        """
        预测
        X_seq: [samples, seq_length, input_size] 时序输入
        X_current: [samples, current_features] 当前特征
        """
        # 提取深度特征
        deep_features = self.lstm_trainer.extract_features(X_seq)

        # 拼接
        X_combined = np.concatenate([deep_features, X_current], axis=1)

        # XGBoost 预测
        predictions = self.xgb_model.predict(X_combined)

        return predictions

    def save(self, path_prefix='models/lstm_xgb_stacked'):
        """保存整个堆叠模型"""
        os.makedirs('models', exist_ok=True)

        # 保存 LSTM
        self.lstm_trainer.save(f'{path_prefix}_lstm')

        # 保存 XGBoost
        joblib.dump(self.xgb_model, f'{path_prefix}_xgb.pkl')

        # 保存配置
        config = {
            'lstm_config': self.lstm_config,
            'xgb_config': self.xgb_config,
            'seq_length': self.seq_length,
            'feature_names': self.feature_names,
        }
        with open(f'{path_prefix}_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # 保存特征拼接信息
        joblib.dump({
            'lstm_output_size': self.lstm_config['output_size'],
            'current_feature_size': len(self.feature_names),
        }, f'{path_prefix}_metadata.pkl')

        print(f"✅ 堆叠模型已保存至: {path_prefix}")

    def load(self, path_prefix='models/lstm_xgb_stacked'):
        """加载整个堆叠模型"""
        # 加载配置
        with open(f'{path_prefix}_config.json', 'r') as f:
            config = json.load(f)

        self.lstm_config = config['lstm_config']
        self.xgb_config = config['xgb_config']
        self.seq_length = config['seq_length']
        self.feature_names = config['feature_names']

        # 加载 LSTM
        self.lstm_trainer = LSTMTrainer(self.lstm_config)
        self.lstm_trainer.load(f'{path_prefix}_lstm')

        # 加载 XGBoost
        self.xgb_model = joblib.load(f'{path_prefix}_xgb.pkl')

        print(f"✅ 堆叠模型已加载: {path_prefix}")
        return self


# ==================== 主训练脚本 ====================
def main():
    # 设置日志和输出目录
    run_dir = setup_logger_and_dir("lstm_xgb_stacked")

    print("=" * 40)
    print("🌟 LSTM + XGBoost 堆叠预测模型 🌟")
    print("=" * 40)

    # 1. 加载数据
    print("\n🔍 读取数据...")
    try:
        loader = WaterDataLoader()
        df = loader.get_all_data()
        print(f"✅ 加载 {len(df)} 条记录")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 2. 数据预处理
    print("\n🔧 数据预处理...")
    # 使用与 utils.py 相同的特征列
    from utils import FEATURE_COLS, TARGET_COL

    # 确保所有必要的列存在
    available_feature_cols = []
    for col in FEATURE_COLS:
        if col in df.columns:
            available_feature_cols.append(col)
        else:
            # 尝试模糊匹配
            for c in df.columns:
                if '浊度' in c and '浊度' in col:
                    available_feature_cols.append(c)
                    break
                elif '原水' in c and '原水' in col:
                    available_feature_cols.append(c)
                    break
                elif '温度' in c and '温度' in col:
                    available_feature_cols.append(c)
                    break
                elif '氨氮' in c and '氨氮' in col:
                    available_feature_cols.append(c)
                    break
                elif 'pH' in c and 'pH' in col:
                    available_feature_cols.append(c)
                    break

    print(f"   - 使用特征: {available_feature_cols}")

    # 检查目标列
    target_col = TARGET_COL
    if target_col not in df.columns:
        for col in df.columns:
            if '矾' in col or 'alum' in col.lower():
                target_col = col
                break

    print(f"   - 目标列: {target_col}")

    # 提取并清洗数据
    cols = available_feature_cols + [target_col, '日期']
    df_clean = df[cols].copy()

    # 转换为数值
    for col in available_feature_cols + [target_col]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # 填充缺失值
    df_clean[available_feature_cols + [target_col]] = df_clean[available_feature_cols + [target_col]].fillna(
        method='ffill').fillna(method='bfill')

    # 过滤异常
    df_clean = df_clean[df_clean[target_col] <= 5000]
    df_clean = df_clean[df_clean[target_col] > 0]

    # 按日期排序
    df_clean['日期'] = pd.to_datetime(df_clean['日期'])
    df_clean = df_clean.sort_values('日期').reset_index(drop=True)

    print(f"   - 有效数据: {len(df_clean)} 条")

    # 3. 训练堆叠模型
    seq_length = 7
    lstm_config = {
        'seq_length': seq_length,
        'input_size': len(available_feature_cols),
        'hidden_size': 64,
        'num_layers': 2,
        'output_size': 32,
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'dropout': 0.2,
    }

    xgb_config = {
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.08,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 2.0,
        'reg_alpha': 0.1,
        'random_state': 42,
    }

    model = LSTMXGBStacked(lstm_config, xgb_config, seq_length)
    model.fit(df_clean, test_size=0.2)

    # 4. 保存模型
    model.save('models/lstm_xgb_stacked')

    # 5. 绘制预测结果
    results = model.training_results
    if results:
        plt.figure(figsize=(12, 6))
        plt.plot(results['y_test'][:100], label='Actual', alpha=0.7)
        plt.plot(results['y_pred_test'][:100], label='Predicted', alpha=0.7)
        plt.title('LSTM+XGBoost 堆叠模型预测效果 (测试集前100样本)')
        plt.xlabel('Sample')
        plt.ylabel('投矾量 (kg)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(run_dir, 'lstm_xgb_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 预测效果图已保存: {os.path.join(run_dir, 'lstm_xgb_predictions.png')}")

    print("\n" + "=" * 40)
    print("🎉 LSTM + XGBoost 堆叠模型训练完成!")
    print("=" * 40)


if __name__ == "__main__":
    main()