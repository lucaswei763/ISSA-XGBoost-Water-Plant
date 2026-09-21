# lstm_xgb_stacked_v4.py
"""
LSTM + XGBoost 堆叠模型 - 兼容最新 pandas
"""

import os
import sys
import json
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Songti SC', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 数据文件路径可通过环境变量覆盖
DATA_FILE = os.getenv(
    'WATER_DATA_FILE',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'water_data.xlsx'),
)

# 特征配置 - 匹配实际数据列名
FEATURE_CONFIG = {
    'time_series_features': [
        '浑浊度_0点',
        '原水量_Km3',
        '温度_C',
        '氨氮_mg_per_L',
        'pH值',
        '库区水位_m',
        '消耗电_kWh',
    ],
    'target': '耗用矾量_kg',
    'date_col': '日期',
}

# LSTM 超参数
LSTM_CONFIG = {
    'seq_length': 7,
    'input_size': 7,
    'hidden_size': 64,
    'num_layers': 2,
    'output_size': 32,
    'batch_size': 32,
    'epochs': 150,
    'learning_rate': 0.001,
    'dropout': 0.2,
}

# XGBoost 超参数
XGB_CONFIG = {
    'n_estimators': 300,
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
    """LSTM 时序特征编码器"""

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
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, output_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_size * 2, output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        deep_features = self.fc(last_hidden)
        return deep_features


# ==================== 数据加载器 ====================
class WaterDataLoaderV4:
    """水厂数据加载器 - 直接读取 Excel"""

    def __init__(self, excel_path=None):
        self.excel_path = excel_path or DATA_FILE
        self.feature_cols = FEATURE_CONFIG['time_series_features']
        self.target_col = FEATURE_CONFIG['target']
        self.date_col = FEATURE_CONFIG['date_col']

    def load_and_clean(self):
        """加载并清洗数据"""
        print(f"📂 读取数据文件: {self.excel_path}")

        if not os.path.exists(self.excel_path):
            print(f"❌ 文件不存在: {self.excel_path}")
            print("请检查文件路径是否正确")
            return None

        # 读取 Excel
        try:
            df = pd.read_excel(self.excel_path, engine='openpyxl')
        except Exception as e:
            print(f"❌ 读取 Excel 失败: {e}")
            try:
                df = pd.read_excel(self.excel_path)
            except Exception as e2:
                print(f"❌ 仍然失败: {e2}")
                return None

        print(f"📊 原始数据: {len(df)} 条记录")
        print(f"   列名: {list(df.columns)}")

        # 检查必要的列是否存在
        available_features = []
        for col in self.feature_cols:
            if col in df.columns:
                available_features.append(col)
            else:
                # 模糊匹配
                found = False
                for c in df.columns:
                    if '浊度' in col and '浊度' in c:
                        available_features.append(c)
                        found = True
                        break
                    elif '原水' in col and '原水' in c:
                        available_features.append(c)
                        found = True
                        break
                    elif '温度' in col and '温度' in c:
                        available_features.append(c)
                        found = True
                        break
                    elif '氨氮' in col and '氨氮' in c:
                        available_features.append(c)
                        found = True
                        break
                    elif 'pH' in col and 'pH' in c:
                        available_features.append(c)
                        found = True
                        break
                    elif '库区' in col and '库区' in c:
                        available_features.append(c)
                        found = True
                        break
                    elif '消耗电' in col and '消耗电' in c:
                        available_features.append(c)
                        found = True
                        break
                if not found:
                    print(f"⚠️ 未找到特征: {col}")

        self.feature_cols = available_features

        # 检查目标列
        if self.target_col not in df.columns:
            for c in df.columns:
                if '矾' in c or 'alum' in c.lower():
                    self.target_col = c
                    break

        # 检查日期列
        if self.date_col not in df.columns:
            for c in df.columns:
                if '日期' in c or 'date' in c.lower():
                    self.date_col = c
                    break

        # 提取需要的列
        cols_to_keep = self.feature_cols + [self.target_col, self.date_col]
        df_clean = df[cols_to_keep].copy()

        # 转换数据类型
        for col in self.feature_cols + [self.target_col]:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # 处理日期
        df_clean[self.date_col] = pd.to_datetime(df_clean[self.date_col], errors='coerce')

        # 删除日期无效的行
        df_clean = df_clean.dropna(subset=[self.date_col])

        # 填充缺失值 - 使用 ffill 和 bfill（兼容新版 pandas）
        numeric_cols = self.feature_cols + [self.target_col]
        df_clean[numeric_cols] = df_clean[numeric_cols].ffill()
        df_clean[numeric_cols] = df_clean[numeric_cols].bfill()
        # 如果还有缺失值，用中位数填充
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)

        # 过滤异常值（投矾量 > 0 且 < 10000）
        df_clean = df_clean[df_clean[self.target_col] > 0]
        df_clean = df_clean[df_clean[self.target_col] < 10000]

        # 按日期排序
        df_clean = df_clean.sort_values(self.date_col).reset_index(drop=True)

        print(f"✅ 清洗后数据: {len(df_clean)} 条记录")
        print(f"   特征列: {self.feature_cols}")
        print(f"   目标列: {self.target_col}")
        print(f"   日期范围: {df_clean[self.date_col].min()} ~ {df_clean[self.date_col].max()}")

        # 打印数据统计
        print(f"\n📈 数据统计:")
        print(df_clean[self.feature_cols + [self.target_col]].describe())

        return df_clean

    def create_sequences(self, df, seq_length=7):
        """创建时序序列"""
        data = df[self.feature_cols + [self.target_col]].values

        X_seq_list = []
        X_current_list = []
        y_list = []
        dates_list = []

        for i in range(seq_length, len(data)):
            seq_data = data[i - seq_length:i, :len(self.feature_cols)]
            X_seq_list.append(seq_data)
            current_data = data[i, :len(self.feature_cols)]
            X_current_list.append(current_data)
            y_list.append(data[i, -1])
            dates_list.append(df.iloc[i][self.date_col])

        X_seq = np.array(X_seq_list, dtype=np.float32)
        X_current = np.array(X_current_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        dates = np.array(dates_list)

        print(f"\n📊 生成时序数据: {len(X_seq)} 个样本")
        print(f"   时序序列形状: {X_seq.shape}")
        print(f"   当前特征形状: {X_current.shape}")

        return X_seq, X_current, y, dates


# ==================== LSTM 训练器 ====================
class LSTMTrainer:
    """训练 LSTM 编码器"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.pred_layer = None

    def train(self, X_seq, X_current, y, val_split=0.2):
        """训练 LSTM 编码器"""
        # 标准化
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

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

        # 创建预测层
        self.pred_layer = nn.Linear(
            self.config['output_size'] + self.config['input_size'], 1
        )

        # 定义优化器和损失函数
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.pred_layer.parameters()),
            lr=self.config['learning_rate']
        )

        # 训练循环
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 30
        patience_counter = 0

        print(f"\n🚀 开始训练 LSTM 编码器...")
        print(f"   时序输入维度: {self.config['input_size']}")
        print(f"   隐藏层维度: {self.config['hidden_size']}")
        print(f"   深度特征维度: {self.config['output_size']}")
        print(f"   训练轮数: {self.config['epochs']}")

        for epoch in range(self.config['epochs']):
            self.model.train()
            self.pred_layer.train()
            epoch_loss = 0.0

            for batch_X_seq, batch_X_current, batch_y in train_loader:
                deep_features = self.model(batch_X_seq)
                combined = torch.cat([deep_features, batch_X_current], dim=1)
                pred = self.pred_layer(combined)
                loss = criterion(pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X_seq.size(0)

            epoch_loss /= len(train_loader.dataset)
            train_losses.append(epoch_loss)

            # 验证
            self.model.eval()
            self.pred_layer.eval()
            with torch.no_grad():
                val_deep = self.model(X_seq_val_t)
                val_combined = torch.cat([val_deep, X_current_val_t], dim=1)
                val_pred = self.pred_layer(val_combined)
                val_loss = criterion(val_pred, y_val_t).item()
                val_losses.append(val_loss)

            if (epoch + 1) % 20 == 0:
                print(
                    f"   [Epoch {epoch + 1}/{self.config['epochs']}] Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f}")

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   [Early Stopping] 在第 {epoch + 1} 轮提前停止")
                    break

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
        """保存模型"""
        os.makedirs('models', exist_ok=True)
        torch.save(self.model.state_dict(), f'{path_prefix}.pth')
        joblib.dump(self.scaler_X, f'{path_prefix}_scaler_X.pkl')
        joblib.dump(self.scaler_y, f'{path_prefix}_scaler_y.pkl')

        config = self.config.copy()
        with open(f'{path_prefix}_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ LSTM 模型已保存至: {path_prefix}")

    def load(self, path_prefix='models/lstm_encoder'):
        """加载模型"""
        with open(f'{path_prefix}_config.json', 'r') as f:
            self.config = json.load(f)

        self.model = LSTMEncoder(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            output_size=self.config['output_size'],
            dropout=self.config['dropout']
        )
        self.model.load_state_dict(torch.load(f'{path_prefix}.pth', map_location='cpu'))
        self.model.eval()

        self.scaler_X = joblib.load(f'{path_prefix}_scaler_X.pkl')
        self.scaler_y = joblib.load(f'{path_prefix}_scaler_y.pkl')

        print(f"✅ LSTM 模型已加载: {path_prefix}")
        return self


# ==================== LSTM + XGBoost 堆叠模型 ====================
class LSTMXGBStacked:
    """LSTM + XGBoost 堆叠预测模型"""

    def __init__(self, lstm_config=None, xgb_config=None, seq_length=7):
        self.lstm_config = lstm_config or LSTM_CONFIG
        self.xgb_config = xgb_config or XGB_CONFIG
        self.seq_length = seq_length
        self.lstm_trainer = None
        self.xgb_model = None
        self.feature_names = None

    def fit(self, df, test_size=0.2):
        """训练堆叠模型"""
        print("\n" + "=" * 60)
        print("🌟 LSTM + XGBoost 堆叠模型训练")
        print("=" * 60)

        # 1. 准备时序数据
        self.loader = WaterDataLoaderV4()
        self.feature_names = self.loader.feature_cols

        # 更新 LSTM 配置
        self.lstm_config['input_size'] = len(self.feature_names)

        X_seq, X_current, y, dates = self.loader.create_sequences(df, self.seq_length)

        # 划分训练/测试集
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
        print(f"   训练集: {len(X_seq_train)} 样本")
        print(f"   测试集: {len(X_seq_test)} 样本")

        # 2. 训练 LSTM 编码器
        print("\n" + "-" * 40)
        print("阶段 1: 训练 LSTM 编码器")
        print("-" * 40)

        self.lstm_trainer = LSTMTrainer(self.lstm_config)
        train_losses, val_losses = self.lstm_trainer.train(
            X_seq_train, X_current_train, y_train
        )

        # 绘制 LSTM 训练曲线
        os.makedirs('outputs', exist_ok=True)
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

        X_seq_combined = np.concatenate([X_seq_train, X_seq_test], axis=0)
        deep_features = self.lstm_trainer.extract_features(X_seq_combined)

        deep_train = deep_features[:len(X_seq_train)]
        deep_test = deep_features[len(X_seq_train):]

        X_train_combined = np.concatenate([deep_train, X_current_train], axis=1)
        X_test_combined = np.concatenate([deep_test, X_current_test], axis=1)

        print(f"   拼接特征维度: {X_train_combined.shape[1]}")
        print(f"     (深度特征 {deep_train.shape[1]} 维 + 当日特征 {X_current_train.shape[1]} 维)")

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
            y_train,
            eval_set=[(X_train_combined, y_train), (X_test_combined, y_test)],
            verbose=False
        )

        # 5. 评估
        print("\n" + "-" * 40)
        print("阶段 4: 模型评估")
        print("-" * 40)

        y_pred_train = self.xgb_model.predict(X_train_combined)
        y_pred_test = self.xgb_model.predict(X_test_combined)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)

        print(f"   训练集 RMSE: {train_rmse:.2f}")
        print(f"   测试集 RMSE: {test_rmse:.2f}")
        print(f"   测试集 MAE: {test_mae:.2f}")
        print(f"   测试集 R²: {test_r2:.4f}")

        # 保存结果
        self.training_results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'y_test': y_test,
            'y_pred_test': y_pred_test,
            'dates_test': dates_test,
        }

        print("\n✅ 堆叠模型训练完成!")

        return self

    def predict(self, X_seq, X_current):
        """预测"""
        deep_features = self.lstm_trainer.extract_features(X_seq)
        X_combined = np.concatenate([deep_features, X_current], axis=1)
        predictions = self.xgb_model.predict(X_combined)
        return predictions

    def save(self, path_prefix='models/lstm_xgb_stacked'):
        """保存模型"""
        os.makedirs('models', exist_ok=True)

        self.lstm_trainer.save(f'{path_prefix}_lstm')
        joblib.dump(self.xgb_model, f'{path_prefix}_xgb.pkl')

        config = {
            'lstm_config': self.lstm_config,
            'xgb_config': self.xgb_config,
            'seq_length': self.seq_length,
            'feature_names': self.feature_names,
        }
        with open(f'{path_prefix}_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ 堆叠模型已保存至: {path_prefix}")

    def load(self, path_prefix='models/lstm_xgb_stacked'):
        """加载模型"""
        with open(f'{path_prefix}_config.json', 'r') as f:
            config = json.load(f)

        self.lstm_config = config['lstm_config']
        self.xgb_config = config['xgb_config']
        self.seq_length = config['seq_length']
        self.feature_names = config['feature_names']

        self.lstm_trainer = LSTMTrainer(self.lstm_config)
        self.lstm_trainer.load(f'{path_prefix}_lstm')

        self.xgb_model = joblib.load(f'{path_prefix}_xgb.pkl')

        print(f"✅ 堆叠模型已加载: {path_prefix}")
        return self


# ==================== 主训练脚本 ====================
def main():
    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("=" * 40)
    print("🌟 LSTM + XGBoost 堆叠预测模型 🌟")
    print("=" * 40)

    # 1. 加载数据
    print("\n🔍 读取数据...")
    loader = WaterDataLoaderV4()
    df = loader.load_and_clean()

    if df is None or len(df) == 0:
        print("❌ 数据加载失败，请检查数据文件路径")
        return

    # 2. 训练堆叠模型
    model = LSTMXGBStacked(
        lstm_config=LSTM_CONFIG,
        xgb_config=XGB_CONFIG,
        seq_length=7
    )
    model.fit(df, test_size=0.2)

    # 3. 保存模型
    model.save('models/lstm_xgb_stacked')

    # 4. 绘制预测效果图
    results = model.training_results
    if results:
        plt.figure(figsize=(14, 6))
        n_show = min(200, len(results['y_test']))
        plt.plot(results['y_test'][:n_show], label='Actual', alpha=0.7)
        plt.plot(results['y_pred_test'][:n_show], label='Predicted', alpha=0.7)
        plt.title('LSTM+XGBoost 堆叠模型预测效果 (测试集)')
        plt.xlabel('Sample')
        plt.ylabel('投矾量 (kg)')
        plt.legend()
        plt.grid(True)
        plt.savefig('outputs/lstm_xgb_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 预测效果图已保存: outputs/lstm_xgb_predictions.png")

    print("\n" + "=" * 40)
    print("🎉 LSTM + XGBoost 堆叠模型训练完成!")
    print("=" * 40)


if __name__ == "__main__":
    main()
