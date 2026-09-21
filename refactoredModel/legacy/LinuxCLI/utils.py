"""
文件名称：refactoredModel/LinuxCLI/utils.py
所属类别：Linux CLI 部署代码 (Linux CLI Release)

功能描述：
    专为 Linux 服务器环境打包的底层辅助工具库。
    主要职责包括：
    1. 特征工程 (`add_engineered_features`)：将原始录入的浑浊度、氨氮等指标进行 7 个工艺派生维度的实时工程计算。
    2. 数据读取与预处理 (`load_and_preprocess_data`)：加载 SQLite 中的历史训练数据并拟合导出转换器。
    3. 神经网络结构定义：包含 `AlumDosageResNet` 等残差网络类以及 Scikit-learn 风格的多折平均回归器包装类 `ResNetRegressor`。

运行与使用方法：
    通常在 Python 脚本中被导入：
    from utils import add_engineered_features, ResNetRegressor

调用与依赖关系：
    - 被同级目录下的 `cli_app.py` 和 `predictor_service.py` 导入和调用。
    - 相比核心生产根目录下的 `utils.py`，本部署包的 `utils.py` 对 `build_database` 和 `matplotlib` 采用了延迟导入 (Lazy Import) 设计，防止没有安装图形绘图库的服务器在启动时崩溃。

设计细节与关键备注：
    - 这是一个自包含镜像，移除了大量前端绘图必需的包，是保障 Linux 端低依赖独立部署运行的关键。
"""
import os
import sys
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ==================== 重要：定义特征列和目标列（匹配数据库实际列名） ====================
TARGET_COL = '耗用矾量（kg）'

# 12个由数据库或用户输入的原始特征列
BASE_FEATURE_COLS = [
    '浑浊度（NTU）',
    '供水量（Km³）',
    '温度（℃）',
    '氨氮（mg/L）',
    'pH值',
    '菌落总数（CFU/mL）',
    '总大肠菌群（CFU/100mL）',
    '粪大肠菌群（CFU/1000mL）',
    '消耗电（kW·h）',
    '原水量（Km³）',
    '高锰酸盐指数（mg/L）',
    '库区水位（m）'
]

# 后台通过特征工程计算得出的 7 个复合列名
ENGINEERED_COLS = [
    '浑浊度_大于_线性期望',  # 二值化分类边界列 (浑浊度 > 51.64 * 氨氮 - 0.72)
    '浑浊度_氨氮_线性偏差',   # 连续线性残差列 (浑浊度 - 51.64 * 氨氮)
    '浑浊度_氨氮_比值',       # 连续比值列 (浑浊度 / (氨氮 + 1e-5))
    '产水率',                 # 供水量 / (原水量 + 1e-5)
    '制水损耗',               # 原水量 - 供水量
    '制水吨水单耗',           # 消耗电 / (供水量 + 1e-5)
    '原水吨水单耗'            # 消耗电 / (原水量 + 1e-5)
]

# 最终参与模型训练的 19 维特征列
FEATURE_COLS = BASE_FEATURE_COLS + ENGINEERED_COLS


def add_engineered_features(df_or_dict):
    """
    根据浑浊度与氨氮的拟合关系 浑浊度 = 51.64 * 氨氮 - 0.72，
    以及其他强相关指标，生成 7 个特征工程列。支持 pandas DataFrame 或单个字典/Series
    """
    if isinstance(df_or_dict, pd.DataFrame):
        df = df_or_dict.copy()
        # 1. 浑浊度与氨氮的二值判定
        df['浑浊度_大于_线性期望'] = (df['浑浊度（NTU）'] > (51.64 * df['氨氮（mg/L）'] - 0.72)).astype(float)
        # 2. 浑浊度与氨氮的线性偏差
        df['浑浊度_氨氮_线性偏差'] = df['浑浊度（NTU）'] - 51.64 * df['氨氮（mg/L）']
        # 3. 浑浊度与氨氮的比值
        df['浑浊度_氨氮_比值'] = df['浑浊度（NTU）'] / (df['氨氮（mg/L）'] + 1e-5)
        # 4. 产水率
        df['产水率'] = df['供水量（Km³）'] / (df['原水量（Km³）'] + 1e-5)
        # 5. 制水损耗
        df['制水损耗'] = df['原水量（Km³）'] - df['供水量（Km³）']
        # 6. 制水吨水单耗
        df['制水吨水单耗'] = df['消耗电（kW·h）'] / (df['供水量（Km³）'] + 1e-5)
        # 7. 原水吨水单耗
        df['原水吨水单耗'] = df['消耗电（kW·h）'] / (df['原水量（Km³）'] + 1e-5)
        return df
    else:
        # 假设是字典或类字典的结构
        data = df_or_dict.copy()
        turb = float(data.get('浑浊度（NTU）', 0.0))
        nh3 = float(data.get('氨氮（mg/L）', 0.0))
        supply = float(data.get('供水量（Km³）', 0.0))
        raw = float(data.get('原水量（Km³）', 0.0))
        power = float(data.get('消耗电（kW·h）', 0.0))
        
        data['浑浊度_大于_线性期望'] = 1.0 if turb > (51.64 * nh3 - 0.72) else 0.0
        data['浑浊度_氨氮_线性偏差'] = turb - 51.64 * nh3
        data['浑浊度_氨氮_比值'] = turb / (nh3 + 1e-5)
        data['产水率'] = supply / (raw + 1e-5)
        data['制水损耗'] = raw - supply
        data['制水吨水单耗'] = power / (supply + 1e-5)
        data['原水吨水单耗'] = power / (raw + 1e-5)
        return data


def setup_logger_and_dir(model_prefix):
    """
    建立按日期和精确分钟命名的嵌套目录，保存于 refactoredModel/outputs/ 下：
    - 每天一个文件夹：outputs/{model_prefix}-YYYY-MM-DD
    - 内部每次一个文件夹：HH-MM (精确到分钟)
    并将 sys.stdout 重定向到该文件夹下的 train_log.txt 记录文件中
    """
    now = datetime.datetime.now()
    
    # 自动获取 refactoredModel 文件夹路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_root = os.path.join(current_dir, "outputs")
    
    # 文件夹结构：每天一个主文件夹，每次运行以小时-分钟作为一个子文件夹
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M")
    
    run_dir = os.path.join(outputs_root, f"{model_prefix}-{date_str}", time_str)
    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "train_log.txt")

    class DualLogger:
        def __init__(self, log_path):
            self.terminal = sys.stdout
            self.log = open(log_path, "a", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = DualLogger(log_file)
    print(f"📄 开始记录日志至: {log_file}")
    
    return run_dir


def load_and_preprocess_data(model_dir='models'):
    """读取数据，清洗，然后划分训练测试集，正确应用插值和标准化，防止数据泄露"""
    from build_database import WaterDataLoader
    loader = WaterDataLoader()
    try:
        df = loader.get_all_data()
    except Exception as e:
        raise RuntimeError(f"数据库读取失败: {e}")

    # 清理非法字符
    df.replace(['/', '\\', '', ' '], np.nan, inplace=True)

    missing_cols = [c for c in BASE_FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        print(f"⚠️ 警告: 数据找不到对应的列 {missing_cols}")

    df = df.dropna(subset=[TARGET_COL]).copy()

    # 检查日期列是否存在以便后续绘图
    date_col = '日期' if '日期' in df.columns else None
    cols_to_convert = BASE_FEATURE_COLS + [TARGET_COL]
    
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

    # 提取需要的列数据，保留日期
    if date_col:
        mask_df = df[BASE_FEATURE_COLS + [TARGET_COL] + [date_col]]
    else:
        mask_df = df[BASE_FEATURE_COLS + [TARGET_COL]]

    idx = mask_df.index
    # 划分数据集
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    train_data = mask_df.loc[train_idx].copy()
    test_data = mask_df.loc[test_idx].copy()

    # 1. 采用 SimpleImputer 在训练集上拟合(fit)，并在测试集上只进行变换(transform)
    imputer = SimpleImputer(strategy='median')
    train_data[BASE_FEATURE_COLS] = imputer.fit_transform(train_data[BASE_FEATURE_COLS])
    test_data[BASE_FEATURE_COLS] = imputer.transform(test_data[BASE_FEATURE_COLS])

    # 获取全量按顺序排序的完整数据，专用于作图
    full_data = mask_df.copy()
    if date_col:
        full_data = full_data.sort_values(by=date_col)
    full_data[BASE_FEATURE_COLS] = imputer.transform(full_data[BASE_FEATURE_COLS])

    # 动态插入特征工程计算（包含二值列与连续比值/电耗差值列）
    train_data = add_engineered_features(train_data)
    test_data = add_engineered_features(test_data)
    full_data = add_engineered_features(full_data)

    X_train = train_data[FEATURE_COLS].values
    y_train = train_data[TARGET_COL].values
    X_test = test_data[FEATURE_COLS].values
    y_test = test_data[TARGET_COL].values

    X_full = full_data[FEATURE_COLS].values
    y_full = full_data[TARGET_COL].values
    full_dates = full_data[date_col].values if date_col else None

    # 提取供水量（Km³），即千吨水，所在位置为 FEATURE_COLS[1] ('供水量（Km³）')
    water_test = X_test[:, 1] if X_test.shape[1] > 1 else None
    water_full = X_full[:, 1] if X_full.shape[1] > 1 else None

    # 2. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_full_scaled = scaler.transform(X_full)

    # 自动保存 scaler，imputer 和 feature 列表用于预测
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(FEATURE_COLS, os.path.join(model_dir, 'selected_features.pkl'))
    joblib.dump(imputer, os.path.join(model_dir, 'imputer.pkl'))

    print(f"✅ 有效数据量: {len(mask_df)} 条 (训练集: {len(X_train)}, 测试集: {len(X_test)})。")
    print("原水量（Km³）描述统计：")
    print(df['原水量（Km³）'].describe())
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_full_scaled, y_full, full_dates, water_test, water_full


def evaluate_and_plot(y_test, y_pred, y_full, y_full_pred, full_dates, model_name, plot_dir="outputs", water_test=None,
                      water_full=None):
    """
    计算指标（仅依靠测试集），并根据 full_dates 将全量预测结果按年份分组并分别绘制实际-预测图。
    这样保证图表是逐天连续的。
    """
    import matplotlib.pyplot as plt
    # 设定中文字体，确保绘图时不乱码
    plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Songti SC', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    os.makedirs(plot_dir, exist_ok=True)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 {model_name} 测试集评估结果 (总投矾量):")
    print(f" -> MAE  (平均绝对误差): {mae:.2f} kg")
    print(f" -> RMSE (均方根误差):   {rmse:.2f} kg")
    print(f" -> R²   (决定系数):     {r2:.4f}")

    if water_test is not None:
        safe_water_test = np.where(water_test == 0, 1e-5, water_test)
        y_test_unit = y_test / safe_water_test
        y_pred_unit = y_pred / safe_water_test

        mae_unit = mean_absolute_error(y_test_unit, y_pred_unit)
        rmse_unit = np.sqrt(mean_squared_error(y_test_unit, y_pred_unit))
        r2_unit = r2_score(y_test_unit, y_pred_unit)

        print(f"\n📊 {model_name} 测试集评估结果 (投矾量/千吨水):")
        print(f" -> MAE  (平均绝对误差): {mae_unit:.2f} kg/千吨水")
        print(f" -> RMSE (均方根误差):   {rmse_unit:.2f} kg/千吨水")
        print(f" -> R²   (决定系数):     {r2_unit:.4f}")

    if full_dates is None:
        print("⚠️ 警告: 未找到日期数据，无法按年份连续绘图。")
        return

    results_df = pd.DataFrame({
        'Date': pd.to_datetime(full_dates),
        'Actual': y_full,
        'Predicted': y_full_pred
    })

    results_df.dropna(subset=['Date'], inplace=True)
    results_df['Year'] = results_df['Date'].dt.year
    years = results_df['Year'].unique()

    if water_full is not None:
        safe_water_full = np.where(water_full == 0, 1e-5, water_full)
        results_df['Actual_Unit'] = results_df['Actual'] / safe_water_full
        results_df['Predicted_Unit'] = results_df['Predicted'] / safe_water_full

    for year in sorted(years):
        # 取该年的全量并排序
        year_df = results_df[results_df['Year'] == year].sort_values(by='Date')
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 逐天绘制连续折线：总投矾量
        axes[0].plot(year_df['Date'], year_df['Actual'], label='实际总耗用量', marker='o', markersize=3, alpha=0.8)
        axes[0].plot(year_df['Date'], year_df['Predicted'], label=f'{model_name}预测总耗用量', marker='x', markersize=3, alpha=0.8)
        axes[0].set_title(f'{model_name} 预测总投矾量表现 - {year}年 (全景视角)')
        axes[0].set_ylabel('耗用矾量 (kg)')
        axes[0].legend()
        axes[0].grid(True)
        
        # 逐天绘制连续折线：单位投矾量
        if 'Actual_Unit' in year_df.columns:
            axes[1].plot(year_df['Date'], year_df['Actual_Unit'], label='实际投矾量/千吨水', marker='o', markersize=3, alpha=0.8, color='green')
            axes[1].plot(year_df['Date'], year_df['Predicted_Unit'], label=f'{model_name}预测投矾量/千吨水', marker='x', markersize=3, alpha=0.8, color='orange')
            axes[1].set_title(f'{model_name} 预测投矾量/千吨水表现 - {year}年 (全景视角)')
            axes[1].set_xlabel('日期')
            axes[1].set_ylabel('投矾量 (kg/Km³)')
            axes[1].legend()
            axes[1].grid(True)
            
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        
        plot_path = os.path.join(plot_dir, f'{model_name.lower().replace("-", "_")}_predict_{year}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ {year}年预测对比图已保存至: {plot_path}")


# ==================== ResNet (残差神经网络) 相关类定义 ====================
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """双层全连接残差模块：前向传播时每两层加一次残差连接，带 LayerNorm 和 GELU 激活"""
    def __init__(self, dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.gelu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.dropout2(out)
        return self.gelu(out + residual)

class AlumDosageResNet(nn.Module):
    def __init__(self, input_dim):
        super(AlumDosageResNet, self).__init__()
        
        # 1. 初始特征提取模块：拓宽宽度至 128，采用 LayerNorm + GELU
        self.init_block = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 2. 残差网络模块：4 个双层残差块 (共 8 层)
        self.res_blocks = nn.Sequential(
            ResidualBlock(128, dropout_rate=0.2),
            ResidualBlock(128, dropout_rate=0.2),
            ResidualBlock(128, dropout_rate=0.2),
            ResidualBlock(128, dropout_rate=0.2)
        )
        
        # 3. 输出回归模块
        self.out_block = nn.Sequential(
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out = self.init_block(x)
        out = self.res_blocks(out)
        out = self.out_block(out)
        return out

class SupervisedAlumDosageResNet(nn.Module):
    """双输出头监督残差网络：前半部对齐 XGBoost 基线，后半部使用残差网络微调修正预测真实投药量"""
    def __init__(self, input_dim):
        super(SupervisedAlumDosageResNet, self).__init__()
        
        # 1. 初始特征提取层（前几层）：维度设为 128，采用 LayerNorm + GELU
        self.init_block = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 辅助输出头：输出基准预测值（对齐并学习 XGBoost）
        self.aux_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # 2. 残差微调网络（后几层）：3 个双层残差块 (共 6 层)
        self.res_blocks = nn.Sequential(
            ResidualBlock(128, dropout_rate=0.2),
            ResidualBlock(128, dropout_rate=0.2),
            ResidualBlock(128, dropout_rate=0.2)
        )
        
        # 最终输出头：结合残差微调输出最终预测值（对齐真实投加量）
        self.out_block = nn.Sequential(
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        feat_base = self.init_block(x)
        y_base = self.aux_head(feat_base)
        
        feat_refined = self.res_blocks(feat_base)
        y_final = self.out_block(feat_refined)
        
        return y_final, y_base

class ResNetRegressor:
    """残差神经网络的 Scikit-Learn 兼容包装器，集成 5 折模型并在前向推断时进行平均"""
    def __init__(self, models_list, input_dim):
        import copy
        # 深度复制 5 折训练模型中的权重字典
        self.state_dicts = [copy.deepcopy(model.state_dict()) for model in models_list]
        self.input_dim = input_dim
        # 记录具体模型类名
        self.model_class_name = models_list[0].__class__.__name__
        
    def predict(self, X):
        import torch
        import numpy as np
        # 兼容输入类型为 list 或 numpy array
        X_arr = np.array(X, dtype=np.float32)
        X_tensor = torch.FloatTensor(X_arr)
        
        # 收集 5 个模型的预测输出
        all_preds = []
        for state_dict in self.state_dicts:
            if self.model_class_name == 'SupervisedAlumDosageResNet':
                model = SupervisedAlumDosageResNet(self.input_dim)
            else:
                model = AlumDosageResNet(self.input_dim)
                
            model.load_state_dict(state_dict)
            model.eval()
            with torch.no_grad():
                if self.model_class_name == 'SupervisedAlumDosageResNet':
                    pred = model(X_tensor)[0].cpu().numpy().flatten()
                else:
                    pred = model(X_tensor).cpu().numpy().flatten()
            all_preds.append(pred)
            
        # 5折结果取均值
        ensemble_pred = np.mean(all_preds, axis=0)
        return ensemble_pred

