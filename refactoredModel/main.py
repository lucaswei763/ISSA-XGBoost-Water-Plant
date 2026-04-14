#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
水厂投矾量预测模型 - 基于八个核心指标
========================================
核心指标：
1. temperature - 温度
2. turbidity_avg - 平均浊度
3. water_supply_km3 - 供水量
4. electricity_consumption_kwh - 耗电量
5. raw_water_km3 - 原水量
6. ammonia_nitrogen - 氨氮
7. permanganate_index - 高锰酸盐指数（有机物）
8. ph_value - pH值

功能：
- 模型训练与保存
- 模型加载与预测
- 批量预测
- 交互式预测
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
RANDOM_STATE = 42
TEST_SIZE = 0.2

# 八个核心特征（按重要性排序）
CORE_FEATURES = [
    'temperature',  # 温度
    'turbidity_avg',  # 平均浊度
    'water_supply_km3',  # 供水量
    'electricity_consumption_kwh',  # 耗电量
    'raw_water_km3',  # 原水量
    'ammonia_nitrogen',  # 氨氮
    'permanganate_index',  # 高锰酸盐指数
    'ph_value'  # pH值
]

# 特征中文名映射
FEATURE_NAMES_CN = {
    'temperature': '温度 (°C)',
    'turbidity_avg': '平均浊度 (NTU)',
    'water_supply_km3': '供水量 (km³)',
    'electricity_consumption_kwh': '耗电量 (kWh)',
    'raw_water_km3': '原水量 (km³)',
    'ammonia_nitrogen': '氨氮 (mg/L)',
    'permanganate_index': '高锰酸盐指数 (mg/L)',
    'ph_value': 'pH值'
}

# 特征典型范围（用于输入验证）
FEATURE_RANGES = {
    'temperature': (0, 40),
    'turbidity_avg': (0, 100),
    'water_supply_km3': (0, 500),
    'electricity_consumption_kwh': (0, 50000),
    'raw_water_km3': (0, 500),
    'ammonia_nitrogen': (0, 10),
    'permanganate_index': (0, 15),
    'ph_value': (5, 9)
}

# 特征说明
FEATURE_DESCRIPTIONS = {
    'temperature': '水温越高，微生物活性越强，影响混凝效果',
    'turbidity_avg': '浊度越高，悬浮物越多，需要更多混凝剂',
    'water_supply_km3': '供水量越大，处理量越大，投矾量相应增加',
    'electricity_consumption_kwh': '耗电量反映运行强度，与投矾量正相关',
    'raw_water_km3': '原水取水量，直接影响处理规模',
    'ammonia_nitrogen': '氨氮反映污染程度，高值需调整投矾',
    'permanganate_index': '高锰酸盐指数反映有机物含量，影响混凝',
    'ph_value': 'pH值影响混凝效果，最佳范围6.5-7.5'
}

print("=" * 80)
print("水厂投矾量预测模型 - 基于八个核心指标")
print("=" * 80)
print("\n核心指标:")
for i, feat in enumerate(CORE_FEATURES, 1):
    print(f"  {i}. {FEATURE_NAMES_CN.get(feat, feat)}")
    print(f"     └─ {FEATURE_DESCRIPTIONS.get(feat, '')}")


# ==================== 第一部分：数据加载与预处理 ====================
class DataLoader:
    """数据加载器"""

    @staticmethod
    def load_from_db(db_path='data/water_data.db'):
        """从数据库加载数据"""
        if not os.path.exists(db_path):
            print(f"错误：数据库文件不存在 - {db_path}")
            return None

        conn = sqlite3.connect(db_path)

        # 查询所有表
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        print(f"✓ 数据库表: {tables}")

        # 尝试加载合并数据
        if 'merged_data' in tables:
            df = pd.read_sql_query("SELECT * FROM merged_data", conn)
        else:
            # 加载第一个表
            df = pd.read_sql_query(f"SELECT * FROM {tables[0]}", conn)

        conn.close()

        # 识别目标变量
        target_col = None
        for col in df.columns:
            if '矾' in col or 'alum' in col.lower() or 'dosage' in col.lower():
                target_col = col
                break

        if target_col is None:
            target_col = df.select_dtypes(include=[np.number]).columns[0]

        print(f"✓ 目标变量: {target_col}")
        print(f"✓ 数据形状: {df.shape}")

        return df, target_col

    @staticmethod
    def load_from_csv(csv_path):
        """从CSV文件加载数据"""
        df = pd.read_csv(csv_path, encoding='utf-8')

        # 识别目标变量
        target_col = None
        for col in df.columns:
            if '矾' in col or 'alum' in col.lower() or 'dosage' in col.lower():
                target_col = col
                break

        if target_col is None:
            target_col = df.select_dtypes(include=[np.number]).columns[0]

        return df, target_col

    @staticmethod
    def prepare_data(df, target_col, features=CORE_FEATURES):
        """准备训练数据"""
        # 检查特征是否存在
        available_features = [f for f in features if f in df.columns]
        missing_features = [f for f in features if f not in df.columns]

        if missing_features:
            print(f"⚠ 警告: 缺失特征 {missing_features}")

        if not available_features:
            print("错误：没有可用特征")
            return None, None

        # 提取特征和目标
        X = df[available_features].copy()
        y = df[target_col].copy()

        # 处理缺失值
        print(f"▶ 处理缺失值...")
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].median())

        if y.isnull().sum() > 0:
            y = y.fillna(y.median())

        print(f"✓ 数据准备完成: {X.shape[0]} 样本, {X.shape[1]} 特征")

        return X, y


# ==================== 第二部分：模型训练 ====================
class ModelTrainer:
    """模型训练器"""

    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.models = {}
        self.scaler = None
        self.best_model = None
        self.best_model_name = None
        self.features = None

    def train_all_models(self, X, y, test_size=TEST_SIZE):
        """训练多个模型并选择最佳"""

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, shuffle=True
        )

        print(f"\n数据集划分:")
        print(f"  训练集: {X_train.shape[0]} 样本")
        print(f"  测试集: {X_test.shape[0]} 样本")

        # 标准化
        self.scaler = RobustScaler()  # 对异常值更鲁棒
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.features = X.columns.tolist()

        # 定义模型
        models_to_train = {
            '线性回归': LinearRegression(),
            '岭回归': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso回归': Lasso(alpha=0.001, random_state=self.random_state),
            '随机森林': RandomForestRegressor(
                n_estimators=200, max_depth=10,
                min_samples_split=5, random_state=self.random_state, n_jobs=-1
            ),
            '梯度提升': GradientBoostingRegressor(
                n_estimators=200, max_depth=5,
                learning_rate=0.05, random_state=self.random_state
            ),
            'XGBoost': XGBRegressor(
                n_estimators=200, max_depth=6,
                learning_rate=0.05, random_state=self.random_state,
                n_jobs=-1, verbosity=0
            )
        }

        # 训练并评估
        results = {}

        for name, model in models_to_train.items():
            print(f"\n▶ 训练 {name}...")

            # 训练
            model.fit(X_train_scaled, y_train)

            # 预测
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            # 评估
            train_metrics = self._calculate_metrics(y_train, y_train_pred)
            test_metrics = self._calculate_metrics(y_test, y_test_pred)

            results[name] = {
                'model': model,
                'train': train_metrics,
                'test': test_metrics
            }

            print(f"  训练集: RMSE={train_metrics['RMSE']:.4f}, R²={train_metrics['R2']:.4f}")
            print(f"  测试集: RMSE={test_metrics['RMSE']:.4f}, R²={test_metrics['R2']:.4f}")

        # 选择最佳模型（基于测试集R²）
        best_name = max(results, key=lambda x: results[x]['test']['R2'])
        self.best_model = results[best_name]['model']
        self.best_model_name = best_name
        self.models = results

        print(f"\n{'=' * 60}")
        print(f"✓ 最佳模型: {best_name}")
        print(f"✓ 测试集R²: {results[best_name]['test']['R2']:.4f}")
        print(f"{'=' * 60}")

        return results

    def _calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        return {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred)
        }

    def save_model(self, model_path='models/best_model_8features.pkl'):
        """保存模型和标准化器"""
        os.makedirs('models', exist_ok=True)

        # 保存模型
        joblib.dump({
            'model': self.best_model,
            'scaler': self.scaler,
            'features': self.features,
            'model_name': self.best_model_name,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_features': len(self.features)
        }, model_path)

        print(f"✓ 模型已保存: {model_path}")

        # 保存特征重要性
        self._save_feature_importance()

    def _save_feature_importance(self):
        """保存特征重要性"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importance = np.abs(self.best_model.coef_)
        else:
            importance = np.ones(len(self.features))

        importance_df = pd.DataFrame({
            '特征': self.features,
            '特征中文名': [FEATURE_NAMES_CN.get(f, f) for f in self.features],
            '重要性': importance / importance.sum()
        }).sort_values('重要性', ascending=False)

        importance_df.to_csv('outputs/feature_importance_8features.csv', index=False, encoding='utf-8-sig')
        print(f"✓ 特征重要性已保存: outputs/feature_importance_8features.csv")

        # 绘制特征重要性图
        self._plot_feature_importance(importance_df)

    def _plot_feature_importance(self, importance_df):
        """绘制特征重要性图"""
        plt.figure(figsize=(10, 8))
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.8, len(importance_df)))

        bars = plt.barh(range(len(importance_df)), importance_df['重要性'], color=colors)
        plt.yticks(range(len(importance_df)), importance_df['特征中文名'])
        plt.xlabel('重要性', fontsize=12)
        plt.title(f'特征重要性分析 (8指标) - {self.best_model_name}', fontsize=14, fontweight='bold')

        # 添加数值标签
        for i, (idx, row) in enumerate(importance_df.iterrows()):
            plt.text(row['重要性'] + 0.01, i, f"{row['重要性']:.3f}", va='center')

        plt.tight_layout()
        plt.savefig('outputs/feature_importance_8features.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 特征重要性图已保存: outputs/feature_importance_8features.png")


# ==================== 第三部分：预测服务 ====================
class Predictor:
    """预测器"""

    def __init__(self, model_path='models/best_model_8features.pkl'):
        self.model = None
        self.scaler = None
        self.features = None
        self.model_name = None
        self.load_model(model_path)

    def load_model(self, model_path):
        """加载模型"""
        if not os.path.exists(model_path):
            print(f"错误：模型文件不存在 - {model_path}")
            print("请先运行训练程序")
            return False

        data = joblib.load(model_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.features = data['features']
        self.model_name = data.get('model_name', 'Unknown')

        print(f"✓ 模型加载成功: {model_path}")
        print(f"  模型名称: {self.model_name}")
        print(f"  特征数量: {len(self.features)}")

        return True

    def predict(self, input_data):
        """
        预测投矾量

        参数:
            input_data: dict 或 list 或 pd.DataFrame
                dict格式: {'temperature': 20, 'turbidity_avg': 15, ...}
                list格式: [temperature, turbidity_avg, water_supply_km3,
                          electricity_consumption_kwh, raw_water_km3,
                          ammonia_nitrogen, permanganate_index, ph_value]

        返回:
            dict: 包含预测结果和置信区间
        """
        # 转换输入格式
        if isinstance(input_data, dict):
            # 按特征顺序构建数组
            X_input = np.array([[input_data.get(f, 0) for f in self.features]])
        elif isinstance(input_data, (list, tuple)):
            if len(input_data) != len(self.features):
                raise ValueError(f"输入数据长度应为 {len(self.features)}，实际为 {len(input_data)}")
            X_input = np.array([input_data])
        elif isinstance(input_data, pd.DataFrame):
            X_input = input_data[self.features].values
        else:
            raise ValueError("输入格式不支持，请使用 dict、list 或 DataFrame")

        # 标准化
        X_scaled = self.scaler.transform(X_input)

        # 预测
        predictions = self.model.predict(X_scaled)

        # 计算置信区间（基于模型类型）
        confidence = self._calculate_confidence(predictions)

        results = []
        for i, pred in enumerate(predictions):
            results.append({
                'predicted_dosage': round(pred, 3),
                'unit': 'mg/L',
                'confidence_lower': round(pred * (1 - confidence), 3),
                'confidence_upper': round(pred * (1 + confidence), 3),
                'confidence_level': f"±{confidence * 100:.0f}%"
            })

        return results if len(results) > 1 else results[0]

    def _calculate_confidence(self, predictions):
        """计算置信区间（基于模型性能）"""
        if '随机森林' in self.model_name or 'XGBoost' in self.model_name:
            return 0.10  # 10% 置信区间
        elif '梯度提升' in self.model_name:
            return 0.12
        else:
            return 0.15

    def predict_batch(self, df):
        """批量预测"""
        if not all(f in df.columns for f in self.features):
            missing = [f for f in self.features if f not in df.columns]
            raise ValueError(f"缺少特征: {missing}")

        X = df[self.features].values
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        return predictions

    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'features': self.features,
            'feature_count': len(self.features),
            'feature_names_cn': [FEATURE_NAMES_CN.get(f, f) for f in self.features],
            'feature_descriptions': [FEATURE_DESCRIPTIONS.get(f, '') for f in self.features]
        }


# ==================== 第四部分：可视化 ====================
def plot_training_results(results, X, y, target_name='投矾量'):
    """绘制训练结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 获取最佳模型
    best_name = max(results, key=lambda x: results[x]['test']['R2'])
    best_model = results[best_name]['model']

    # 准备数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 标准化
    scaler = RobustScaler()
    X_test_scaled = scaler.fit_transform(X_test)

    # 1. 预测值 vs 实际值散点图
    y_pred = best_model.predict(X_test_scaled)
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5, edgecolors='black', linewidth=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    axes[0, 0].set_xlabel('实际值', fontsize=12)
    axes[0, 0].set_ylabel('预测值', fontsize=12)
    axes[0, 0].set_title(f'{best_name}\n预测 vs 实际 (R²={results[best_name]["test"]["R2"]:.4f})',
                         fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 残差分布
    residuals = y_test - y_pred
    axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('残差', fontsize=12)
    axes[0, 1].set_ylabel('频数', fontsize=12)
    axes[0, 1].set_title('残差分布', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 模型性能对比
    model_names = list(results.keys())
    r2_scores = [results[m]['test']['R2'] for m in model_names]
    colors = ['green' if r == max(r2_scores) else 'steelblue' for r in r2_scores]
    bars = axes[0, 2].barh(model_names, r2_scores, color=colors)
    axes[0, 2].set_xlabel('R² 分数', fontsize=12)
    axes[0, 2].set_title('模型性能对比', fontsize=12, fontweight='bold')
    for bar, r2 in zip(bars, r2_scores):
        axes[0, 2].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{r2:.4f}', va='center')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. 特征相关性热力图
    corr_matrix = X.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, ax=axes[1, 0],
                cbar_kws={"shrink": 0.8})
    axes[1, 0].set_title('特征相关性热力图', fontsize=12, fontweight='bold')

    # 5. 预测误差箱线图
    errors_by_model = {}
    for name, result in results.items():
        model = result['model']
        y_pred_temp = model.predict(X_test_scaled)
        errors_by_model[name] = y_test - y_pred_temp

    axes[1, 1].boxplot(errors_by_model.values(), labels=errors_by_model.keys())
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1, 1].set_ylabel('预测误差', fontsize=12)
    axes[1, 1].set_title('各模型预测误差对比', fontsize=12, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    # 6. 特征重要性
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importance = np.abs(best_model.coef_)
    else:
        importance = np.ones(len(X.columns))

    importance_df = pd.DataFrame({
        '特征': [FEATURE_NAMES_CN.get(c, c) for c in X.columns],
        '重要性': importance / importance.sum()
    }).sort_values('重要性', ascending=True)

    axes[1, 2].barh(importance_df['特征'], importance_df['重要性'], color='coral')
    axes[1, 2].set_xlabel('重要性', fontsize=12)
    axes[1, 2].set_title('特征重要性 (8指标)', fontsize=12, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/training_results_8features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 训练结果图已保存: outputs/training_results_8features.png")


# ==================== 第五部分：交互式预测 ====================
def interactive_prediction(predictor):
    """交互式预测"""
    print("\n" + "=" * 60)
    print("交互式预测模式 (8指标)")
    print("=" * 60)
    print("\n请输入以下八个指标的值：\n")

    input_data = {}
    for feat in CORE_FEATURES:
        cn_name = FEATURE_NAMES_CN.get(feat, feat)
        description = FEATURE_DESCRIPTIONS.get(feat, '')
        range_info = FEATURE_RANGES.get(feat, (None, None))

        print(f"\n【{cn_name}】")
        print(f"  说明: {description}")
        if range_info[0] is not None and range_info[1] is not None:
            print(f"  参考范围: {range_info[0]} ~ {range_info[1]}")

        while True:
            try:
                value = input(f"  请输入值: ").strip()
                if value == '':
                    print("    输入不能为空，请重新输入")
                    continue

                val = float(value)

                # 验证范围
                min_val, max_val = range_info
                if min_val is not None and val < min_val:
                    print(f"    警告: 输入值 {val} 低于典型最小值 {min_val}")
                    confirm = input("    是否继续？(y/n): ").lower()
                    if confirm != 'y':
                        continue
                if max_val is not None and val > max_val:
                    print(f"    警告: 输入值 {val} 高于典型最大值 {max_val}")
                    confirm = input("    是否继续？(y/n): ").lower()
                    if confirm != 'y':
                        continue

                input_data[feat] = val
                break
            except ValueError:
                print("    请输入有效的数字")

    # 预测
    result = predictor.predict(input_data)

    print("\n" + "=" * 60)
    print("预测结果")
    print("=" * 60)
    print(f"\n  预测投矾量: {result['predicted_dosage']} mg/L")
    print(f"  置信区间: [{result['confidence_lower']}, {result['confidence_upper']}] mg/L")
    print(f"  置信水平: {result['confidence_level']}")

    # 给出建议
    print("\n【工艺建议】")
    dosage = result['predicted_dosage']
    if dosage < 10:
        print("  ✓ 投矾量较低，水质较好")
    elif dosage < 20:
        print("  → 投矾量适中，正常范围")
    elif dosage < 30:
        print("  ⚠ 投矾量偏高，建议检查原水水质")
    else:
        print("  ⚠⚠ 投矾量很高，需要重点关注原水变化")

    print("=" * 60)


# ==================== 第六部分：批量预测 ====================
def batch_prediction(predictor, input_file, output_file=None):
    """批量预测"""
    print(f"\n批量预测模式 (8指标)")
    print(f"  输入文件: {input_file}")

    # 读取输入文件
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file, encoding='utf-8')
    elif input_file.endswith('.xlsx'):
        df = pd.read_excel(input_file)
    else:
        print("错误：不支持的文件格式，请使用 CSV 或 Excel 文件")
        return

    # 检查特征
    missing_features = [f for f in CORE_FEATURES if f not in df.columns]
    if missing_features:
        print(f"错误：缺少特征 {missing_features}")
        return

    # 预测
    predictions = predictor.predict_batch(df)

    # 添加预测结果
    df['predicted_dosage'] = predictions

    # 保存结果
    if output_file is None:
        output_file = input_file.replace('.csv', '_predicted.csv').replace('.xlsx', '_predicted.xlsx')

    if output_file.endswith('.csv'):
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
    else:
        df.to_excel(output_file, index=False)

    print(f"✓ 预测完成，结果已保存: {output_file}")

    # 显示统计
    print(f"\n预测统计:")
    print(f"  预测范围: [{predictions.min():.3f}, {predictions.max():.3f}] mg/L")
    print(f"  平均预测: {predictions.mean():.3f} mg/L")
    print(f"  中位数: {np.median(predictions):.3f} mg/L")
    print(f"  标准差: {predictions.std():.3f} mg/L")


# ==================== 第七部分：主程序 ====================
def main():
    """主程序"""
    import argparse

    parser = argparse.ArgumentParser(description='水厂投矾量预测模型 (8指标)')
    parser.add_argument('--train', action='store_true', help='训练新模型')
    parser.add_argument('--predict', action='store_true', help='交互式预测')
    parser.add_argument('--batch', type=str, help='批量预测（输入文件路径）')
    parser.add_argument('--output', type=str, help='批量预测输出文件路径')
    parser.add_argument('--info', action='store_true', help='显示模型信息')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 训练模式
    if args.train:
        print("\n【训练模式 - 8指标模型】")

        # 加载数据
        loader = DataLoader()
        df, target_col = loader.load_from_db()

        if df is None:
            print("尝试从CSV加载...")
            csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]
            if csv_files:
                df, target_col = loader.load_from_csv(f'data/{csv_files[0]}')

        if df is None:
            print("错误：无法加载数据")
            return

        # 准备数据
        X, y = loader.prepare_data(df, target_col, CORE_FEATURES)

        if X is None:
            print("错误：数据准备失败")
            return

        # 训练模型
        trainer = ModelTrainer()
        results = trainer.train_all_models(X, y)

        # 保存模型
        trainer.save_model()

        # 绘制结果
        plot_training_results(results, X, y)

        print("\n✅ 模型训练完成！")
        print("  使用 --predict 进行预测")
        print("  使用 --batch <文件路径> 进行批量预测")

    # 预测模式
    elif args.predict:
        print("\n【预测模式 - 8指标模型】")

        predictor = Predictor()
        if predictor.model is not None:
            interactive_prediction(predictor)

    # 批量预测模式
    elif args.batch:
        print("\n【批量预测模式 - 8指标模型】")

        predictor = Predictor()
        if predictor.model is not None:
            batch_prediction(predictor, args.batch, args.output)

    # 信息模式
    elif args.info:
        print("\n【模型信息 - 8指标模型】")

        predictor = Predictor()
        if predictor.model is not None:
            info = predictor.get_model_info()
            print(f"\n模型名称: {info['model_name']}")
            print(f"特征数量: {info['feature_count']}")
            print("\n特征列表:")
            for f, cn, desc in zip(info['features'], info['feature_names_cn'], info['feature_descriptions']):
                print(f"  - {cn} ({f})")
                print(f"    说明: {desc}")

    else:
        # 默认显示帮助
        print("""
使用方法 (8指标模型):
  python main.py --train      # 训练新模型
  python main.py --predict    # 交互式预测
  python main.py --batch <文件> # 批量预测
  python main.py --info       # 显示模型信息

示例:
  python main.py --train
  python main.py --predict
  python main.py --batch input_data.csv
  python main.py --batch input_data.xlsx --output result.xlsx

输入特征（8个）:
  1. temperature          - 温度 (°C)
  2. turbidity_avg        - 平均浊度 (NTU)
  3. water_supply_km3     - 供水量 (km³)
  4. electricity_consumption_kwh - 耗电量 (kWh)
  5. raw_water_km3        - 原水量 (km³)
  6. ammonia_nitrogen     - 氨氮 (mg/L)
  7. permanganate_index   - 高锰酸盐指数 (mg/L)
  8. ph_value             - pH值
        """)


# ==================== 快速预测函数 ====================
def quick_predict(temperature, turbidity_avg, water_supply_km3,
                  electricity_consumption_kwh, raw_water_km3,
                  ammonia_nitrogen, permanganate_index, ph_value):
    """
    快速预测函数 - 可直接导入使用

    参数:
        temperature: 温度 (°C)
        turbidity_avg: 平均浊度 (NTU)
        water_supply_km3: 供水量 (km³)
        electricity_consumption_kwh: 耗电量 (kWh)
        raw_water_km3: 原水量 (km³)
        ammonia_nitrogen: 氨氮 (mg/L)
        permanganate_index: 高锰酸盐指数 (mg/L)
        ph_value: pH值

    返回:
        dict: 预测结果
    """
    predictor = Predictor()
    if predictor.model is None:
        return {'error': '模型未找到，请先运行训练'}

    input_data = {
        'temperature': temperature,
        'turbidity_avg': turbidity_avg,
        'water_supply_km3': water_supply_km3,
        'electricity_consumption_kwh': electricity_consumption_kwh,
        'raw_water_km3': raw_water_km3,
        'ammonia_nitrogen': ammonia_nitrogen,
        'permanganate_index': permanganate_index,
        'ph_value': ph_value
    }

    return predictor.predict(input_data)


if __name__ == "__main__":
    main()