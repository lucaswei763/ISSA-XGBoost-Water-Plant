# predictor_service.py - 集成 LSTM+XGBoost 堆叠模型
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
水厂投矾量预测服务 - 集成 LSTM+XGBoost 堆叠模型
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入堆叠模型
try:
    from lstm_xgb_stacked_v2 import LSTMXGBStacked, WaterDataLoaderV4

    STACKED_AVAILABLE = True
except ImportError:
    STACKED_AVAILABLE = False
    logger.warning("堆叠模型未找到，使用原始模型")


class WaterPredictor:
    """水厂投矾量预测服务"""

    def __init__(self, model_dir='models', db_path='data/water_data.db', use_cache=True):
        """
        初始化预测器
        Args:
            model_dir: 模型目录
            db_path: 数据库路径
            use_cache: 是否使用缓存
        """
        import sys
        import os
        if hasattr(sys, '_MEIPASS'):
            if model_dir == 'models':
                model_dir = os.path.join(sys._MEIPASS, 'models')
            if db_path == 'data/water_data.db':
                db_path = os.path.join(sys._MEIPASS, 'data/water_data.db')
        else:
            model_dir = os.path.abspath(model_dir)
            db_path = os.path.abspath(db_path)

        self.model_dir = model_dir
        self.db_path = db_path
        self.use_cache = use_cache
        self.cache = {}

        # 加载模型
        self._load_models()

    def _load_models(self):
        """加载所有模型"""
        # 1. 先尝试加载堆叠模型
        self.use_stacked = False
        self.stacked_model = None

        if STACKED_AVAILABLE:
            stacked_path = os.path.join(self.model_dir, 'lstm_xgb_stacked')
            if os.path.exists(f'{stacked_path}_config.json'):
                try:
                    logger.info("正在加载 LSTM+XGBoost 堆叠模型...")
                    self.stacked_model = LSTMXGBStacked().load(stacked_path)
                    self.use_stacked = True
                    self.model_type = "LSTM+XGBoost"
                    self.version = "2.0"
                    self.features = self.stacked_model.feature_names
                    self.seq_length = self.stacked_model.seq_length
                    self.target_col = "耗用矾量_kg"
                    logger.info("✅ LSTM+XGBoost 堆叠模型加载成功")
                    return
                except Exception as e:
                    logger.warning(f"堆叠模型加载失败: {e}，回退到原始模型")

        # 2. 回退到原始 XGBoost 模型
        self._load_xgboost_model()

    def _load_xgboost_model(self):
        """加载原始 XGBoost 模型"""
        try:
            self.model = joblib.load(os.path.join(self.model_dir, 'best_model.pkl'))
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
            self.imputer = joblib.load(os.path.join(self.model_dir, 'imputer.pkl'))
            self.features = joblib.load(os.path.join(self.model_dir, 'selected_features.pkl'))

            # 加载元数据
            metadata_path = os.path.join(self.model_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                    self.target_col = meta.get('target_col', '耗用矾量（kg）')
                    self.model_type = meta.get('model_type', 'XGBoost')
                    self.version = meta.get('version', '1.0')
            else:
                self.target_col = '耗用矾量（kg）'
                self.model_type = 'XGBoost'
                self.version = '1.0'

            self.use_stacked = False
            logger.info(f"✅ 原始 XGBoost 模型加载成功")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.model = None
            raise

    def _get_history_data(self, date_str, days=7):
        """获取历史数据"""
        try:
            conn = sqlite3.connect(self.db_path)
            # 解析日期
            target_date = pd.to_datetime(date_str)
            start_date = target_date - timedelta(days=days + 1)

            # 查询历史数据
            query = """
                SELECT * FROM water_records 
                WHERE 日期 >= ? AND 日期 < ?
                ORDER BY 日期 ASC
            """
            df = pd.read_sql(query, conn, params=(start_date.strftime('%Y-%m-%d'), target_date.strftime('%Y-%m-%d')))
            conn.close()

            if len(df) >= days:
                return df
            return None
        except:
            return None

    def _predict_stacked(self, input_data):
        """使用堆叠模型预测"""
        try:
            # 检查历史数据
            date_str = input_data.get('日期')
            history_df = None

            if date_str:
                history_df = self._get_history_data(date_str, self.seq_length)

            # 提取特征值
            feature_values = []
            for f in self.features:
                val = input_data.get(f)
                if val is None:
                    # 尝试匹配
                    for key, value in input_data.items():
                        if f.replace('_', '') in key.replace('_', '') or key.replace('_', '') in f.replace('_', ''):
                            val = float(value)
                            break
                if val is None:
                    val = 0.0
                feature_values.append(float(val))

            # 构建时序序列
            seq_len = self.seq_length
            if history_df is not None and len(history_df) >= seq_len:
                # 使用历史数据
                hist_data = history_df.tail(seq_len)[self.features].values
                X_seq = np.array([hist_data])
            else:
                # 使用当前值填充
                X_seq = np.array([feature_values] * seq_len).reshape(1, seq_len, -1)

            X_current = np.array([feature_values])

            # 标准化
            lstm = self.stacked_model.lstm_trainer
            X_seq_scaled = lstm.scaler_X.transform(
                X_seq.reshape(-1, len(self.features))
            ).reshape(1, seq_len, len(self.features))
            X_current_scaled = lstm.scaler_X.transform(X_current)

            # 预测
            pred = self.stacked_model.predict(X_seq_scaled, X_current_scaled)

            # 检查输入范围（简化版）
            warnings = []
            for i, f in enumerate(self.features):
                if feature_values[i] < -100 or feature_values[i] > 10000:
                    warnings.append(f"{f} 值异常: {feature_values[i]}")

            return float(pred[0]), warnings

        except Exception as e:
            logger.error(f"堆叠模型预测失败: {e}")
            # 降级到原始模型
            return self._predict_xgboost(input_data)

    def _predict_xgboost(self, input_data):
        """使用原始 XGBoost 模型预测"""
        try:
            # 构建输入向量
            input_vector = []
            for f in self.features:
                val = input_data.get(f)
                if val is None:
                    # 尝试匹配
                    for key, value in input_data.items():
                        if f in key or key in f:
                            val = float(value)
                            break
                if val is None:
                    val = 0.0
                input_vector.append(float(val))

            # 转换为 DataFrame 进行插值
            input_df = pd.DataFrame([input_vector], columns=self.features)
            input_imputed = self.imputer.transform(input_df)
            input_scaled = self.scaler.transform(input_imputed)

            # 预测
            pred = self.model.predict(input_scaled)[0]

            # 检查输入范围
            warnings = []
            for i, f in enumerate(self.features):
                if input_vector[i] < -100 or input_vector[i] > 10000:
                    warnings.append(f"{f} 值异常: {input_vector[i]}")

            return float(pred), warnings

        except Exception as e:
            logger.error(f"XGBoost 预测失败: {e}")
            raise

    def predict(self, input_data):
        """
        预测投矾量
        Args:
            input_data: 输入数据字典
        Returns:
            (预测值, 警告列表)
        """
        # 优先使用堆叠模型
        if self.use_stacked and self.stacked_model is not None:
            try:
                return self._predict_stacked(input_data)
            except Exception as e:
                logger.warning(f"堆叠模型预测失败，降级到原始模型: {e}")
                return self._predict_xgboost(input_data)
        else:
            return self._predict_xgboost(input_data)

    def predict_batch(self, df, parallel=False):
        """批量预测"""
        results = []
        for idx, row in df.iterrows():
            input_dict = row.to_dict()
            try:
                pred, warnings = self.predict(input_dict)
                results.append(pred)
            except:
                results.append(None)

        df['预测投矾量_kg'] = results
        return df

    def predict_with_interval(self, input_data, alpha=0.05):
        """预测并返回置信区间（仅支持原始模型）"""
        if self.use_stacked:
            pred, _ = self._predict_stacked(input_data)
            # 堆叠模型暂不支持置信区间
            return pred, pred * 0.9, pred * 1.1

        # 原始模型支持置信区间
        pred, _ = self._predict_xgboost(input_data)
        # 简单置信区间
        return pred, pred * (1 - alpha), pred * (1 + alpha)

    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_type': self.model_type,
            'version': self.version,
            'features': self.features,
            'target_col': self.target_col,
            'use_stacked': self.use_stacked,
            'seq_length': getattr(self, 'seq_length', None),
        }

    def close(self):
        """关闭资源"""
        pass


# ==================== 测试 ====================
if __name__ == "__main__":
    # 测试预测器
    print("=" * 50)
    print("测试 WaterPredictor")
    print("=" * 50)

    try:
        predictor = WaterPredictor()
        print(f"✅ 模型加载成功")
        print(f"   类型: {predictor.model_type}")
        print(f"   版本: {predictor.version}")
        print(f"   使用堆叠: {predictor.use_stacked}")
        print(f"   特征数: {len(predictor.features)}")

        # 测试预测
        test_data = {
            '日期': '2026-01-15',
            '浑浊度_0点': 1.0,
            '原水量_Km3': 200,
            '温度_C': 15,
            '氨氮_mg_per_L': 0.1,
            'pH值': 7.2,
            '库区水位_m': 150,
            '消耗电_kWh': 45000,
        }

        pred, warnings = predictor.predict(test_data)
        print(f"\n📊 测试预测:")
        print(f"   输入: {test_data}")
        print(f"   预测投矾量: {pred:.2f} kg")
        if warnings:
            print(f"   警告: {warnings}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")