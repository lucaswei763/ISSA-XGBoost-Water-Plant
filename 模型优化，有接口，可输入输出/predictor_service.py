#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
水厂投矾量预测服务模块（最终稳定版）
- 支持随机森林和 XGBoost 模型
- 元数据管理、特征工程、置信区间、批量预测
- 数据库连接池（SQLAlchemy，兼容 SQLite）
- 线程安全、缓存修复（字典缓存）
"""

import os
import sys
import json
import hashlib
import logging
import sqlite3
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Union, Optional, Tuple
from contextlib import contextmanager

# 可选导入
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import QueuePool
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

logger = logging.getLogger(__name__)

# ==================== 配置 ====================
MODEL_DIR = os.getenv('MODEL_DIR', 'models')
DB_PATH = os.getenv('DB_PATH', 'data/water_data.db')
USE_CACHE = os.getenv('USE_CACHE', 'true').lower() == 'true'
CACHE_SIZE = int(os.getenv('CACHE_SIZE', 1024))
DEFAULT_CHUNK_SIZE = int(os.getenv('DEFAULT_CHUNK_SIZE', 5000))
PARALLEL_BATCH = os.getenv('PARALLEL_BATCH', 'false').lower() == 'true'
NUM_WORKERS = int(os.getenv('NUM_WORKERS', 4))

# 默认特征范围
DEFAULT_FEATURE_RANGES = {
    '浊度': (0, 100),
    '流量': (0, 200000),
    '耗氧量': (0, 10),
    'pH': (0, 14),
    '温度': (-10, 50),
    '氨氮': (0, 5),
    '锰': (0, 1),
    '铁': (0, 2),
    '亚硝酸盐氮': (0, 1),
    '库区水位': (0, 100)
}

FIELD_MAPPING = {
    'turbidity': '浊度',
    'flow': '流量',
    'codmn': '耗氧量',
    'ph': 'pH',
    'temperature': '温度',
    'ammonia': '氨氮',
    'manganese': '锰',
    'iron': '铁',
    'nitrite': '亚硝酸盐氮',
    'water_level': '库区水位'
}


class WaterPredictor:
    """水厂投矾量预测器（最终稳定版）"""

    def __init__(self, model_dir=MODEL_DIR, db_path=DB_PATH, use_cache=USE_CACHE):
        self.model_dir = self._resolve_path(model_dir)
        self.db_path = self._resolve_path(db_path)
        self.use_cache = use_cache
        self.model = None
        self.scaler = None
        self.features = None
        self.target_col = None
        self.model_type = None
        self.version = None
        self.feature_ranges = {}
        self.training_stats = {}
        self.engine = None
        self.Session = None
        self._cache = {}  # 手动缓存字典
        self._load_components()
        self._init_db()
        logger.info(f"预测器初始化完成: {self.model_type}, 版本={self.version}")

    def _resolve_path(self, path):
        if getattr(sys, 'frozen', False):
            base = sys._MEIPASS
        else:
            base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, path)

    def _load_components(self):
        """加载模型及元数据"""
        metadata_path = os.path.join(self.model_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            self.model_type = metadata.get('model_type', 'RandomForest')
            self.version = metadata.get('version', '1.0')
            self.features = metadata.get('features')
            self.target_col = metadata.get('target_col')
            self.feature_ranges = metadata.get('feature_ranges', DEFAULT_FEATURE_RANGES)
            self.training_stats = metadata.get('training_stats', {})
            # 加载标度器
            scaler_path = metadata.get('scaler_path', 'scaler.pkl')
            if scaler_path and os.path.exists(os.path.join(self.model_dir, scaler_path)):
                self.scaler = joblib.load(os.path.join(self.model_dir, scaler_path))
            # 加载主模型
            model_file = metadata.get('model_file', 'best_model.pkl')
            model_path = os.path.join(self.model_dir, model_file)
            if self.model_type == 'XGBoost':
                import xgboost as xgb
                self.model = xgb.Booster()
                self.model.load_model(model_path)
            else:
                self.model = joblib.load(model_path)
        else:
            # 兼容旧版
            logger.warning("未找到 metadata.json，尝试加载旧版模型文件")
            self.model = joblib.load(os.path.join(self.model_dir, 'best_model.pkl'))
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
            self.features = joblib.load(os.path.join(self.model_dir, 'selected_features.pkl'))
            self.model_type = 'RandomForest'
            self.version = '1.0'
            self.feature_ranges = DEFAULT_FEATURE_RANGES
        logger.info(f"模型加载成功: {self.model_type}, 特征数={len(self.features)}")

    def _init_db(self):
        """初始化数据库连接（兼容 SQLite）"""
        if not os.path.exists(self.db_path):
            logger.warning(f"数据库文件不存在: {self.db_path}，滞后特征将使用默认值")
            return
        if HAS_SQLALCHEMY:
            try:
                self.engine = create_engine(
                    f'sqlite:///{self.db_path}',
                    connect_args={'check_same_thread': False}
                )
                self.Session = sessionmaker(bind=self.engine)
                logger.info("数据库连接池初始化成功")
                return
            except Exception as e:
                logger.warning(f"SQLAlchemy 连接失败: {e}，降级使用 sqlite3")
        # 降级使用 sqlite3
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            logger.info("使用 sqlite3 直连数据库")
        except Exception as e:
            logger.warning(f"数据库连接失败: {e}，滞后特征将使用默认值")

    def _get_db_session(self):
        if hasattr(self, 'Session') and self.Session:
            return self.Session()
        elif hasattr(self, 'conn'):
            return self.conn
        return None

    def _get_default_value(self, feature: str) -> float:
        if feature in self.feature_ranges:
            low, high = self.feature_ranges[feature]
            return (low + high) / 2.0
        return 0.0

    def _get_lag_value(self, feature: str) -> float:
        session = self._get_db_session()
        if session is None:
            return self._get_default_value(feature)
        try:
            if HAS_SQLALCHEMY and hasattr(session, 'execute'):
                result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                tables = [row[0] for row in result.fetchall()]
            else:
                cursor = session.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]

            table_name = None
            for tbl in ['merged_data', 'water_quality', 'quality', 'original_data']:
                if tbl in tables:
                    table_name = tbl
                    break
            if table_name is None and tables:
                table_name = tables[0]
            if not table_name:
                return self._get_default_value(feature)

            if HAS_SQLALCHEMY and hasattr(session, 'execute'):
                col_info = session.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
                columns = [row[1] for row in col_info]
                matched_col = None
                for col in columns:
                    if feature.lower() in col.lower() or col.lower() in feature.lower():
                        matched_col = col
                        break
                if not matched_col:
                    return self._get_default_value(feature)
                date_col = None
                for col in columns:
                    if '日期' in col or 'date' in col.lower():
                        date_col = col
                        break
                if date_col:
                    query = f"SELECT {matched_col} FROM {table_name} ORDER BY {date_col} DESC LIMIT 1"
                else:
                    query = f"SELECT {matched_col} FROM {table_name} LIMIT 1"
                result = session.execute(text(query)).fetchone()
                if result and result[0] is not None:
                    return float(result[0])
            else:
                cursor = session.cursor()
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [row[1] for row in cursor.fetchall()]
                matched_col = None
                for col in columns:
                    if feature.lower() in col.lower() or col.lower() in feature.lower():
                        matched_col = col
                        break
                if not matched_col:
                    return self._get_default_value(feature)
                date_col = None
                for col in columns:
                    if '日期' in col or 'date' in col.lower():
                        date_col = col
                        break
                if date_col:
                    query = f"SELECT {matched_col} FROM {table_name} ORDER BY {date_col} DESC LIMIT 1"
                else:
                    query = f"SELECT {matched_col} FROM {table_name} LIMIT 1"
                cursor.execute(query)
                row = cursor.fetchone()
                if row and row[0] is not None:
                    return float(row[0])
        except Exception as e:
            logger.warning(f"获取滞后值失败: {e}")
        finally:
            if HAS_SQLALCHEMY and hasattr(session, 'close'):
                session.close()
        return self._get_default_value(feature)

    def _clip_to_range(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if col in self.feature_ranges:
                low, high = self.feature_ranges[col]
                original = df[col]
                clipped = original.clip(low, high)
                if not original.equals(clipped):
                    changed = (original != clipped).sum()
                    logger.warning(f"特征 {col} 有 {changed} 个值超出范围，已裁剪")
                df[col] = clipped
        return df

    def _build_features(self, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        df = input_df.copy()
        warnings = {}

        if '日期' not in df.columns:
            raise ValueError("输入数据缺少日期列")

        # 时间特征
        df['年'] = df['日期'].dt.year
        df['月'] = df['日期'].dt.month
        df['日'] = df['日期'].dt.day
        df['星期几'] = df['日期'].dt.dayofweek
        df['是否为周末'] = (df['星期几'] >= 5).astype(int)

        # 季节
        def get_season(month):
            if month in [3,4,5]: return '春'
            if month in [6,7,8]: return '夏'
            if month in [9,10,11]: return '秋'
            return '冬'
        df['季节'] = df['月'].apply(get_season)
        season_dummies = pd.get_dummies(df['季节'], prefix='季节')
        df = pd.concat([df, season_dummies], axis=1)
        df.drop('季节', axis=1, inplace=True)

        # 滞后特征（从数据库获取）
        lag_features = [f for f in self.features if 'lag' in f]
        for feat in lag_features:
            if feat in df.columns:
                continue
            base = feat.replace('_lag_', '_').split('_')[0]
            df[feat] = df.apply(lambda row: self._get_lag_value(base), axis=1)

        # 滚动统计特征（使用默认值）
        rolling_features = [f for f in self.features if '均值' in f or '标准差' in f]
        for feat in rolling_features:
            if feat not in df.columns:
                df[feat] = self._get_default_value(feat)

        # 交互特征
        turb_col = None
        flow_col = None
        for col in df.columns:
            col_low = col.lower()
            if any(k in col_low for k in ['浊度', 'turbidity']):
                turb_col = col
            if any(k in col_low for k in ['流量', 'flow', '库区水位']):
                flow_col = col
        if turb_col and flow_col:
            df['浊度_流量_交互'] = df[turb_col] * df[flow_col]

        # 确保所有特征存在
        for feat in self.features:
            if feat not in df.columns:
                df[feat] = self._get_default_value(feat)
            else:
                df[feat] = df[feat].fillna(self._get_default_value(feat))

        # 检查是否超出训练集范围
        if self.training_stats:
            for feat in self.features:
                if feat in self.training_stats:
                    min_val = self.training_stats[feat].get('min')
                    max_val = self.training_stats[feat].get('max')
                    if min_val is not None and max_val is not None:
                        if (df[feat] < min_val).any() or (df[feat] > max_val).any():
                            warnings[feat] = {
                                'min': min_val, 'max': max_val,
                                'current_min': df[feat].min(), 'current_max': df[feat].max()
                            }

        # 数值裁剪
        df = self._clip_to_range(df)

        # 按训练顺序选取特征
        X = df[self.features]
        return X, warnings

    def _predict_without_cache(self, input_data: Union[Dict, pd.DataFrame]) -> Tuple[Union[float, np.ndarray], Dict]:
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()

        if '日期' in input_df.columns:
            input_df['日期'] = pd.to_datetime(input_df['日期'], errors='coerce')
            input_df = input_df.dropna(subset=['日期'])
        else:
            raise ValueError("输入数据必须包含'日期'列")

        if input_df.empty:
            raise ValueError("没有有效日期数据")

        X, warnings = self._build_features(input_df)
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)

        if len(preds) == 1:
            return preds[0], warnings
        else:
            return preds, warnings

    def predict(self, input_data: Union[Dict, pd.DataFrame]) -> Tuple[Union[float, np.ndarray], Dict]:
        if self.use_cache and isinstance(input_data, dict):
            key = hashlib.md5(json.dumps(input_data, sort_keys=True).encode()).hexdigest()
            if key in self._cache:
                return self._cache[key]
            pred, warnings = self._predict_without_cache(input_data)
            self._cache[key] = (pred, warnings)
            return pred, warnings
        else:
            return self._predict_without_cache(input_data)

    def predict_with_interval(self, input_data: Union[Dict, pd.DataFrame], alpha=0.05):
        """返回 (预测值, 下限, 上限) 仅对随机森林有效"""
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()

        if '日期' in input_df.columns:
            input_df['日期'] = pd.to_datetime(input_df['日期'], errors='coerce')
            input_df = input_df.dropna(subset=['日期'])
        else:
            raise ValueError("输入数据必须包含'日期'列")

        if input_df.empty:
            raise ValueError("没有有效日期数据")

        X, _ = self._build_features(input_df)
        X_scaled = self.scaler.transform(X)

        if self.model_type == 'RandomForest' and hasattr(self.model, 'estimators_'):
            all_preds = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
            pred_mean = np.mean(all_preds, axis=0)
            lower = np.percentile(all_preds, 100 * alpha / 2, axis=0)
            upper = np.percentile(all_preds, 100 * (1 - alpha / 2), axis=0)
            if len(pred_mean) == 1:
                return pred_mean[0], lower[0], upper[0]
            else:
                return pred_mean, lower, upper
        else:
            pred = self.model.predict(X_scaled)
            if len(pred) == 1:
                return pred[0], None, None
            else:
                return pred, None, None

    def predict_batch(self, df: pd.DataFrame, chunk_size=DEFAULT_CHUNK_SIZE, parallel=PARALLEL_BATCH) -> pd.DataFrame:
        if '日期' not in df.columns:
            raise ValueError("DataFrame must contain '日期' column")

        if HAS_PSUTIL:
            mem = psutil.virtual_memory()
            row_memory_estimate = 1024
            max_chunk = int(mem.available * 0.8 / row_memory_estimate)
            chunk_size = min(chunk_size, max_chunk)

        df = df.sort_values('日期').reset_index(drop=True)

        def process_chunk(chunk):
            preds, _ = self._predict_without_cache(chunk)
            return preds

        if parallel and NUM_WORKERS > 1:
            from concurrent.futures import ThreadPoolExecutor
            chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                preds_list = list(executor.map(process_chunk, chunks))
            predictions = np.concatenate(preds_list)
        else:
            predictions = []
            for start in range(0, len(df), chunk_size):
                chunk = df.iloc[start:start+chunk_size]
                preds, _ = self._predict_without_cache(chunk)
                predictions.extend(preds)
            predictions = np.array(predictions)

        df['预测投矾量'] = predictions
        return df

    def close(self):
        if hasattr(self, 'engine') and self.engine:
            self.engine.dispose()
        logger.info("资源已释放")