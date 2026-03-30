#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
水厂投矾量预测服务模块（终极性能 + 平滑算法融合版）
- 修复 N+1 数据库查询性能瓶颈（一次性拉取 7天 快照支撑平滑特征）
- 引入 LRU 缓存机制防内存泄漏
- 完美适配 ISSA-XGBoost 平滑抗锯齿模型
"""

import os
import sys
import json
import hashlib
import logging
import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from typing import Dict, Union, Tuple
from collections import OrderedDict

logger = logging.getLogger(__name__)

# ==================== 配置 ====================
MODEL_DIR = os.getenv('MODEL_DIR', 'models')
DB_PATH = os.getenv('DB_PATH', 'data/water_data.db')
USE_CACHE = os.getenv('USE_CACHE', 'true').lower() == 'true'
CACHE_SIZE = int(os.getenv('CACHE_SIZE', 1024))
DEFAULT_CHUNK_SIZE = int(os.getenv('DEFAULT_CHUNK_SIZE', 5000))

# 默认特征极值兜底
DEFAULT_FEATURE_RANGES = {
    '浊度': (0, 100), '流量': (0, 200000), 'pH': (0, 14), '温度': (-10, 50)
}


class WaterPredictor:
    def __init__(self, model_dir=MODEL_DIR, db_path=DB_PATH, use_cache=USE_CACHE):
        self.model_dir = self._resolve_path(model_dir)
        self.db_path = self._resolve_path(db_path)

        # LRU 缓存设置
        self.use_cache = use_cache
        self.cache_size = CACHE_SIZE
        self._cache = OrderedDict()

        self.model = None
        self.scaler = None
        self.features = None
        self.target_col = None
        self.model_type = None
        self.feature_ranges = {}

        self.conn = None
        # 👑 核心融合：改用 DataFrame 存储 7 天快照，支撑平滑计算
        self._recent_db_records = pd.DataFrame()

        self._load_components()
        self._init_db()

    def _resolve_path(self, path):
        if getattr(sys, 'frozen', False):
            base = sys._MEIPASS
        else:
            base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, path)

    def _load_components(self):
        """加载最新 ISSA-XGBoost 模型及元数据"""
        metadata_path = os.path.join(self.model_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            self.model_type = metadata.get('model_type', 'XGBoost')
            self.features = metadata.get('features')
            self.target_col = metadata.get('target_col')
            self.feature_ranges = metadata.get('feature_ranges', DEFAULT_FEATURE_RANGES)
        else:
            self.features = joblib.load(os.path.join(self.model_dir, 'selected_features.pkl'))
            self.feature_ranges = DEFAULT_FEATURE_RANGES

        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        self.model = joblib.load(os.path.join(self.model_dir, 'best_model.pkl'))
        logger.info(f"模型加载成功，需要特征数={len(self.features)}")

    def _init_db(self):
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        except Exception as e:
            logger.warning(f"数据库连接失败: {e}")

    def _get_default_value(self, feature: str) -> float:
        if feature in self.feature_ranges:
            low, high = self.feature_ranges[feature]
            return (low + high) / 2.0
        return 0.0

    def _refresh_recent_records(self):
        """👑 核心修改：一次性从数据库获取最近 7 天记录，解决 N+1 问题并支撑平滑计算"""
        if self.conn is None:
            self._recent_db_records = pd.DataFrame()
            return

        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            table_name = next((tbl for tbl in ['merged_data', 'water_quality', 'original_data'] if tbl in tables), None)
            if not table_name:
                return

            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            date_col = next((col for col in columns if '日期' in col or 'date' in col.lower()), None)

            # 提取 7 天快照！
            if date_col:
                query = f"SELECT * FROM {table_name} ORDER BY {date_col} DESC LIMIT 7"
            else:
                query = f"SELECT * FROM {table_name} LIMIT 7"

            df_recent = pd.read_sql_query(query, self.conn)

            if date_col:
                df_recent[date_col] = pd.to_datetime(df_recent[date_col])
                # 正序排列，最旧的在上面，最新的在下面，方便 rolling 计算
                df_recent = df_recent.sort_values(by=date_col).reset_index(drop=True)

            self._recent_db_records = df_recent

        except Exception as e:
            logger.warning(f"获取最新滞后数据快照失败: {e}")
            self._recent_db_records = pd.DataFrame()

    def _build_features(self, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        df = input_df.copy()
        pred_warnings = {}

        if '日期' not in df.columns:
            # 如果UI没传日期，默认补上今天
            df['日期'] = pd.Timestamp.today()

        df['日期'] = pd.to_datetime(df['日期'])
        df['年'] = df['日期'].dt.year
        df['月'] = df['日期'].dt.month
        df['星期几'] = df['日期'].dt.dayofweek
        df['是否为周末'] = (df['星期几'] >= 5).astype(int)

        # 刷新 7 天内存快照
        self._refresh_recent_records()
        hist_df = self._recent_db_records

        turb_col = next((c for c in df.columns if '浊度' in c or 'turbidity' in c.lower()), None)
        flow_col = next((c for c in df.columns if '流量' in c or 'flow' in c.lower() or 'supply' in c.lower()), None)

        hist_target_col = next(
            (c for c in hist_df.columns if any(kw in c.lower() for kw in ['矾', 'alum', '投矾']) and 'lag' not in c),
            None)
        hist_turb_col = next((c for c in hist_df.columns if '浊度' in c or 'turbidity' in c.lower()), None)
        hist_flow_col = next(
            (c for c in hist_df.columns if '流量' in c or 'flow' in c.lower() or 'supply' in c.lower()), None)

        # 1. 直接从内存快照中提取滞后特征 (O(1) 性能)
        for lag in [1, 2, 3]:
            feat_name = f'{self.target_col}_lag_{lag}天' if self.target_col else f'alum_kg_lag_{lag}天'
            if feat_name in self.features:
                if not hist_df.empty and hist_target_col and len(hist_df) >= lag:
                    df[feat_name] = hist_df.iloc[-lag][hist_target_col]
                else:
                    df[feat_name] = self._get_default_value(feat_name)

        # 2. 结合内存快照和当前输入，计算 3天/7天平滑特征
        if turb_col and hist_turb_col and not hist_df.empty:
            turb_series = pd.concat([hist_df[hist_turb_col], df[turb_col]]).reset_index(drop=True)
            df[f'{turb_col}_3天平滑'] = turb_series.rolling(3, min_periods=1).mean().iloc[-1]
            df[f'{turb_col}_7天平滑'] = turb_series.rolling(7, min_periods=1).mean().iloc[-1]

        if flow_col and hist_flow_col and not hist_df.empty:
            flow_series = pd.concat([hist_df[hist_flow_col], df[flow_col]]).reset_index(drop=True)
            df[f'{flow_col}_3天平滑'] = flow_series.rolling(3, min_periods=1).mean().iloc[-1]
            df[f'{flow_col}_7天平滑'] = flow_series.rolling(7, min_periods=1).mean().iloc[-1]

        # 计算交互特征
        if turb_col and flow_col:
            t_3 = df[f'{turb_col}_3天平滑'].values[0] if f'{turb_col}_3天平滑' in df else df[turb_col].values[0]
            f_3 = df[f'{flow_col}_3天平滑'].values[0] if f'{flow_col}_3天平滑' in df else df[flow_col].values[0]
            df['平滑后浊度_流量交互'] = t_3 * f_3

        # 兜底：补齐模型需要的剩余特征
        for feat in self.features:
            if feat not in df.columns:
                df[feat] = self._get_default_value(feat)
            else:
                df[feat] = df[feat].fillna(self._get_default_value(feat))

        return df[self.features], pred_warnings

    def _predict_without_cache(self, input_data: Union[Dict, pd.DataFrame]) -> Tuple[Union[float, np.ndarray], Dict]:
        input_df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data.copy()

        X, pred_warnings = self._build_features(input_df)
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)

        return float(preds[0]) if len(preds) == 1 else preds, pred_warnings

    def predict(self, input_data: Union[Dict, pd.DataFrame]) -> Tuple[Union[float, np.ndarray], Dict]:
        """LRU 缓存机制，避免重复预测导致内存膨胀"""
        if self.use_cache and isinstance(input_data, dict):
            key = hashlib.md5(json.dumps(input_data, sort_keys=True).encode()).hexdigest()

            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

            pred, pred_warnings = self._predict_without_cache(input_data)
            self._cache[key] = (pred, pred_warnings)

            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

            return pred, pred_warnings
        else:
            return self._predict_without_cache(input_data)

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("数据库连接已释放")