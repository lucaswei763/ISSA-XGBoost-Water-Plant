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

# 导入特征配置
try:
    from data_loader import (
        ALL_FEATURES, BASE_FEATURES_MAP, UI_FEATURES,
        DERIVED_FEATURES, TARGET_COL, DATE_COL,
        compute_derived_features, compute_default_values,
        load_raw_data, clean_data,
    )
    DATA_LOADER_OK = True
except ImportError:
    DATA_LOADER_OK = False
    logger.warning("data_loader 未找到，特征补全功能不可用")

# 尝试导入 SHAP 可解释性
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap 未安装，预测可解释性功能不可用")

# 加载模型


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
        self.default_values = {}  # 特征默认值（中位数）

        # 加载默认值
        self._load_defaults()

        # 加载模型
        self._load_models()

        # 初始化 SHAP 可解释性
        self.shap_explainer = None
        self.shap_background = None
        self.shap_expected = None
        if SHAP_AVAILABLE:
            self._init_shap()

        # 初始化异常工况检测
        self.anomaly_detector = None
        self.anomaly_threshold = None
        self.anomaly_features = None
        self.feature_ranges = {}
        self.pred_bounds = None  # 预测区间
        self._init_anomaly()
        self._init_bounds()

    def _load_defaults(self):
        """加载特征默认值 — 预测时补全缺失的实验室/衍生特征"""
        defaults_path = os.path.join(self.model_dir, 'feature_defaults.json')
        try:
            if os.path.exists(defaults_path):
                with open(defaults_path, 'r') as f:
                    self.default_values = json.load(f)
                logger.info(f"✅ 特征默认值已加载 ({len(self.default_values)} 个)")
            elif DATA_LOADER_OK:
                # 从 Excel 实时计算
                from data_loader import find_data_file
                excel_path = find_data_file()
                df = load_raw_data(excel_path)
                df, _ = clean_data(df)
                self.default_values = compute_default_values(df)
                # 缓存以供后续使用
                os.makedirs(os.path.dirname(defaults_path) or '.', exist_ok=True)
                with open(defaults_path, 'w') as f:
                    json.dump(self.default_values, f, indent=2)
                logger.info(f"✅ 特征默认值已计算并缓存 ({len(self.default_values)} 个)")
            else:
                logger.warning("无法加载特征默认值，缺失特征将填 0")
        except Exception as e:
            logger.warning(f"加载默认值失败: {e}，缺失特征将填 0")

    def _augment_input(self, input_data):
        """
        将 UI 的 8 个输入补全为模型所需的完整特征字典。
        - 8 个 UI 基础特征直接映射
        - lag1/lag2 从 Excel 历史数据查找（支持重启场景）
        """
        augmented = dict(input_data)

        # === 1. UI 键映射到特征名 ===
        # 注意：模型训练时使用的是 '原水量_Km3'（日值），需要将 UI 传入的 '原水量_Km3_hour'（小时值）转换后映射过去
        ui_map = {
            '浑浊度_0点': '浑浊度_0点', '浑浊度（NTU）_0点': '浑浊度_0点',
            '原水量_Km3_hour': ('原水量_Km3', 24.0), '原水量（Km³/h）': ('原水量_Km3', 24.0),
            '温度_C': '温度_C', '温度（℃）_9点': '温度_C',
            '氨氮_mg_per_L': '氨氮_mg_per_L', '氨氮（mg/L）_9点': '氨氮_mg_per_L',
            'pH值': 'pH值', 'pH值_9点': 'pH值',
        }
        for ui_key, feat_info in ui_map.items():
            if ui_key in input_data:
                try:
                    val = float(input_data[ui_key])
                    if isinstance(feat_info, tuple):
                        feat_name, multiplier = feat_info
                        if feat_name not in augmented:
                            augmented[feat_name] = val * multiplier
                    else:
                        feat_name = feat_info
                        if feat_name not in augmented:
                            augmented[feat_name] = val
                except (ValueError, TypeError):
                    pass

        # === 2. 补全缺失的基础特征 ===
        base_feats = ['浑浊度_0点','原水量_Km3_hour','温度_C','氨氮_mg_per_L','pH值']
        for f in base_feats:
            if f not in augmented or augmented.get(f) is None:
                augmented[f] = self.default_values.get(f, 0.0)
            # 额外检查：已经是nan的情况
            try:
                if np.isnan(float(augmented.get(f, 0))):
                    augmented[f] = self.default_values.get(f, 0.0)
            except (ValueError, TypeError):
                pass

        # === 3. 投矾量滞后特征 — 从 Excel 查历史值 ===
        date_str = augmented.get('日期') or input_data.get('日期')
        if 'lag1' not in augmented or augmented.get('lag1') is None:
            augmented['lag1'] = self._get_last_dosage(date_str, days_ago=0)
        if 'lag2' not in augmented or augmented.get('lag2') is None:
            augmented['lag2'] = self._get_last_dosage(date_str, days_ago=1)

        return augmented

    def _get_runtime_dir(self):
        """运行时数据目录：exe同级/data/ (打包后) 或 项目data/ (开发时)"""
        if getattr(sys, 'frozen', False):
            return os.path.join(os.path.dirname(sys.executable), 'data')
        return os.path.abspath('data')

    def _get_last_dosage(self, date_str, days_ago=0):
        """
        获取历史投矾量：runtime记录 > Excel历史 > 中位数
        """
        raw_val = None

        # 1. 运行时记录（最新，支持重启）
        runtime_path = os.path.join(self._get_runtime_dir(), 'runtime_records.json')
        try:
            if os.path.exists(runtime_path):
                with open(runtime_path, 'r') as f:
                    records = json.load(f)
                target_date = pd.to_datetime(date_str) - timedelta(days=days_ago + 1)
                key = target_date.strftime('%Y-%m-%d')
                if key in records and records[key] and float(records[key]) > 0:
                    raw_val = float(records[key])
        except Exception:
            pass

        # 2. Excel — 近3日加权均值（平滑极端日间波动）
        try:
            if date_str and DATA_LOADER_OK:
                from data_loader import load_raw_data, clean_data, TARGET_COL as TC, DATE_COL as DC
                target_date = pd.to_datetime(date_str) - timedelta(days=days_ago + 1)
                if not hasattr(self, '_history_df') or self._history_df is None:
                    df = load_raw_data()
                    df, _ = clean_data(df)
                    df = df[df[TC] > 0][[DC, TC]].copy()
                    df['_ds'] = pd.to_datetime(df[DC]).dt.strftime('%Y-%m-%d')
                    self._history_df = df.set_index('_ds').sort_index()
                    self._history_col = TC
                # 近3日加权：当天50% + 前1天30% + 前2天20%
                vals = []
                for i, w in [(0, 0.5), (1, 0.3), (2, 0.2)]:
                    lookup = (target_date - timedelta(days=i)).strftime('%Y-%m-%d')
                    if lookup in self._history_df.index:
                        v = float(self._history_df.loc[lookup, self._history_col])
                        if hasattr(v, '__len__') and not isinstance(v, str):
                            v = float(v.iloc[0]) if len(v) > 0 else 0
                        if v > 0:
                            vals.append((v, w))
                if vals:
                    raw_val = sum(v * w for v, w in vals) / sum(w for _, w in vals)
        except Exception:
            pass

        # 3. 内存缓存
        if raw_val is None:
            if days_ago == 0 and 'last_prediction' in self.cache:
                raw_val = self.cache['last_prediction']
            elif days_ago == 1 and 'prev_prediction' in self.cache:
                raw_val = self.cache['prev_prediction']

        # 4. fallback（数值全部取自本地 models/ 配置，源码内不留业务常量）
        if raw_val is None:
            fallback = getattr(self, 'shap_expected', None) or self.default_values.get(TARGET_COL)
            if fallback is None or fallback == 0:
                fallback = self.default_values.get(TARGET_COL)
            if fallback is None:
                logger.warning('缺少 lag 基准值：请提供 models/feature_defaults.json')
                return 0.0
            return float(fallback)

        # 5. 极端值保护：lag偏离中位数>3σ时用中位数替代
        median = self.default_values.get(TARGET_COL)
        target_std = self.default_values.get('_target_std')
        if median and target_std and median > 0 and raw_val is not None:
            if abs(raw_val - median) > target_std * 3:
                logger.info(f'lag值{raw_val:.0f}偏离中位数{median:.0f}(>3σ={target_std*3:.0f})，用中位数替代')
                return float(median)

        return float(raw_val)

    def record_actual(self, date_str, actual_value):
        """
        记录实际投矾量到运行时数据库。
        下次预测同日期时，lag 将使用此值。
        """
        runtime_path = os.path.join(self._get_runtime_dir(), 'runtime_records.json')
        records = {}
        try:
            if os.path.exists(runtime_path):
                with open(runtime_path, 'r') as f:
                    records = json.load(f)
        except Exception:
            pass
        records[date_str] = float(actual_value)
        os.makedirs(os.path.dirname(runtime_path) or '.', exist_ok=True)
        with open(runtime_path, 'w') as f:
            json.dump(records, f, indent=2)
        logger.info(f"✅ 已记录 {date_str} 实际投矾量: {actual_value} kg")
        return True

    def _load_models(self):
        """加载模型"""
        self.use_stacked = False
        self._load_xgboost_model()

    def _load_xgboost_model(self):
        """加载原始 XGBoost 模型（回退用，缺失文件不阻止启动）"""
        try:
            model_path = os.path.join(self.model_dir, 'best_model.pkl')
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            imputer_path = os.path.join(self.model_dir, 'imputer.pkl')
            features_path = os.path.join(self.model_dir, 'selected_features.pkl')

            for path, name in [(model_path, 'best_model.pkl'),
                               (scaler_path, 'scaler.pkl'),
                               (features_path, 'selected_features.pkl')]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"缺少回退模型文件: {name}")

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.features = joblib.load(features_path)

            # imputer.pkl 容错：缺失时用默认均值填充器（先 fit 一个占位样本）
            if os.path.exists(imputer_path):
                self.imputer = joblib.load(imputer_path)
            else:
                from sklearn.impute import SimpleImputer
                self.imputer = SimpleImputer(strategy='mean')
                # 用全零占位数据 fit，后续预测输入无 NaN 时不影响结果
                dummy = np.zeros((1, len(self.features)))
                self.imputer.fit(dummy)
                logger.warning("imputer.pkl 不存在，使用默认均值填充器（回退模型预测可能不准）")

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
            logger.info(f"✅ 原始 XGBoost 回退模型加载成功（{len(self.features)} 个特征）")

        except Exception as e:
            logger.warning(f"XGBoost 回退模型不可用: {e}（这不影响堆叠模型预测）")
            self.model = None
            self.scaler = None
            self.imputer = None
            self.features = []

    def _init_shap(self):
        """初始化 SHAP 解释器"""
        bg_path = os.path.join(self.model_dir, 'shap_background.pkl')
        exp_path = os.path.join(self.model_dir, 'shap_expected.pkl')
        try:
            if os.path.exists(bg_path) and os.path.exists(exp_path) and self.model is not None:
                self.shap_background = joblib.load(bg_path)
                self.shap_expected = float(joblib.load(exp_path))
                self.shap_explainer = shap.TreeExplainer(
                    self.model, feature_perturbation='tree_path_dependent'
                )
                logger.info(f"✅ SHAP 解释器已初始化 (expected={self.shap_expected:.1f})")
            else:
                logger.info("SHAP 背景文件不存在，跳过可解释性初始化")
        except Exception as e:
            logger.warning(f"SHAP 初始化失败: {e}")

    def _init_bounds(self):
        """加载预测区间参数"""
        bounds_path = os.path.join(self.model_dir, 'prediction_bounds.pkl')
        try:
            if os.path.exists(bounds_path):
                self.pred_bounds = joblib.load(bounds_path)
                logger.info(f"✅ 预测区间已加载 (90%置信: {self.pred_bounds['p5']:.0f}~{self.pred_bounds['p95']:.0f} kg)")
        except Exception as e:
            logger.warning(f"预测区间加载失败: {e}")

    def _init_anomaly(self):
        """初始化异常工况检测器"""
        det_path = os.path.join(self.model_dir, 'anomaly_detector.pkl')
        thr_path = os.path.join(self.model_dir, 'anomaly_threshold.pkl')
        feat_path = os.path.join(self.model_dir, 'anomaly_features.pkl')
        rng_path = os.path.join(self.model_dir, 'feature_ranges.json')
        try:
            if os.path.exists(det_path) and os.path.exists(thr_path):
                self.anomaly_detector = joblib.load(det_path)
                self.anomaly_threshold = float(joblib.load(thr_path))
                if os.path.exists(feat_path):
                    self.anomaly_features = joblib.load(feat_path)
            if os.path.exists(rng_path):
                with open(rng_path, 'r') as f:
                    self.feature_ranges = json.load(f)
            logger.info(f"✅ 异常检测器已加载 (阈值={self.anomaly_threshold:.0f})")
        except Exception as e:
            logger.warning(f"异常检测器加载失败: {e}")

    def _check_trend(self, current_pred, date_str=None):
        """
        投矾量突变趋势预警。
        对比 Excel 中过去7天的实际投矾量均值，而非缓存的上次预测值。
        阈值：>20% warning, >40% critical
        """
        if not date_str:
            return None

        try:
            # 从 Excel 计算过去7天实际投矾量均值
            if not hasattr(self, '_history_df') or self._history_df is None:
                return None

            target_date = pd.to_datetime(date_str)
            recent_vals = []
            for d in range(1, 8):
                lookup = (target_date - timedelta(days=d)).strftime('%Y-%m-%d')
                if lookup in self._history_df.index:
                    recent_vals.append(float(self._history_df.loc[lookup, self._history_col]))
            if len(recent_vals) < 3:
                return None
            avg_7d = np.mean(recent_vals)
        except Exception:
            return None

        delta = current_pred - avg_7d
        pct = abs(delta) / max(abs(avg_7d), 1) * 100

        if pct > 40:
            return {
                'level': 'critical',
                'message': f'与基准线({avg_7d:.0f}kg)相比变化 {pct:.0f}%',
                'delta_kg': round(float(delta), 1),
                'delta_pct': round(float(pct), 1),
                'baseline': round(float(avg_7d), 1),
                'baseline_desc': '近7日实际均值',
            }
        elif pct > 20:
            return {
                'level': 'warning',
                'message': f'与基准线({avg_7d:.0f}kg)相比变化 {pct:.0f}%',
                'delta_kg': round(float(delta), 1),
                'delta_pct': round(float(pct), 1),
                'baseline': round(float(avg_7d), 1),
                'baseline_desc': '近7日实际均值',
            }
        return None

    def _check_confidence(self, input_data):
        """
        z-score 优先异常检测。
        只要有一个输入特征偏离训练均值超过 5σ，直接判定为 low。
        """
        if self.anomaly_detector is None or self.anomaly_features is None:
            return {'confidence': 1.0, 'level': 'high', 'distance': 0, 'anomaly_flags': []}

        try:
            feat_vec = []
            for f in self.anomaly_features:
                val = input_data.get(f)
                if val is None:
                    val = self.default_values.get(f, 0)
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = self.default_values.get(f, 0)
                if np.isnan(val):
                    val = self.default_values.get(f, 0)
                feat_vec.append(float(val))
        except Exception:
            return {'confidence': 1.0, 'level': 'high', 'distance': 0, 'anomaly_flags': []}

        # 马氏距离
        distance = float(self.anomaly_detector.mahalanobis([feat_vec])[0])
        threshold = self.anomaly_threshold

        # 单特征 z-score + 超出训练范围检测
        max_z = 0
        anomaly_flags = []
        feat_stats_path = os.path.join(self.model_dir, 'feature_stats.pkl')
        try:
            if os.path.exists(feat_stats_path):
                feat_stats = joblib.load(feat_stats_path)
                for f in self.anomaly_features:
                    if f in feat_stats:
                        mu = feat_stats[f]['mean']
                        std = feat_stats[f]['std']
                        if std > 0:
                            z = abs(feat_vec[self.anomaly_features.index(f)] - mu) / std
                            max_z = max(max_z, z)
                            if z > 3.0:
                                anomaly_flags.append({
                                    'feature': f, 'value': round(feat_vec[self.anomaly_features.index(f)], 2),
                                    'expected': round(mu, 1), 'z_score': round(z, 1)
                                })
        except Exception:
            pass

        # 补充：超出训练数据极值范围（比z-score更严格）
        try:
            if hasattr(self, 'feature_ranges') and self.feature_ranges:
                for f in self.anomaly_features:
                    if f in self.feature_ranges:
                        rng = self.feature_ranges[f]
                        val = feat_vec[self.anomaly_features.index(f)]
                        if val < rng['min'] or val > rng['max']:
                            rng_min = rng['min']; rng_max = rng['max']
                            max_z = max(max_z, 6.0)  # 触发 low
                            if not any(af['feature'] == f for af in anomaly_flags):
                                anomaly_flags.append({
                                    'feature': f, 'value': round(val, 2),
                                    'expected': f'{rng_min:.0f}~{rng_max:.0f}',
                                    'z_score': 999.0
                                })
        except Exception:
            pass

        # z-score 优先分级：单特征极端 > 综合距离
        if max_z > 5.0 or distance > threshold * 4:
            level = 'low'
            confidence = max(0.1, 0.5 - (max_z - 5) * 0.1)
        elif max_z > 3.0 or distance > threshold * 2:
            level = 'medium'
            confidence = 0.7
        else:
            level = 'high'
            confidence = 1.0

        return {
            'confidence': round(min(1.0, max(0.1, confidence)), 4),
            'level': level,
            'distance': round(distance, 1),
            'anomaly_flags': anomaly_flags
        }

    def explain(self, input_data):
        """
        SHAP 可解释预测。
        返回:
        {
            'prediction': float,      # 预测值
            'base_value': float,      # 基准值（训练集期望预测）
            'contributions': [        # 各特征贡献，按影响绝对值降序
                {'feature': str, 'value': float, 'shap': float, 'direction': 'up'|'down'},
                ...
            ]
        }
        """
        # 特征补全 + 预测（即使SHAP不可用也要进行预测）
        input_data = self._augment_input(input_data)
        pred, warnings = self.predict(input_data)

        if pred is None:
            return {
                'prediction': None,
                'error': '模型预测失败，请检查模型文件是否完整'
            }

        # SHAP 不可用时，返回基础预测结果
        if self.shap_explainer is None:
            confidence_info = self._check_confidence(input_data)
            date_str = input_data.get('日期') or input_data.get('date')
            trend_alert = self._check_trend(pred, date_str)
            return {
                'prediction': round(float(pred), 1),
                'lower_bound': round(float(pred + self.pred_bounds['p5']), 1) if self.pred_bounds else None,
                'upper_bound': round(float(pred + self.pred_bounds['p95']), 1) if self.pred_bounds else None,
                'base_value': None,
                'contributions': [],
                'confidence': confidence_info,
                'trend_alert': trend_alert,
                'warnings': warnings,
                'error': 'SHAP 解释器未初始化，无法提供特征贡献分析'
            }

        # 构建标准化输入向量
        input_vector = []
        for f in self.features:
            val = input_data.get(f, 0)
            try:
                input_vector.append(float(val))
            except (ValueError, TypeError):
                input_vector.append(0.0)

        X_input = np.array([input_vector])

        # 插值 + 标准化
        if self.imputer is not None:
            X_input = self.imputer.transform(X_input)
        if self.scaler is not None:
            X_input = self.scaler.transform(X_input)

        # SHAP 计算
        shap_values = self.shap_explainer.shap_values(X_input)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_vals = shap_values[0]

        # 构建贡献列表（只保留用户输入的关键特征）
        key_features = ['浑浊度_0点', '原水量_Km3', '温度_C', '氨氮_mg_per_L', 'pH值', 'lag1', 'lag2']
        contributions = []
        for i, f in enumerate(self.features):
            if f in key_features:
                contributions.append({
                    'feature': f,
                    'value': round(input_vector[i], 4),
                    'shap': round(float(shap_vals[i]), 1),
                    'direction': 'up' if shap_vals[i] > 0 else 'down'
                })

        # 按影响绝对值排序
        contributions.sort(key=lambda x: abs(x['shap']), reverse=True)

        # 异常工况置信度
        confidence_info = self._check_confidence(input_data)

        # 趋势突变预警（对比Excel近7日实际均值）
        date_str = input_data.get('日期') or input_data.get('date')
        trend_alert = self._check_trend(pred, date_str)

        return {
            'prediction': round(float(pred), 1),
            'lower_bound': round(float(pred + self.pred_bounds['p5']), 1) if self.pred_bounds else None,
            'upper_bound': round(float(pred + self.pred_bounds['p95']), 1) if self.pred_bounds else None,
            'base_value': round(float(self.shap_expected), 1),
            'contributions': contributions,
            'confidence': confidence_info,
            'trend_alert': trend_alert,
            'warnings': warnings
        }

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

    def _predict_xgboost(self, input_data):
        """使用原始 XGBoost 模型预测"""
        try:
            # 特征补全
            input_data = self._augment_input(input_data)

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

            return float(pred), []

        except Exception as e:
            logger.error(f"XGBoost 预测失败: {e}")
            raise

    def predict(self, input_data):
        """预测投矾量"""
        try:
            if self.model is None:
                return None, ["模型未加载，请检查 models/ 目录"]
            if self.scaler is None:
                return None, ["数据标准化器未加载"]
            pred, warnings = self._predict_xgboost(input_data)
            if pred is not None:
                self._cache_prediction(pred)
            return pred, warnings
        except Exception as e:
            logger.error(f"预测异常: {e}")
            import traceback
            traceback.print_exc()
            return None, [f"预测失败: {str(e)[:100]}"]

    def _cache_prediction(self, pred):
        """缓存预测结果，供下次预测的 lag 特征使用"""
        if 'last_prediction' in self.cache:
            self.cache['prev_prediction'] = self.cache['last_prediction']
        self.cache['last_prediction'] = float(pred)

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
            'model_type': getattr(self, 'model_type', '未知'),
            'version': getattr(self, 'version', '?'),
            'features': getattr(self, 'features', []),
            'target_col': getattr(self, 'target_col', '?'),
            'use_stacked': getattr(self, 'use_stacked', False),
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
            '原水量_Km3_hour': 8.3,
            '温度_C': 15,
            '氨氮_mg_per_L': 0.1,
            'pH值': 7.2,
        }

        pred, warnings = predictor.predict(test_data)
        print(f"\n📊 测试预测:")
        print(f"   输入: {test_data}")
        print(f"   预测投矾量: {pred:.2f} kg")
        if warnings:
            print(f"   警告: {warnings}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")