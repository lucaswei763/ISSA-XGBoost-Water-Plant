#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
水厂投矾量预测 Web 服务（最终稳定版）
- 根据模型实际特征名构造输入（包含 alum_per_unit）
- 健康检查、单条预测、批量预测、热加载
- 日志轮转、优雅退出、请求追踪
"""

import os
import sys
import io
import json
import uuid
import signal
import logging
import logging.handlers
import traceback
from flask import Flask, request, jsonify, send_file, abort, g
from werkzeug.utils import secure_filename
import pandas as pd

from predictor_service import WaterPredictor

# ==================== 配置 ====================
MODEL_DIR = os.getenv('MODEL_DIR', 'models')
DB_PATH = os.getenv('DB_PATH', 'data/water_data.db')
PORT = int(os.getenv('PORT', 5000))
HOST = os.getenv('HOST', '0.0.0.0')
USE_CACHE = os.getenv('USE_CACHE', 'true').lower() == 'true'
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')
MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
RELOAD_TOKEN = os.getenv('RELOAD_TOKEN', 'changeme')
RATE_LIMIT = os.getenv('RATE_LIMIT', '')
CORS_ENABLED = os.getenv('CORS_ENABLED', 'false').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

# 创建必要目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# ==================== 日志配置 ====================
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        try:
            record.request_id = getattr(g, 'request_id', '-')
        except RuntimeError:
            record.request_id = '-'
        return True

file_handler = logging.handlers.RotatingFileHandler(
    'logs/app.log', maxBytes=10*1024*1024, backupCount=5
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(request_id)s - %(message)s'
))
file_handler.addFilter(RequestIdFilter())

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(request_id)s - %(message)s'
))
console_handler.addFilter(RequestIdFilter())

root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

# ==================== Flask 应用 ====================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['JSON_SORT_KEYS'] = False

# 跨域支持
if CORS_ENABLED:
    try:
        from flask_cors import CORS
        CORS(app)
        logger.info("CORS 已启用")
    except ImportError:
        logger.warning("flask-cors 未安装，跨域功能未启用")

# 限流支持
if RATE_LIMIT:
    try:
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address
        limiter = Limiter(app, key_func=get_remote_address)
        logger.info(f"限流已启用: {RATE_LIMIT}")
    except ImportError:
        logger.warning("flask-limiter 未安装，限流功能未启用")
        limiter = None
else:
    limiter = None

# 全局预测器
predictor = None

def load_model():
    global predictor
    try:
        predictor = WaterPredictor(model_dir=MODEL_DIR, db_path=DB_PATH, use_cache=USE_CACHE)
        logger.info("预测器加载成功")
    except Exception as e:
        logger.error(f"预测器加载失败: {e}")
        predictor = None

load_model()

# ==================== 请求追踪 ====================
@app.before_request
def before_request():
    g.request_id = str(uuid.uuid4())[:8]

# ==================== 模型热加载 ====================
@app.route('/reload', methods=['POST'])
def reload_model():
    token = request.headers.get('X-Reload-Token')
    if token != RELOAD_TOKEN:
        return jsonify({'error': '未授权'}), 401
    global predictor
    try:
        new_predictor = WaterPredictor(model_dir=MODEL_DIR, db_path=DB_PATH, use_cache=USE_CACHE)
        predictor = new_predictor
        logger.info("模型热加载成功")
        return jsonify({'status': 'ok', 'message': '模型已重新加载'})
    except Exception as e:
        logger.error(f"模型热加载失败: {e}")
        return jsonify({'error': str(e)}), 500

# ==================== 健康检查 ====================
@app.route('/health', methods=['GET'])
def health_check():
    if predictor is None:
        return jsonify({'status': 'unhealthy', 'reason': '模型未加载'}), 503
    status = {
        'status': 'healthy',
        'model_loaded': True,
        'model_type': predictor.model_type,
        'model_version': predictor.version,
        'db_connected': predictor.engine is not None or hasattr(predictor, 'conn'),
        'cache_enabled': USE_CACHE,
        'features': predictor.features,
        'target': predictor.target_col,
        'feature_ranges': predictor.feature_ranges
    }
    return jsonify(status)

# ==================== 单条预测 ====================
@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    if predictor is None:
        return jsonify({'error': '预测模型未加载，请联系管理员'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': '未提供JSON数据'}), 400

    # 必填字段
    required_fields = ['date', 'turbidity', 'flow', 'ph', 'temperature']
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({'error': f'缺少必填字段: {", ".join(missing)}'}), 400

    try:
        # 根据模型实际特征名构造输入字典（包含 alum_per_unit）
        input_dict = {
            '日期': data['date'],
            'turbidity_0': float(data['turbidity']),
            'water_supply_km3': float(data['flow']),
            'ph_value': float(data['ph']),
            'temperature': float(data['temperature']),
            'ammonia_nitrogen': float(data.get('ammonia', 0.0)),
            'reservoir_level': float(data.get('water_level', 30.0)),
            'alum_per_unit': float(data.get('alum_per_unit', 0.0))
        }

        # 是否返回置信区间
        with_interval = request.args.get('interval', 'false').lower() == 'true'

        if with_interval and predictor.model_type == 'RandomForest':
            pred, lower, upper = predictor.predict_with_interval(input_dict, alpha=0.05)
            response = {
                'prediction': float(pred),
                'lower_bound': float(lower) if lower is not None else None,
                'upper_bound': float(upper) if upper is not None else None
            }
        else:
            pred, warnings = predictor.predict(input_dict)
            response = {'prediction': float(pred)}
            if warnings:
                response['warning'] = f"输入值超出训练范围，预测可能不准。详情：{warnings}"

        return jsonify(response)

    except ValueError as e:
        return jsonify({'error': f'数值错误: {str(e)}'}), 400
    except Exception as e:
        logger.exception("预测失败")
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

# 若启用限流，为路由添加装饰器
if RATE_LIMIT and limiter:
    predict_v1 = limiter.limit(RATE_LIMIT)(predict_v1)

# ==================== 批量预测 ====================
@app.route('/v1/batch', methods=['POST'])
def batch_predict_v1():
    if predictor is None:
        return jsonify({'error': '预测模型未加载，请联系管理员'}), 503

    if 'file' not in request.files:
        return jsonify({'error': '未上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    filename = secure_filename(file.filename)
    if not filename.endswith(('.xlsx', '.xls')):
        return jsonify({'error': '文件必须是 Excel 格式（.xlsx 或 .xls）'}), 400

    try:
        df = pd.read_excel(file)

        if '日期' not in df.columns:
            return jsonify({'error': 'Excel 文件必须包含 "日期" 列'}), 400

        # 必须的列名（中文）
        required_cols = ['浊度', '流量', 'pH', '温度']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            return jsonify({'error': f'Excel 缺少必要列: {", ".join(missing_cols)}'}), 400

        # 重命名列以匹配模型特征
        df = df.rename(columns={
            '浊度': 'turbidity_0',
            '流量': 'water_supply_km3',
            'pH': 'ph_value',
            '温度': 'temperature',
            '日期': '日期'
        })
        # 可选列
        if '氨氮' in df.columns:
            df['ammonia_nitrogen'] = df['氨氮']
        else:
            df['ammonia_nitrogen'] = 0.0
        if '库区水位' in df.columns:
            df['reservoir_level'] = df['库区水位']
        else:
            df['reservoir_level'] = 30.0
        # 添加 alum_per_unit 特征（默认0）
        if 'alum_per_unit' not in df.columns:
            df['alum_per_unit'] = 0.0

        # 确保所有特征都存在（缺失的用默认值）
        for col in ['turbidity_0', 'water_supply_km3', 'ph_value', 'temperature', 'ammonia_nitrogen', 'reservoir_level', 'alum_per_unit']:
            if col not in df.columns:
                df[col] = 0.0

        parallel = request.args.get('parallel', 'false').lower() == 'true'
        result_df = predictor.predict_batch(df, parallel=parallel)

        # 输出结果
        output = io.BytesIO()
        result_df.to_excel(output, index=False)
        output.seek(0)

        return send_file(
            output,
            as_attachment=True,
            download_name=f'result_{filename}',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        logger.exception("批量预测失败")
        return jsonify({'error': f'批量预测失败: {str(e)}'}), 500

# ==================== 文件下载 ====================
@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    safe_name = secure_filename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        abort(404)
    return send_file(file_path, as_attachment=True)

# ==================== 错误处理 ====================
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(traceback.format_exc())
    return jsonify({
        'error': '服务器内部错误',
        'message': str(e) if app.debug else '请稍后再试'
    }), 500

# ==================== 优雅退出 ====================
def signal_handler(sig, frame):
    logger.info("收到退出信号，正在关闭...")
    if predictor:
        predictor.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ==================== 启动 ====================
if __name__ == '__main__':
    try:
        from waitress import serve
        logger.info("使用 waitress 生产级服务器启动...")
        serve(app, host=HOST, port=PORT)
    except ImportError:
        logger.info("使用 Flask 内置服务器启动（仅开发）...")
        app.run(host=HOST, port=PORT, debug=False)