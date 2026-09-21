"""训练与调参参数加载器。

超参数属于本地配置，按项目约定不纳入版本库。
请在 models/ 目录下准备 train_params.json，结构如下：

{
  "baseline":   { "n_estimators": 0, "max_depth": 0, "learning_rate": 0.0,
                  "subsample": 0.0, "reg_lambda": 0.0, "min_child_weight": 0 },
  "candidates": { "<候选名称>": { "n_estimators": 0, "max_depth": 0, "...": 0 } }
}

文件缺失时直接报错，避免用猜测值训练出无法复现的模型。
random_state 与 n_jobs 由本模块统一注入，不写入 JSON。
"""

import json
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
PARAMS_FILE = os.path.join(MODEL_DIR, 'train_params.json')

_FIXED = {'random_state': 42, 'n_jobs': -1}

_MISSING_HINT = (
    '该文件包含项目超参数，按约定不随仓库分发。'
    '请向项目维护者获取后放入 models/ 目录。'
)


def _load():
    if not os.path.exists(PARAMS_FILE):
        raise FileNotFoundError(f'缺少本地训练参数文件: {PARAMS_FILE}\n{_MISSING_HINT}')
    with open(PARAMS_FILE, encoding='utf-8') as f:
        return json.load(f)


def load_train_params(section='baseline'):
    """读取一组训练超参数。"""
    data = _load()
    if section not in data:
        raise KeyError(f'train_params.json 缺少 "{section}" 段')
    params = dict(data[section])
    params.update(_FIXED)
    return params


def load_candidates():
    """读取候选超参数字典，用于在测试集上比较后选优。"""
    data = _load()
    candidates = data.get('candidates')
    if not candidates:
        raise KeyError('train_params.json 缺少 "candidates" 段')
    return {name: {**params, **_FIXED} for name, params in candidates.items()}
