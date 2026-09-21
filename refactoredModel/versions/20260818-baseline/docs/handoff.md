# 水厂投矾量智能预测系统 - 交接文档

## 一、项目概述

### 1.1 系统名称
**水厂投矾量智能预测系统 v2.0**

### 1.2 系统功能
- 基于 XGBoost 模型的投矾量预测
- 特征贡献度分析（SHAP）
- 变频泵频率计算
- 异常工况检测
- 趋势预警

### 1.3 技术栈
| 模块 | 技术 | 版本要求 |
|------|------|----------|
| 语言 | Python | >= 3.9 |
| GUI | CustomTkinter | >= 5.0 |
| 模型 | XGBoost | >= 1.7 |
| 可解释性 | SHAP | >= 0.52 |
| 数据处理 | Pandas/Numpy/SciPy | 最新 |
| 打包 | PyInstaller | >= 6.0 |

---

## 二、项目结构

```
refactoredModel/
├── ui_app.py              # 桌面GUI主程序
├── predictor_service.py   # 预测服务核心逻辑
├── data_loader.py         # 数据加载与特征工程
├── app.py                 # Flask Web服务（可选）
├── utils.py               # 工具函数
├── build_exe.py           # EXE打包脚本
├── build_exe_debug.py     # 调试版打包脚本
├── retrain_anomaly.py     # 异常检测器重训练脚本
├── test_prediction.py     # 预测测试脚本
├── requirements.txt       # 依赖列表
├── data/                  # 数据目录
│   └── runtime_records.json  # 运行时记录
├── models/                # 模型文件目录
│   ├── feature_defaults.json # 特征默认值
│   ├── feature_ranges.json   # 特征统计范围
│   ├── feature_stats.json    # 特征统计信息
│   └── metadata.json         # 模型元数据
└── dist/                  # 打包输出目录
    └── WaterPredictor.exe # 最终可执行文件
```

---

## 三、核心文件说明

### 3.1 ui_app.py - 桌面GUI

**功能**：用户交互界面，包含输入面板、预测结果展示、SHAP贡献度分析、泵频率计算

**关键组件**：
- 输入面板：浑浊度、原水量、温度、氨氮、pH值、冲程
- 结果展示：日投矾量、单位投加量、小时投加量、纯矾/稀释体积
- SHAP解释条：显示各特征对预测结果的贡献度
- 置信度徽章：基于异常检测的置信度评估
- 趋势预警：对比历史数据的趋势变化

**入口**：直接运行 `python ui_app.py`

### 3.2 predictor_service.py - 预测服务

**功能**：核心预测逻辑，包含模型加载、特征补全、SHAP解释、异常检测

**关键方法**：
| 方法 | 功能 |
|------|------|
| `__init__()` | 加载模型、初始化SHAP解释器 |
| `predict(input_data)` | 执行预测 |
| `explain(input_data)` | 预测+SHAP解释+置信度+趋势预警 |
| `_augment_input(input_data)` | 特征补全（缺失值填充、滞后特征） |
| `_check_confidence(input_data)` | 异常检测与置信度评估 |
| `_check_trend(pred, date_str)` | 趋势预警分析 |

### 3.3 data_loader.py - 数据加载器

**功能**：数据读取、清洗、特征工程

**特征配置**：
```python
BASE_FEATURES_MAP = {
    '浑浊度_0点': '浑浊度_0点',
    '原水量_Km3_hour': '原水量_Km3_hour',
    '温度_C': '温度_C',
    '氨氮_mg_per_L': '氨氮_mg_per_L',
    'pH值': 'pH值',
}

DERIVED_FEATURES = {
    '浊度_氨氮_比值': 浑浊度 / 氨氮,
    '浊度_异常_幅度': 与7日均值的偏差,
    '浊度_氨氮_偏差': 浊度与氨氮趋势偏差,
}
```

---

## 四、单位体系

### 4.1 输入单位

| 特征 | 单位 | 范围 | 说明 |
|------|------|------|------|
| 浑浊度_0点 | NTU | 0-100 | 0点测量值 |
| 原水量_Km3_hour | Km³/h | 5-20 | **每小时水量** |
| 温度_C | °C | -10~50 | 9点测量值 |
| 氨氮_mg_per_L | mg/L | 0.02-0.86 | 9点测量值 |
| pH值 | - | 6-8 | 9点测量值 |
| 冲程 | % | 0-100 | 泵频率百分比 |

### 4.2 输出单位

| 指标 | 单位 | 说明 |
|------|------|------|
| 日投矾量 | kg/d | 模型预测结果 |
| 单位投加量 | kg/km³ | 日投矾量 / 日原水量 |
| 小时投加量 | kg/h | 日投矾量 / 24 |
| 纯矾体积 | L/h | 小时投加量 / 1.3 (密度) |
| 稀释后体积 | L/h | 纯矾体积 × 稀释倍数 |
| 泵频率 | Hz | 基于目标流量计算 |

### 4.3 单位转换关系

```
原水量_Km3（日值） ÷ 24 = 原水量_Km3_hour（小时值）
原水量_Km3_hour × 24 = 原水量_Km3（用于模型）

日投矾量 ÷ 日原水量 = 单位投加量 (kg/km³)
日投矾量 ÷ 24 = 小时投加量 (kg/h)
小时投加量 ÷ 1.3 = 纯矾体积 (L/h)
```

---

## 五、数据流程

### 5.1 预测流程

```
用户输入 → _collect_input_data() → _augment_input() → predict()/explain() → 结果展示
              (UI收集)            (特征补全)       (模型预测)      (SHAP分析)
```

### 5.2 特征补全逻辑

1. 基础特征缺失 → 使用 [feature_defaults.json](models/feature_defaults.json) 中的中位数
2. 滞后特征(lag1/lag2) → 使用历史数据或默认值
3. 原水量转换 → `原水量_Km3_hour × 24 = 原水量_Km3`（供模型使用）

### 5.3 异常检测

- 使用 `EllipticEnvelope`（椭圆包络）算法
- 特征：浑浊度、原水量、温度、氨氮、pH值（5个）
- 阈值：基于训练数据的2%异常分位数

---

## 六、模型配置

### 6.1 模型文件

| 文件 | 用途 |
|------|------|
| `metadata.json` | 模型版本、特征列表、评估指标 |
| `feature_defaults.json` | 特征缺失时的默认值 |
| `feature_ranges.json` | 特征统计范围（用于越界警告） |
| `feature_stats.json` | 特征统计信息 |

### 6.2 模型特征列表

```python
['浑浊度_0点', '原水量_Km3', '温度_C', '氨氮_mg_per_L', 'pH值', 
 '库区水位_m', '消耗电_kWh', '供水量_Km3', 'lag1', 'lag2']
```

**注意**：模型训练时包含10个特征，但实际预测时仅使用用户输入的5个核心特征，其余特征使用默认值补全。

### 6.3 模型评估指标

评估指标随每次重训变化，**不随仓库分发**。运行时从 `models/metadata.json` 读取，
需要完整评估报告请向项目维护者获取。

---

## 七、打包与部署

### 7.1 打包脚本

使用 [build_exe.py](build_exe.py) 进行打包：

```bash
python build_exe.py
```

### 7.2 打包配置要点

| 配置项 | 说明 |
|--------|------|
| `--onefile` | 单文件模式 |
| `--noconsole` | 无控制台窗口 |
| `--collect-all xgboost` | 完整收集XGBoost依赖 |
| `--hidden-import shap` | 显式导入SHAP |
| `--upx-exclude xgboost.dll` | **禁止UPX压缩xgboost.dll**（压缩后会损坏） |
| `--upx-exclude vcomp140.dll` | 禁止UPX压缩vcomp140.dll |
| `--add-data` | 包含models和data目录 |

### 7.3 已知打包问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| UPX压缩损坏xgboost.dll | xgboost.dll不兼容UPX压缩 | 添加`--upx-exclude xgboost.dll` |
| SHAP不可用 | 打包时排除了numba | 移除`--exclude-module numba` |
| 权限错误 | PyInstaller默认缓存目录无权限 | 设置`PYINSTALLER_CONFIG_DIR`环境变量 |

### 7.4 运行要求

- Windows 10/11 64位系统
- 无需安装Python环境
- 首次运行需等待模型加载（约5-10秒）

---

## 八、常见问题与解决方案

### 8.1 EXE无法启动

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 双击无反应 | UPX压缩损坏xgboost.dll | 重新打包，添加`--upx-exclude xgboost.dll` |
| 闪退 | SHAP/numba缺失 | 重新打包，包含numba |
| 提示模型文件缺失 | 运行目录不对 | 确保EXE与models目录同层级 |

### 8.2 预测结果异常

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 预测值为None | 模型加载失败或特征不匹配 | 检查模型文件完整性 |
| 置信度为low | 输入超出历史范围 | 检查输入值是否合理 |
| "为什么是这个值"无内容 | SHAP未初始化 | 检查numba是否安装并包含在打包中 |
| 单位投加量异常 | 原水量单位错误 | 确认输入的是小时值(Km³/h) |

### 8.3 数据更新

如需更新训练数据：
1. 将新Excel数据放入`data/`目录
2. 运行`retrain_anomaly.py`重新训练异常检测器
3. 重新打包EXE

---

## 九、维护说明

### 9.1 特征默认值更新

`models/feature_defaults.json` 由 [retrain_anomaly.py](../src/retrain_anomaly.py)
从训练数据自动生成，其中的中位数值属于业务数据，**不随仓库分发**。
需要更新时在本地重新运行该脚本即可。

### 9.2 特征范围更新

`models/feature_ranges.json` 同样由 [retrain_anomaly.py](../src/retrain_anomaly.py)
生成，统计值属于业务数据，**不随仓库分发**，需要时在本地重新生成。

### 9.3 异常检测器重训练

运行 [retrain_anomaly.py](retrain_anomaly.py)：
```bash
python retrain_anomaly.py
```

---

## 十、版本变更记录

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| v2.0 | 2026-07-25 | 统一原水量单位为Km³/h；修复SHAP贡献度展示；修复单位投加量计算；重新打包EXE |
| v1.0 | 初始版本 | 基础预测功能 |

---

## 十一、联系方式

如有问题请联系开发人员。
