# refactoredModel 版本归档

本目录归档水厂投矾量预测系统 `refactoredModel` 的三个源码快照，便于团队对照演进过程、定位历史行为差异。

## 版本

| 目录 | 状态 | 说明 |
| --- | --- | --- |
| `20260718-original/` | 历史快照 | 早期版本，包含 LSTM、XGBoost、ISSA、MLP 等多条技术路线与报告脚本 |
| `20260818-baseline/` | 历史快照 | 重构后的桌面版基线，含原始预测服务与 UI |
| `v12.2-final/` | 当前版本 | 单位投矾量模型 + 单位滞后特征，为交付版本 |

## 目录约定

每个版本内部统一下列结构：

```text
<version>/
├── src/         运行、训练、验证代码
├── docs/        接口与架构说明
└── packaging/   打包脚本与 PyInstaller 配置
```

`packaging/` 不叫 `build/`，是为了避开 `refactoredModel/.gitignore` 中针对 PyInstaller 产物目录的 `build/` 忽略规则。

## 模型设计要点（v12.2）

- 预测目标是**单位投矾量**（耗用矾量 / 原水量），预测时再乘回原水量得到总量。
- 单位投矾量与水量无关，因此「水量上升 → 投矾量上升」是严格成立的，不会与业务常识冲突。
- 滞后特征使用**单位滞后** `lag1_unit` / `lag2_unit`，与目标同量纲，避免把水量信息混进自回归项。
- 特征中不含任何水量列，保证总量与水量严格正比。

## 本地资产（不入库）

以下内容按项目约定不纳入版本库，需要由维护者单独分发：

| 路径 | 内容 |
| --- | --- |
| `data/` | 原始水厂数据（Excel）与运行时记录 |
| `models/` | 训练好的模型、标准化器、插补器、特征选择结果 |
| `models/feature_defaults.json` | 特征默认值 |
| `models/feature_ranges.json` | 特征统计范围 |
| `models/metadata.json` | 模型版本、特征列表与评估指标 |
| `models/train_params.json` | 训练与调参超参数 |
| `models/shap_*.pkl` | SHAP 基准数据 |
| `models/prediction_bounds.pkl` | 预测区间 |

上述文件缺失时程序仍可启动，但预测精度不可用；训练脚本会直接报错并要求补齐配置。

## 运行方式

```bash
pip install -r src/requirements.txt

# 桌面界面
python src/ui_app.py

# Web 服务（可选）
python src/app.py

# 端到端验证（需要本地 data/ 与 models/）
python src/verify_v122.py
```

## 数据处理约定

- 原水量在数据层统一为日值 `原水量_Km3`，界面输入为小时值 `原水量_Km3_hour`，按 ×24 换算。
- 缺失值按「前向填充 → 后向填充 → 中位数」顺序处理。
- 目标列过滤 `0 < 耗用矾量_kg < 50000`。

## 打包

```bash
cd packaging
python build_exe.py
```

打包注意事项见 `20260818-baseline/docs/handoff.md` 的打包与部署章节。
