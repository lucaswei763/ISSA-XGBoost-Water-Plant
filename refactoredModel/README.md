# refactoredModel

## ⭐ 当前交付版本

**请直接使用 [`versions/v12.2-final/`](versions/v12.2-final/)。**
日常开发、交付、打包都以它为准。

## 目录结构

```text
refactoredModel/
├── versions/                 版本归档
│   ├── v12.2-final/          ⭐ 当前交付版本
│   ├── 20260818-baseline/    历史快照
│   └── 20260718-original/    历史快照
├── LinuxCLI/                 命令行版本
└── 顶层 *.py 等脚本          早期重构版本
```

## 版本归档

| 目录 | 用途 |
| --- | --- |
| [`versions/v12.2-final/`](versions/v12.2-final/) | **当前版本**：单位投矾量模型 + 单位滞后特征 |
| [`versions/20260818-baseline/`](versions/20260818-baseline/) | 历史快照，仅供对照，不要在其上继续开发 |
| [`versions/20260718-original/`](versions/20260718-original/) | 历史快照，仅供对照，不要在其上继续开发 |

版本之间的差异和运行方式见 [`versions/README.md`](versions/README.md)。

## 关于本目录顶层的脚本

顶层这些文件**不是本次归档新加的，也不是可以清理的废文件**。

| 顶层条目 | 加入时间 · 提交 | 作者 |
| --- | --- | --- |
| `.gitignore` | 2026-04-14 · `7c36c9b` | dong2007-png |
| `app.py`、`ui_app.py`、`cli_app.py`、`train_hybrid.py`、`compare_models.py`、`compare_trained_models.py`、`generate_report.py`、`generate_report_metrics.py`、`issa_optimizer.py`、`mlp_gd_train.py`、`origin_ui_app.py`、`predictor_service.py`、`build_database.py`、`utils.py`、`xgb_issa_train.py`、`test_predictor.py`、`requirements.txt`、`LinuxCLI/` | 2026-06-06 · `f0b8855`、`19bdca4` | lucaswei763 |

补充说明：

- 它们来自 lucaswei763 的独立实现（提交说明为「refactor: 更新代码，新加入残差神经网络预测模型」），
  其中 `train_hybrid.py` 是残差神经网络 / 混合模型那条线的代码。
- 经内容哈希逐一比对，**顶层文件与 `versions/` 下的三个快照没有任何重复**，
  也就意味着它们既不是旧版本的拷贝，也不能被 `versions/` 里的文件替代。
- 需要清空或搬迁顶层目录时，请先与 lucaswei763 确认；这是他的代码。

## 新旧代码的摆放约定

新代码写到 `versions/v12.2-final/src/` 下，不要再往顶层加文件，
避免和上面的早期版本继续混在一起。

## 本地资产

`data/` 与 `models/` 不入库，需要由维护者单独分发，否则只能运行界面、无法得到有效预测。
详见 [`versions/README.md`](versions/README.md) 的「本地资产」一节。
