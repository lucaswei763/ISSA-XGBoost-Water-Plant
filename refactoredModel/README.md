# refactoredModel

## ⭐ 当前交付版本

**请直接使用 [`versions/v12.2-final/`](versions/v12.2-final/)。**

日常开发、交付、打包都以它为准。

| 目录 | 用途 |
| --- | --- |
| [`versions/v12.2-final/`](versions/v12.2-final/) | **当前版本**：单位投矾量模型 + 单位滞后特征 |
| [`versions/20260818-baseline/`](versions/20260818-baseline/) | 历史快照，仅供对照，不要在其上继续开发 |
| [`versions/20260718-original/`](versions/20260718-original/) | 历史快照，仅供对照，不要在其上继续开发 |
| [`LinuxCLI/`](LinuxCLI/) | 命令行版本，独立维护 |

版本之间的差异和运行方式见 [`versions/README.md`](versions/README.md)。

## 本目录顶层的脚本

本目录顶层的 `app.py`、`ui_app.py`、`predictor_service.py`、`train_hybrid.py` 等文件是更早的工作副本，
保留用于对照历史行为。**新代码请写到 `versions/v12.2-final/src/` 下**，不要再往顶层加文件。

## 本地资产

`data/` 与 `models/` 不入库，需要由维护者单独分发，否则只能运行界面、无法得到有效预测。
详见 [`versions/README.md`](versions/README.md) 的「本地资产」一节。
