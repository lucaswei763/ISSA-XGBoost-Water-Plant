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
├── legacy/                   早期实现归档（原顶层脚本，内容未改）
└── .gitignore
```

顶层只有这三项，不再堆放散落的脚本。

## 版本归档

| 目录 | 用途 |
| --- | --- |
| [`versions/v12.2-final/`](versions/v12.2-final/) | **当前版本**：单位投矾量模型 + 单位滞后特征 |
| [`versions/20260818-baseline/`](versions/20260818-baseline/) | 历史快照，仅供对照，不要在其上继续开发 |
| [`versions/20260718-original/`](versions/20260718-original/) | 历史快照，仅供对照，不要在其上继续开发 |

版本之间的差异和运行方式见 [`versions/README.md`](versions/README.md)。

## legacy/ —— 早期实现归档

顶层原来散放着 18 个条目（`app.py`、`ui_app.py`、`cli_app.py`、`train_hybrid.py`、
`compare_models.py`、`issa_optimizer.py`、`LinuxCLI/` 等），
已于 2026-09-21 整体移入 [`legacy/`](legacy/)，**内容一字未改**。

这些文件来自 lucaswei763 的独立实现，经过内容哈希比对，
**与 `versions/` 下的三个快照没有任何重复内容**，所以不是旧版本的拷贝，
不要当成垃圾清理。清理前请先与他确认。

详见 [`legacy/README.md`](legacy/README.md)。

## 新旧代码的摆放约定

新代码写到 `versions/v12.2-final/src/` 下，不要再往顶层加文件，
避免和归档的早期实现继续混在一起。

## 本地资产

`data/` 与 `models/` 不入库，需要由维护者单独分发，否则只能运行界面、无法得到有效预测。
详见 [`versions/README.md`](versions/README.md) 的「本地资产」一节。
