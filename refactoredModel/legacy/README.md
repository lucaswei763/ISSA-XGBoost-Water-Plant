# legacy —— 归档的早期实现

这里的文件原本散放在 `refactoredModel/` 顶层，2026-09-21 整体移到本目录，
**内容一字未改**，只是位置调整，让顶层只保留当前版本的入口。

来源：lucaswei763 于 2026-06-06 的提交 `f0b8855`
（refactor: 更新代码，新加入残差神经网络预测模型）与 `19bdca4`。

## 里面的东西

| 文件 | 说明 |
| --- | --- |
| `train_hybrid.py` | 残差神经网络 / 混合模型训练 |
| `utils.py` | 特征工程、`ResNetRegressor`、数据加载 |
| `issa_optimizer.py` | ISSA 优化 XGBoost |
| `build_database.py` | Excel 转 SQLite |
| `app.py` / `ui_app.py` / `origin_ui_app.py` | Web 服务与桌面界面 |
| `cli_app.py`、`LinuxCLI/` | 命令行版本，`LinuxCLI/` 是自包含副本 |
| `compare_models.py` / `compare_trained_models.py` | 模型对比 |
| `generate_report.py` / `generate_report_metrics.py` | 报告生成 |
| `mlp_gd_train.py` / `xgb_issa_train.py` | 早期训练脚本 |
| `test_predictor.py` | 预测自测 |
| `requirements.txt` | 这套实现自己的依赖清单 |

## 使用注意

- 这套代码与 `versions/` 下的三个快照**没有任何重复内容**，不能用那边的文件替换。
- `LinuxCLI/utils.py` 依赖当前工作目录导入 `build_database`，
  所以要在本目录（`legacy/`）下运行，相对位置才和归档前一致。
- 新代码不要往这里加，请写到 `../versions/v12.2-final/src/`。
- 本目录代码属于 lucaswei763，如需删除请先与他确认。
