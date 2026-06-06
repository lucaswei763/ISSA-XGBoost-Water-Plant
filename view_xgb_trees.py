"""
文件名称：view_xgb_trees.py
所属类别：根目录辅助工具 (Root Level Utilities)

功能描述：
    用于将已训练的 ISSA-XGBoost 模型中的每棵集成决策树的内部分支规则以可视文本形式导出的调试工具。
    主要职责：
    1. 自动定位加载最新的 xgboost pkl 模型文件；
    2. 使用 Booster 提取并打印出第一棵树（Tree 0）的分支逻辑（如：判断特征阈值、左右子树分配和叶节点输出值）；
    3. 将所有决策树（通常有数百棵）完整的分支路径以可读文本形式保存到本地文件 `xgb_tree_structures.txt` 中。

运行与使用方法：
    直接运行此文件开始树结构分析导出：
    python view_xgb_trees.py

调用与依赖关系：
    - 被开发人员用于模型可解释性分析，提取和审阅模型对水质因子做决策时所用的多维阈值边界。
    - 依赖于 `xgboost` 及其 C 核心 API `get_booster().get_dump()`。
"""
import os
import joblib
import json

# 1. 加载你在 xgb_issa_train.py 中保存的最优模型
model_path = 'models/xgboost/best_model.pkl' if os.path.exists('models/xgboost/best_model.pkl') else 'models/best_model.pkl'
model = joblib.load(model_path)

# 2. 将所有树的分支逻辑导出为文本列表（每棵树是一个字符串）
# fmap 可以绑定特征名称，使导出的变量名不是 f0, f1 而是具体的“原水浊度”等
trees_dump = model.get_booster().get_dump(with_stats=True)

# 3. 打印第一棵树（Tree 0）的所有分支节点来看看
print("🌳 --- 正在查看第 1 棵决策树的分支节点结构 --- 🌳")
print(trees_dump[0])

# 4. 如果你想把所有树存成文本文件方便慢慢研究，可以取消下面三行的注释：
with open('xgb_tree_structures.txt', 'w', encoding='utf-8') as f:
    for i, tree in enumerate(trees_dump):
        f.write(f"\n\n=========== 第 {i} 棵树 ===========\n{tree}")