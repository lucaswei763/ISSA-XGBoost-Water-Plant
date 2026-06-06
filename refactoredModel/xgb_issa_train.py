"""
文件名称：refactoredModel/xgb_issa_train.py
所属类别：重构核心生产代码 (Refactored Core Production)

功能描述：
    独立训练并优化 ISSA-XGBoost 模型的主程序流水线。
    执行步骤包括：
    1. 加载并预处理水厂工况历史数据集，导出模型专用的转换特征列表和 imputer 缺省值；
    2. 实例化并调用 ISSA 麻雀寻优算法（32 个种群粒子，迭代 200 次）搜索最优的 6 维 XGBoost 超参数；
    3. 使用最优的参数子集拟合并训练最终的 XGBRegressor，绘制训练与测试 RMSE 损失下降曲线；
    4. 将模型保存为 `models/xgboost/best_model.pkl` 并调用 `utils.evaluate_and_plot` 生成按年连续的回归预测对比图。

运行与使用方法：
    直接在控制台启动模型自动参数寻优与训练：
    python xgb_issa_train.py

调用与依赖关系：
    - 导入并依赖 `utils.py` 的数据处理、画图评估模块。
    - 导入并调用 `issa_optimizer.ISSA_XGBoost` 进行粒子群参数寻优。
    - 训练产出物被保存至 `models/xgboost/best_model.pkl`，供 `ui_app.py`、`app.py`、`cli_app.py` 加载和提供决策建议。
"""
import time
from xgboost import XGBRegressor
import joblib
from utils import load_and_preprocess_data, evaluate_and_plot, setup_logger_and_dir
from issa_optimizer import ISSA_XGBoost

def main():
    # 初始化按天创建的日志和图表目录，并接管标准输出
    run_dir = setup_logger_and_dir("xgb_train")
    
    print("=" * 40)
    print("🌟 ISSA-XGBoost 总投矾量预测模型 🌟")
    print("=" * 40)
    
    # 1. 使用优化的通用数据处理函数
    print("\n🔍 读取并处理数据集中...")
    try:
        X_train_scaled, X_test_scaled, y_train, y_test, X_full_scaled, y_full, full_dates, water_test, water_full = load_and_preprocess_data(model_dir='models/xgboost')
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 2. 优化运行
    print("\n🚀 启动 ISSA-XGBoost 优化过程...")
    print("参数设定: 种群数 (pop_size) = 32, 迭代次数 (max_iter) = 200")
    
    start_time = time.time()
    issa = ISSA_XGBoost(pop_size=32, max_iter=200)
    best_params = issa.optimize(X_train_scaled, y_train)
    
    print("\n🏆 ISSA 寻优得到的最优参数:")
    for k, v in best_params.items():
        print(f"   {k}: {v}")
        
    print(f"\n⏳ 优化耗时: {(time.time() - start_time):.2f} 秒")

    # 3. 训练最终模型
    print("\n⚙️ 正在使用最优参数训练最终 XGBoost 模型...")
    final_model = XGBRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
        verbose=False
    )
    
    try:
        import matplotlib.pyplot as plt
        import os
        results = final_model.evals_result()
        epochs = len(results['validation_0']['rmse'])
        x_axis = range(0, epochs)
        plt.figure(figsize=(10, 5))
        plt.plot(x_axis, results['validation_0']['rmse'], label='Train RMSE')
        plt.plot(x_axis, results['validation_1']['rmse'], label='Validation RMSE')
        plt.legend()
        plt.title('XGBoost Training and Validation RMSE')
        plt.ylabel('RMSE Loss')
        plt.xlabel('Epochs / Trees')
        plt.grid(True)
        plt.savefig(os.path.join(run_dir, 'xgb_loss_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"⚠️ 无法保存 XGBoost Loss 曲线: {e}")
        
    # 保存最终训练好的模型到 models/xgboost 目录
    import os
    os.makedirs('models/xgboost', exist_ok=True)
    joblib.dump(final_model, 'models/xgboost/best_model.pkl')
    print("✅ XGBoost 模型及环境变量已成功打包保存至 models/xgboost 目录，现在可以直接打开运行 ui_app！")
    
    # 4. 测试集评估与绘制全量按年份连续图表
    y_pred = final_model.predict(X_test_scaled)     # 仅用于评估测算
    y_full_pred = final_model.predict(X_full_scaled) # 用于全景绘图
    
    # 按年份导出存放在对应日期的 run_dir 目录下
    evaluate_and_plot(y_test, y_pred, y_full, y_full_pred, full_dates, model_name="ISSA-XGBoost", plot_dir=run_dir, water_test=water_test, water_full=water_full)

if __name__ == "__main__":
    main()
