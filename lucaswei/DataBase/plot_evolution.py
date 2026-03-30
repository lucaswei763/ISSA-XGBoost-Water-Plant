import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 0. 环境配置 ---
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
current_dir = os.path.dirname(os.path.abspath(__file__))
history_path = os.path.join(current_dir, 'history.csv')
output_dir = os.path.join(current_dir, 'Analysis_Results')


def plot_model_evolution():
    if not os.path.exists(history_path):
        print("❌ 找不到 history.csv，请先运行几次 train.py！")
        return

    # 1. 读取历史数据
    df = pd.read_csv(history_path)
    if len(df) < 2:
        print("💡 数据点太少（至少需要2次记录），请多运行几次训练。")
        return

    # 2. 创建画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    epochs = range(1, len(df) + 1)

    # --- 图 1：R² 精度进化 ---
    ax1.plot(epochs, df['train_r2'], 'o-', label='训练集 R²', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, df['test_r2'], 's--', label='测试集 R²', color='#ff7f0e', linewidth=2)
    ax1.fill_between(epochs, df['train_r2'], df['test_r2'], color='gray', alpha=0.1, label='泛化间隙 (Gap)')
    ax1.set_ylabel('决定系数 R²')
    ax1.set_title('ISSA-XGBoost 模型精度进化趋势', fontsize=14)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- 图 2：核心正则化参数 Gamma 变化 ---
    ax2.plot(epochs, df['gamma'], 'D-', color='#2ca02c', label='正则化参数 (Gamma)', linewidth=2)
    ax2.set_ylabel('Gamma 值')
    ax2.set_xlabel('迭代训练次数 (Reinforced Rounds)')
    ax2.set_title('模型防御力 (Gamma) 演变轨迹', fontsize=14)

    # 在图上标注 Gamma 升高的意义
    for i, txt in enumerate(df['gamma']):
        ax2.annotate(f'{txt:.2f}', (epochs[i], df['gamma'][i]), textcoords="offset points", xytext=(0, 10), ha='center')

    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

    # 3. 保存结果
    save_path = os.path.join(output_dir, '6_model_evolution_trend.png')
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"✨ 进化趋势图已生成：{save_path}")


if __name__ == "__main__":
    plot_model_evolution()