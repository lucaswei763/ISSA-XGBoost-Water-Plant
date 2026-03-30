import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- 1. 中文显示与环境配置 ---
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Heiti TC')

# --- 2. 颜色映射函数 ---
def get_influence_color(val):
    v = abs(val)
    if v >= 0.6: return '#e74c3c'  # 红色: 显著影响
    elif v >= 0.3: return '#f1c40f' # 黄色: 一半影响
    else: return '#95a5a6'           # 灰色: 没什么影响

# --- 3. 加载并筛选全量维度 ---
conn = sqlite3.connect('water_plant_final.db')
df = pd.read_sql_query("SELECT * FROM filled_data", conn)
conn.close()

# 定义目标列（矾浓度）
target_col = '矾\n（kg/Km³）'

# 自动化剔除：日期、类型、空列、以及“数据泄漏”列（耗用矾量kg）
drop_keywords = ['日期', 'date', 'Unnamed', 'file_type', '耗用矾量\n（kg）']
cols_to_analyze = [c for c in df.columns if not any(k in c for k in drop_keywords) and c != target_col]

# 只提取数值型数据进行计算
numeric_df = df[cols_to_analyze + [target_col]].select_dtypes(include=[np.number])

# --- 4. 计算相关系数并排序 ---
pearson = numeric_df.corr(method='pearson')[target_col].drop(target_col)
spearman = numeric_df.corr(method='spearman')[target_col].drop(target_col)

# 创建结果表并按 Spearman 绝对值降序排列
corr_summary = pd.DataFrame({
    'Feature': spearman.index,
    'Pearson': pearson.values,
    'Spearman': spearman.values,
    'Abs_Spearman': spearman.abs().values
}).sort_values(by='Abs_Spearman', ascending=False)

# 转换格式供 Seaborn 使用
plot_data = corr_summary.melt(id_vars='Feature', value_vars=['Pearson', 'Spearman'],
                              var_name='Method', value_name='Coefficient')

# --- 5. 绘图 ---
plt.figure(figsize=(16, 9))
ax = sns.barplot(data=plot_data, x='Feature', y='Coefficient', hue='Method')

# 手动给柱子涂色
n_feats = len(corr_summary)
for i, bar in enumerate(ax.patches):
    feat_idx = i % n_feats
    corr_val = corr_summary.iloc[feat_idx]['Spearman']
    color = get_influence_color(corr_val)
    bar.set_facecolor(color)
    # Pearson 柱子设为半透明，Spearman 设为全色以示区别
    if i < n_feats: bar.set_alpha(0.4)
    else: bar.set_alpha(1.0)

# --- 6. 辅助线与图例 ---
plt.axhline(0.6, color='#e74c3c', linestyle='--', alpha=0.3)
plt.axhline(0.3, color='#f1c40f', linestyle='--', alpha=0.3)
plt.axhline(0, color='black', linewidth=0.8)

# 构造分类图例
red_p = mpatches.Patch(color='#e74c3c', label='显著影响 (|r| ≥ 0.6)')
gold_p = mpatches.Patch(color='#f1c40f', label='一半影响 (0.3 ≤ |r| < 0.6)')
grey_p = mpatches.Patch(color='#95a5a6', label='没什么影响 (|r| < 0.3)')
plt.legend(handles=[red_p, gold_p, grey_p], loc='upper right', title="Spearman 影响分级")

plt.title('全量维度与矾消耗量(浓度)相关性排序分析 (Pearson vs Spearman)', fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.ylabel('相关系数数值', fontsize=12)
plt.tight_layout()

# 打印文本报告供参考
print("\n--- 维度影响分析明细表 ---")
corr_summary['Influence'] = corr_summary['Spearman'].apply(lambda x:
    '显著影响' if abs(x)>=0.6 else ('一半影响' if abs(x)>=0.3 else '没什么影响'))
print(corr_summary[['Feature', 'Pearson', 'Spearman', 'Influence']])

plt.show()