import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

# --- 1. macOS 环境配置 ---
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Heiti TC')

# --- 2. 核心清洗函数：将“脏字符串”转为数字 ---
def clean_to_float(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int, float)): return float(val)
    # 正则提取：只保留数字和小数点
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
    return float(nums[0]) if nums else np.nan

# --- 3. 加载数据 ---
conn = sqlite3.connect('water_plant_final.db')
df = pd.read_sql_query("SELECT * FROM filled_data", conn)
conn.close()

target_col = '矾\n（kg/Km³）'

# --- 4. 强制转换所有疑似特征列 ---
# 找出所有列，排除掉日期和目标列
exclude = ['日期', 'date', 'Unnamed', 'file_type', '耗用矾量', 'target_dosage']
potential_features = [c for c in df.columns if not any(k in c for k in exclude) and c != target_col]

for col in potential_features:
    df[col] = df[col].apply(clean_to_float)
    # 转换后如果还有空值，用中位数填充（保证计算不中断）
    df[col] = df[col].fillna(df[col].median())

# --- 5. 计算相关系数 ---
pearson = df[potential_features + [target_col]].corr(method='pearson')[target_col].drop(target_col)
spearman = df[potential_features + [target_col]].corr(method='spearman')[target_col].drop(target_col)

# 汇总并排序
corr_summary = pd.DataFrame({
    'Feature': spearman.index,
    'Pearson': pearson.values,
    'Spearman': spearman.values,
    'Abs_Spearman': spearman.abs().values
}).sort_values(by='Abs_Spearman', ascending=False)

# --- 6. 绘图 (带颜色分级) ---
def get_color(val):
    v = abs(val)
    if v >= 0.6: return '#e74c3c'  # 红
    elif v >= 0.3: return '#f1c40f' # 黄
    else: return '#95a5a6'           # 灰

plt.figure(figsize=(16, 10))
ax = sns.barplot(data=corr_summary.melt(id_vars='Feature', value_vars=['Pearson', 'Spearman']),
                 x='Feature', y='value', hue='variable')

# 涂色逻辑
n = len(corr_summary)
for i, bar in enumerate(ax.patches):
    feat_idx = i % n
    val = corr_summary.iloc[feat_idx]['Spearman']
    bar.set_facecolor(get_color(val))
    if i < n: bar.set_alpha(0.4) # Pearson 浅色

plt.title('全量维度(含隐藏化学指标)相关性分析', fontsize=18)
plt.xticks(rotation=45, ha='right')
plt.legend(handles=[
    mpatches.Patch(color='#e74c3c', label='显著影响 (|r|≥0.6)'),
    mpatches.Patch(color='#f1c40f', label='一半影响 (0.3≤|r|<0.6)'),
    mpatches.Patch(color='#95a5a6', label='没什么影响 (|r|<0.3)')
], loc='upper right')
plt.tight_layout()
plt.show()

# 打印结果看看“新秀”排名
print(corr_summary[['Feature', 'Spearman']].head(15))