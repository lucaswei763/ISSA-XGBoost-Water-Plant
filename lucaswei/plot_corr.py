import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 设定您的文件夹路径 (请修改为您实际的路径)
folder_path = 'DataBase/21年-26年药耗、原水数据'

# 读取上一步生成的分析结果
file_path = os.path.join(folder_path, '投矾量相关性分析结果.xlsx')

print("正在读取数据，准备绘制可视化图片...")
try:
    df = pd.read_excel(file_path, engine='openpyxl')
except FileNotFoundError:
    print(f"❌ 找不到文件：{file_path}")
    exit()

# 2. 解决 Mac 系统下图表中文显示为方块的问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac专属中文字体
plt.rcParams['axes.unicode_minus'] = False              # 正常显示负号

# 3. 创建画布，设置图片大小 (宽12，高8)
plt.figure(figsize=(12, 8))

# 提取数据
factors = df['影响因素'].tolist()
pearson = df['皮尔逊(Pearson)相关系数'].fillna(0).tolist()
spearman = df['斯皮尔曼(Spearman)相关系数'].fillna(0).tolist()

# 设置柱子的位置和宽度
x = np.arange(len(factors))
width = 0.35

# 4. 绘制双柱状图
# 皮尔逊用天蓝色，斯皮尔曼用珊瑚红
plt.bar(x - width/2, pearson, width, label='皮尔逊 (Pearson)', color='#5D9CEC')
plt.bar(x + width/2, spearman, width, label='斯皮尔曼 (Spearman)', color='#FC6E51')

# 5. 美化图表
plt.ylabel('相关系数值', fontsize=12)
plt.title('各因素与【投矾量】的相关系数对比', fontsize=16, pad=20)

# 设置X轴标签，倾斜45度防止文字重叠
plt.xticks(x, factors, rotation=45, ha='right', fontsize=11)

# 添加一条 y=0 的基准线，方便看清正相关和负相关
plt.axhline(0, color='gray', linewidth=1, linestyle='--')

# 显示图例
plt.legend(fontsize=12)

# 自动调整布局，防止底部的文字被画面截断
plt.tight_layout()

# 6. 保存图片到电脑
output_path = os.path.join(folder_path, '投矾量相关性可视化.png')
# dpi=300 表示保存为高清图
plt.savefig(output_path, dpi=300)
print(f"🎉 绘图完成！高清可视化图片已保存至: {output_path}")