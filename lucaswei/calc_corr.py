import os
import pandas as pd

# 1. 设定您的文件夹路径 (请修改为您实际的路径)
folder_path = 'DataBase/21年-26年药耗、原水数据'

# 读取刚刚合并好的大表
file_path = os.path.join(folder_path, '最终合并数据大表.xlsx')

print("正在读取合并大表，准备计算相关系数...\n")
try:
    df = pd.read_excel(file_path, engine='openpyxl')
except FileNotFoundError:
    print(f"❌ 找不到文件：{file_path}，请确认路径是否正确。")
    exit()

# 2. 自动寻找代表“投矾量”的列
target_col = None
for col in df.columns:
    if '矾' in str(col):  # 只要列名里包含“矾”字就锁定它（如果有多列，可以改成更精确的比如 '耗用矾量'）
        target_col = col
        break

if not target_col:
    print("❌ 在表格中没有找到包含“矾”字的列名，请检查大表的列名！")
    exit()

print(f"🎯 成功锁定目标变量：【{target_col}】")

# 3. 数据清洗：强制将所有待分析的列转换为数字
# 去掉“日期”列，因为它不是数字，不能参与相关性计算
cols_to_analyze = [col for col in df.columns if col != '日期']

for col in cols_to_analyze:
    # errors='coerce' 是一招“魔法”：它会把里面的 '/'、'\'、'空' 等乱七八糟的文字全部变成空白空值(NaN)
    # 这样它们就不会干扰数学计算了
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("🔄 数据已完成数值化清洗，正在计算皮尔逊和斯皮尔曼相关系数...")

# 4. 计算皮尔逊 (Pearson) 和 斯皮尔曼 (Spearman) 相关系数
# pandas 默认会自动忽略空值 (NaN) 进行计算
pearson_corr = df[cols_to_analyze].corr(method='pearson')[target_col]
spearman_corr = df[cols_to_analyze].corr(method='spearman')[target_col]

# 5. 把结果整理成一个漂亮的新表格
result_df = pd.DataFrame({
    '影响因素': pearson_corr.index,
    '皮尔逊(Pearson)相关系数': pearson_corr.values,
    '斯皮尔曼(Spearman)相关系数': spearman_corr.values
})

# 剔除自己和自己的相关系数（因为投矾量和投矾量自己的相关系数肯定是 1.0，没有意义）
result_df = result_df[result_df['影响因素'] != target_col]

# 为了直观，我们按照皮尔逊相关系数的“绝对值”从大到小排序，这样最相关的因素会排在最上面
result_df['绝对值排序辅助'] = result_df['皮尔逊(Pearson)相关系数'].abs()
result_df = result_df.sort_values(by='绝对值排序辅助', ascending=False).drop(columns=['绝对值排序辅助'])

# 6. 打印在控制台给您预览
print(f"\n📊 【{target_col}】与各因素的相关性结果：")
# 打印格式化，不要省略行
pd.set_option('display.max_rows', None)
print(result_df.to_string(index=False))

# 7. 保存结果为新的 Excel
output_path = os.path.join(folder_path, '投矾量相关性分析结果.xlsx')
result_df.to_excel(output_path, index=False, engine='openpyxl')
print(f"\n🎉 计算完成！分析报告已保存至: {output_path}")