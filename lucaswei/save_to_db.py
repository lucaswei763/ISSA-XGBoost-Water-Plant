import os
import pandas as pd
import sqlite3

# 1. 设定您的文件夹路径 (请修改为您实际的路径)
folder_path = 'DataBase/21年-26年药耗、原水数据'

# 读取刚刚合并好的大表
excel_path = os.path.join(folder_path, '最终合并数据大表.xlsx')

print("正在读取合并大表...\n")
try:
    df = pd.read_excel(excel_path, engine='openpyxl')
except FileNotFoundError:
    print(f"❌ 找不到文件：{excel_path}，请确认路径是否正确。")
    exit()

# 2. 清理表头名称 (非常重要)
# 数据库的列名最好不要有换行符和空格，这里自动帮您把 "\n" 和空格删掉
print("正在优化数据库字段名...")
clean_columns = []
for col in df.columns:
    new_col = str(col).replace('\n', '').replace('\r', '').replace(' ', '')
    clean_columns.append(new_col)
df.columns = clean_columns

# 3. 创建并连接到 SQLite 数据库
# 如果这个文件不存在，Python 会自动帮您创建一个
db_path = os.path.join(folder_path, '水务数据中心.db')
conn = sqlite3.connect(db_path)

print(f"🔗 成功连接到数据库：{db_path}")

# 4. 将 DataFrame 一键写入数据库
# name='daily_records' 是我们在数据库里给这张表起的名字
# if_exists='replace' 表示如果表已经存在，就覆盖它；您也可以改成 'append' 表示追加数据
# index=False 表示不要把行号也存进数据库
df.to_sql(name='daily_records', con=conn, if_exists='replace', index=False)

# 5. 关闭数据库连接
conn.close()

print(f"\n🎉 完美搞定！一共 {len(df)} 条数据已成功存入数据库！")
print(f"数据库文件已保存在: {db_path}")