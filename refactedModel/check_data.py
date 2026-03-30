import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

print("=> 正在从底层数据库拉取时间序列数据...")
conn = sqlite3.connect('data/water_data.db')

# 自动寻找表并合并数据
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [t[0] for t in cursor.fetchall()]

if 'merged_data' in tables:
    df = pd.read_sql_query("SELECT * FROM merged_data", conn)
    date_col = next((c for c in df.columns if '日期' in c or 'date' in c.lower()), df.columns[0])
    df['日期'] = pd.to_datetime(df[date_col])
else:
    cons_tab = next((t for t in tables if 'consumption' in t.lower() or '药耗' in t), tables[0])
    qual_tab = next((t for t in tables if 'quality' in t.lower() or '水质' in t), tables[1] if len(tables) > 1 else tables[0])
    df_c = pd.read_sql_query(f"SELECT * FROM {cons_tab}", conn)
    df_q = pd.read_sql_query(f"SELECT * FROM {qual_tab}", conn)
    dc_c = next((c for c in df_c.columns if '日期' in c or 'date' in c.lower()), df_c.columns[0])
    dc_q = next((c for c in df_q.columns if '日期' in c or 'date' in c.lower()), df_q.columns[0])
    df_c[dc_c] = pd.to_datetime(df_c[dc_c])
    df_q[dc_q] = pd.to_datetime(df_q[dc_q])
    df = pd.merge(df_c, df_q, left_on=dc_c, right_on=dc_q, how='inner')
    df['日期'] = df[dc_c]

conn.close()

# 严格按时间排序
df = df.sort_values(by='日期').reset_index(drop=True)

# 截取最后 20% 的数据 (也就是测试集对应的时间段)
test_size = int(len(df) * 0.2)
test_df = df.iloc[-test_size:]

# 智能识别列名
turb_col = next((c for c in test_df.columns if '浊度' in c or 'turbidity' in c.lower()), None)
flow_col = next((c for c in test_df.columns if '流量' in c or 'flow' in c.lower() or 'water_supply' in c.lower() or 'supply' in c.lower()), None)
target_col = next((c for c in test_df.columns if '矾' in c or 'alum' in c.lower()), None)

# 设置 Mac 字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(15, 10))

# 画图1：实际投矾量
plt.subplot(3, 1, 1)
plt.plot(test_df['日期'], test_df[target_col], 'b-', linewidth=1.5)
plt.title(f'测试集时间段：实际投矾量 [{target_col}]', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# 画图2：浊度
if turb_col:
    plt.subplot(3, 1, 2)
    plt.plot(test_df['日期'], test_df[turb_col], 'orange', linewidth=1.5)
    plt.title(f'测试集时间段：原水浊度 [{turb_col}]', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)

# 画图3：流量
if flow_col:
    plt.subplot(3, 1, 3)
    plt.plot(test_df['日期'], test_df[flow_col], 'g-', linewidth=1.5)
    plt.title(f'测试集时间段：进水流量 [{flow_col}]', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
print("=> 绘图完成！请查看弹出的窗口。")
plt.show()