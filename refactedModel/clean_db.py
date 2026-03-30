import sqlite3

print("=> 准备对数据库进行物理清理手术...")

# 连接到你的数据库
db_path = 'data/water_data.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 获取所有表名
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [t[0] for t in cursor.fetchall()]

total_deleted = 0

for table in tables:
    try:
        # 获取当前表的所有列名
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]

        # 寻找目标列（alum_kg, 投矾量等）
        target_col = None
        for col in columns:
            if any(kw in col.lower() for kw in ['矾', 'alum', '药耗']) and 'lag' not in col.lower():
                target_col = col
                break

        if target_col:
            # 查询有多少条异常数据
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {target_col} > 5000")
            outlier_count = cursor.fetchone()[0]

            if outlier_count > 0:
                # 执行物理删除
                cursor.execute(f"DELETE FROM {table} WHERE {target_col} > 5000")
                print(f"✅ 成功从表 [{table}] 中永久删除了 {outlier_count} 条超过 5000kg 的异常数据！")
                total_deleted += outlier_count
            else:
                print(f"ℹ️ 表 [{table}] 数据正常，无极端异常值。")

    except Exception as e:
        print(f"⚠️ 处理表 {table} 时跳过: {e}")

# 提交事务并关闭连接
if total_deleted > 0:
    conn.commit()
    print(f"\n🎉 手术成功！数据库已更新，共抹除了 {total_deleted} 条脏数据。")
else:
    print("\n✅ 数据库非常干净，没有发现需要删除的脏数据。")

conn.close()