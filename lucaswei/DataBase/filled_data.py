import sqlite3
import pandas as pd


def show_all_columns(db_path):
    conn = sqlite3.connect(db_path)
    # 获取表名列表
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]

    print(f"--- 数据库维度检查 ---")
    for table in tables:
        df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 1", conn)
        print(f"\n表名: 【{table}】", end= '  ')
        print(f"包含维度 (列名):", end= '  ')
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}", end= '  ')

    conn.close()


if __name__ == "__main__":
    show_all_columns('water_plant_final.db')