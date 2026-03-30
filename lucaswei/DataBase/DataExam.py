import sqlite3
import pandas as pd


def audit_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. 检查所有表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    print(f"--- 数据库表结构 ---")
    print(f"发现表: {tables}")

    for table in tables:
        df_tmp = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        print(f"\n[{table}] 表详情:")
        print(f"  - 总行数: {len(df_tmp)}")
        print(f"  - 列名: {list(df_tmp.columns)}")

        # 2. 检查缺失值情况
        null_counts = df_tmp.isnull().sum().sum()
        print(f"  - 缺失值数量: {null_counts}")

        # 3. 统计关键列的数值区间
        if '矾\n（kg/Km³）' in df_tmp.columns or 'target_dosage' in df_tmp.columns:
            col = 'target_dosage' if 'target_dosage' in df_tmp.columns else '矾\n（kg/Km³）'
            print(f"  - {col} 区间: [{df_tmp[col].min()}, {df_tmp[col].max()}]")

    conn.close()


if __name__ == "__main__":
    # 请确保路径指向你的数据库文件
    audit_database('water_plant_final.db')