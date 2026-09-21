import os
import pandas as pd
import sqlite3
import datetime

class WaterDataLoader:
    def __init__(self, db_path=None):
        """
        初始化数据加载器。
        如果 db_path 为 None，默认使用 lucaswei/DataBase/21年-26年药耗、原水数据/水务数据中心.db
        """
        if db_path is None:
            # 自动推断相对项目根目录的正确路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            
            # 使用项目中的默认数据位置
            self.db_path = os.path.join(
                project_root, 
                "lucaswei", 
                "DataBase", 
                "21年-26年药耗、原水数据", 
                "水务数据中心.db"
            )
        else:
            self.db_path = db_path
            
    def _clean_columns(self, df):
        """
        清洗列名，移除换行符及空格等不规范字符，以便存入 SQLite
        """
        clean_columns = []
        for col in df.columns:
            new_col = str(col).replace('\n', '').replace('\r', '').replace(' ', '')
            clean_columns.append(new_col)
        df.columns = clean_columns
        return df

    def _parse_date(self, date_str):
        """将例如 '2021年05月01日' 转换为 '2021-05-01' 或 datetime 对象以便数据库处理"""
        try:
            return pd.to_datetime(date_str, format='%Y年%m月%d日').strftime('%Y-%m-%d')
        except:
            # 如果已经是标准格式或其他格式，尝试通用的 pandas 时间解析
            try:
                return pd.to_datetime(date_str).strftime('%Y-%m-%d')
            except:
                return date_str

    def build_database_from_excel(self, excel_path=None):
        """
        根据“最终合并数据大表.xlsx”重建 SQLite 数据库
        """
        if excel_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            excel_path = os.path.join(
                project_root, 
                "lucaswei", 
                "DataBase", 
                "21年-26年药耗、原水数据", 
                "最终合并数据大表.xlsx"
            )

        print(f"正在读取合并大表：{excel_path} ...")
        
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"找不到 Excel 文件：{excel_path}。请确保文件存在。")

        df = pd.read_excel(excel_path, engine='openpyxl')
        
        print("清理字段格式，标准化日期...")
        df = self._clean_columns(df)
        
        if '日期' in df.columns:
            df['日期'] = df['日期'].apply(self._parse_date)
            
            # 按日期排序
            df = df.sort_values(by='日期')

        print(f"准备写入 SQLite 数据库：{self.db_path}")
        # 如果目录不存在，自动创建
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # 连接数据库
        conn = sqlite3.connect(self.db_path)
        try:
            # 写入数据库，替换原有表
            df.to_sql(name='water_records', con=conn, if_exists='replace', index=False)
            print(f"🎉 成功构建数据库。共 {len(df)} 条记录存入 'water_records' 表。")
        except Exception as e:
            print(f"❌ 写入数据库失败：{e}")
        finally:
            conn.close()

    def get_all_data(self):
        """
        从 SQLite 数据库获取所有数据，以 pandas DataFrame 的形式返回
        """
        if not os.path.exists(self.db_path):
            print(f"警告: 数据库 {self.db_path} 不存在！正在自动尝试构建...")
            self.build_database_from_excel()
            
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM water_records ORDER BY 日期 ASC"
        df = pd.read_sql(query, conn)
        conn.close()
        
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
        return df

    def get_data_by_date_range(self, start_date, end_date):
        """
        按照日期范围筛选数据（例如 start_date='2021-05-01', end_date='2021-12-31'）
        返回 DataFrame
        """
        df = self.get_all_data()
        mask = (df['日期'] >= pd.to_datetime(start_date)) & (df['日期'] <= pd.to_datetime(end_date))
        return df[mask]

if __name__ == "__main__":
    # 作为主程序运行测试
    loader = WaterDataLoader()
    print("------- 开始构建或刷新本地数据库 -------")
    loader.build_database_from_excel()
    
    print("\n------- 测试读取功能 -------")
    df_all = loader.get_all_data()
    print(f"读取全量数据成功，共有 {len(df_all)} 条记录。")
    print(df_all.head(3))
