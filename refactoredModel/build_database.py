"""
文件名称：refactoredModel/build_database.py
所属类别：重构核心生产代码 (Refactored Core Production)

功能描述：
    本项目的数据仓储层构件类 `WaterDataLoader`。
    用于解析水厂提供的 Excel 历史记录大表（包括 2021 年至 2026 年的日度药耗与原水水质特征）并初始化建立本地 SQLite 数据库：
    1. 清洗原始 Excel 列名（去除隐藏换行、空格等杂乱字符）；
    2. 将中文日期（例如 'YYYY年MM月DD日'）标准化为标准 ISO 格式（'YYYY-MM-DD'）；
    3. 连接并覆盖写入本地 SQLite 的 `water_records` 表。

运行与使用方法：
    1. 直接运行以从默认位置的合并大表 Excel 重建本地 SQLite 数据库：
       python build_database.py
    2. 可以在其他模块中导入类，用于快速按时间范围拉取清洗后的 DataFrame 数据：
       from build_database import WaterDataLoader
       loader = WaterDataLoader(db_path='data/water_data.db')
       df = loader.get_all_data()

调用与依赖关系：
    - 被核心特征工程与预处理工具 `utils.py` (以及 Linux 版本的 utils) 导入以读取特征数据集。
    - 依赖 `openpyxl` 来读取现代 Excel 格式文件。
    - 数据库存储路径为 `lucaswei/DataBase/21年-26年药耗、原水数据/水务数据中心.db`。

设计细节与关键备注：
    - 数据库写入时采用 `if_exists='replace'`，确保每次重构或清空数据时是一次性完全重写。
"""
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
            try:
                # 优先尝试中文年月日格式
                df['日期'] = pd.to_datetime(df['日期'], format='%Y年%m月%d日')
            except Exception:
                try:
                    # 尝试标准日期格式
                    df['日期'] = pd.to_datetime(df['日期'], format='%Y-%m-%d')
                except Exception:
                    # 兜底通用解析
                    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
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
