from build_database import WaterDataLoader
import pandas as pd

loader = WaterDataLoader()
df = loader.get_all_data()

# 打印原水量列的描述统计
print("原水量（Km³）列的描述统计：")
print(df['原水量（Km³）'].describe())
print("\n前10行原水量数据：")
print(df['原水量（Km³）'].head(10))