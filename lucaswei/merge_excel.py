import os
import glob
import pandas as pd
import re

# 1. 设置您的Excel文件夹路径 (请修改为您实际的路径)
folder_path = 'DataBase/21年-26年药耗、原水数据'

all_data_frames = []
excel_files = glob.glob(os.path.join(folder_path, '*.xls'))

print("开始处理，请稍候...\n")

for file in excel_files:
    filename = os.path.basename(file)
    year_match = re.search(r'(\d{4})', filename)
    file_year = year_match.group(1) if year_match else '2021'

    print(f"📄 正在处理文件: {filename}")

    try:
        sheets_dict = pd.read_excel(file, sheet_name=None, header=None, engine='xlrd')
    except Exception as e:
        print(f"  ❌ 无法读取该文件: {e}")
        continue

    for sheet_name, df in sheets_dict.items():
        if df.empty:
            continue

        header_idx = -1
        for i in range(min(10, len(df))):
            if any('日期' in str(val) for val in df.iloc[i].values):
                header_idx = i
                break

        if header_idx == -1:
            continue

        main_header = df.iloc[header_idx].tolist()

        has_sub_header = False
        sub_header = []
        if header_idx + 1 < len(df):
            row_vals = [str(x) for x in df.iloc[header_idx + 1].values]
            if any("点" in val or "时间" in val for val in row_vals):
                has_sub_header = True
                sub_header = df.iloc[header_idx + 1].tolist()

        data_start_idx = header_idx + 2 if has_sub_header else header_idx + 1

        filled_main_header = []
        current_main = ""
        for val in main_header:
            val_str = str(val).strip()
            if pd.notna(val) and val_str != "" and val_str != "nan" and "Unnamed" not in val_str:
                current_main = val_str
            filled_main_header.append(current_main)

        cols_to_keep = []
        new_col_names = []
        seen = {}

        for i in range(len(filled_main_header)):
            main_name = filled_main_header[i]
            sub_name = str(sub_header[i]).strip().replace(" ", "") if has_sub_header and i < len(sub_header) else ""

            # 保留9点数据的核心逻辑
            if "浑浊度" in main_name and has_sub_header:
                if sub_name and "9点" not in sub_name:
                    continue

            base_name = main_name
            if not base_name:
                base_name = f"未命名列_{i}"

            if base_name in seen:
                seen[base_name] += 1
                final_name = f"{base_name}_{seen[base_name]}"
            else:
                seen[base_name] = 0
                final_name = base_name

            cols_to_keep.append(i)
            new_col_names.append(final_name)

        df = df.iloc[:, cols_to_keep]
        df.columns = new_col_names
        df = df.iloc[data_start_idx:].copy()

        date_col_name = None
        for col in df.columns:
            if '日期' in col:
                date_col_name = col
                break

        if not date_col_name:
            continue


        def parse_date(x):
            if pd.isna(x): return pd.NaT
            if isinstance(x, pd.Timestamp) or hasattr(x, 'strftime'): return pd.to_datetime(x)
            if isinstance(x, (int, float)): return pd.to_datetime(x, unit='D', origin='1899-12-30')

            x_str = str(x).strip()
            if x_str.replace('.', '', 1).isdigit():
                return pd.to_datetime(float(x_str), unit='D', origin='1899-12-30')
            try:
                parsed = pd.to_datetime(x_str, errors='coerce')
                if pd.notna(parsed) and parsed.year == 1900:
                    parsed = parsed.replace(year=int(file_year))
                return parsed
            except:
                return pd.NaT


        df['辅助_排序时间'] = df[date_col_name].apply(parse_date)
        df = df.dropna(subset=['辅助_排序时间']).copy()

        df['日期'] = df['辅助_排序时间'].dt.strftime('%Y年%m月%d日')

        if date_col_name != '日期':
            df = df.drop(columns=[date_col_name])

        all_data_frames.append(df)

# 2. 合并与智能对齐输出
if all_data_frames:
    print("\n🔄 正在合并表格，并将同日期的多张表数据横向拼接到同一行...")
    # 第一步：把所有数据先上下堆叠起来
    master_df = pd.concat(all_data_frames, ignore_index=True)

    # 【新增核心代码】：按时间把属于同一天的数据“压平”合并到一行
    # 先按照时间排个序，确保万无一失
    master_df = master_df.sort_values(by='辅助_排序时间', ascending=True)

    # groupby 会把同一天的两行找出来，first() 会自动把各自有数据的列互补拼成一行！
    master_df = master_df.groupby('辅助_排序时间', as_index=False).first()

    # 清理掉辅助的时间列
    master_df = master_df.sort_values(by='辅助_排序时间', ascending=True)
    master_df = master_df.drop(columns=['辅助_排序时间'])

    # 把“日期”列提拔到最左边的第一列
    cols = master_df.columns.tolist()
    if '日期' in cols:
        cols.insert(0, cols.pop(cols.index('日期')))
        master_df = master_df[cols]

    # 3. 最终输出
    output_path = os.path.join(folder_path, '最终合并数据大表.xlsx')
    master_df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"\n🎉 完美搞定！每一天的数据都已对齐到同一行！一共生成了 {len(master_df)} 个独立日期的数据。")
    print(f"文件已保存至: {output_path}")
else:
    print("\n❌ 没有提取到数据。")