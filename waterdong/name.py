# -*- coding: utf-8 -*-
"""
Excel 表头结构分析器（增强版）
- 自动识别每个 sheet 的数据起始行
- 输出前5行原始数据
- 建议日期列和 PAC 列的位置
- 生成结构化报告，便于排查解析问题
"""

import pandas as pd
import os
import re

DATA_FOLDER = "./data"          # 数据文件夹
OUTPUT_FILE = "../列名详细分析.txt"  # 输出文件名

def parse_excel_date(val):
    """尝试判断值是否为日期格式"""
    if pd.isna(val):
        return False
    try:
        # 尝试字符串日期
        pd.to_datetime(str(val), errors='raise')
        return True
    except:
        try:
            # 尝试 Excel 数字日期
            float(val)
            return True
        except:
            return False

def analyze_file(file_path):
    """分析单个 Excel 文件的所有 sheet"""
    result = []
    try:
        xls = pd.ExcelFile(file_path)
        for sheet in xls.sheet_names:
            df_raw = pd.read_excel(file_path, sheet_name=sheet, header=None)
            if df_raw.empty:
                continue
            # 删除全空行
            df_raw = df_raw.dropna(how='all', axis=0).reset_index(drop=True)
            if df_raw.empty:
                continue

            # 尝试找数据起始行（第一列包含日期或“日期”字样的行）
            start_row = None
            for i in range(min(20, len(df_raw))):
                first_cell = str(df_raw.iloc[i, 0]).strip()
                if re.match(r'\d{4}-\d{1,2}-\d{1,2}', first_cell) or '日期' in first_cell:
                    start_row = i
                    break
                # 尝试 Excel 数字日期
                try:
                    date_num = float(first_cell)
                    if 40000 <= date_num <= 50000:
                        start_row = i
                        break
                except:
                    pass
            if start_row is None:
                start_row = 0  # 若找不到，假设从第一行开始

            # 提取前5行数据（用于人工查看）
            preview = df_raw.iloc[:5].to_string(index=False, header=False).replace('\n', '; ')

            # 尝试识别日期列和 PAC 列
            # 先合并表头区域（0到start_row-1）作为候选表头
            header_rows = df_raw.iloc[:max(1, start_row)].fillna('').astype(str)
            merged_headers = []
            for col_idx in range(len(df_raw.columns)):
                col_vals = header_rows.iloc[:, col_idx].tolist()
                merged = ' '.join([v for v in col_vals if v.strip()]).strip()
                merged_headers.append(merged)

            # 日期列
            date_col_idx = None
            for idx, header in enumerate(merged_headers):
                if '日期' in header:
                    date_col_idx = idx
                    break
            if date_col_idx is None:
                # 根据内容判断
                for idx in range(len(df_raw.columns)):
                    sample = df_raw.iloc[start_row:start_row+20, idx].dropna()
                    if len(sample) == 0:
                        continue
                    if all(parse_excel_date(v) for v in sample):
                        date_col_idx = idx
                        break
            if date_col_idx is None:
                date_col_idx = 0  # 默认第一列

            # PAC 列
            pac_col_idx = None
            for idx, header in enumerate(merged_headers):
                if re.search(r'PAC|矾|聚合氯化铝|投加量', header, re.I):
                    pac_col_idx = idx
                    break
            if pac_col_idx is None:
                for idx in range(len(df_raw.columns)):
                    if idx == date_col_idx:
                        continue
                    sample = df_raw.iloc[start_row:start_row+20, idx].dropna()
                    if len(sample) == 0:
                        continue
                    numeric = pd.to_numeric(sample, errors='coerce')
                    if numeric.notna().sum() > len(sample)*0.8:
                        pac_col_idx = idx
                        break
            if pac_col_idx is None:
                pac_col_idx = -1  # 未找到

            result.append({
                'sheet': sheet,
                'start_row': start_row,
                'preview': preview,
                'date_col_idx': date_col_idx,
                'pac_col_idx': pac_col_idx,
                'merged_headers': merged_headers
            })
    except Exception as e:
        print(f"文件 {file_path} 分析失败: {e}")
    return result

def main():
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("Excel 表头结构分析报告\n")
        f.write("=" * 80 + "\n\n")
        for file in os.listdir(DATA_FOLDER):
            if file.endswith(('.xls', '.xlsx')):
                file_path = os.path.join(DATA_FOLDER, file)
                f.write(f"文件: {file}\n")
                f.write("-" * 60 + "\n")
                results = analyze_file(file_path)
                for r in results:
                    f.write(f"  Sheet: {r['sheet']}\n")
                    f.write(f"    数据起始行索引: {r['start_row']}\n")
                    f.write(f"    前5行预览: {r['preview'][:200]}...\n")  # 截断过长的
                    f.write(f"    建议日期列索引: {r['date_col_idx']} (列名: {r['merged_headers'][r['date_col_idx']] if r['date_col_idx'] < len(r['merged_headers']) else 'N/A'})\n")
                    f.write(f"    建议PAC列索引: {r['pac_col_idx']} (列名: {r['merged_headers'][r['pac_col_idx']] if r['pac_col_idx'] >=0 and r['pac_col_idx'] < len(r['merged_headers']) else 'N/A'})\n")
                    f.write("\n")
                f.write("\n")
        print(f"分析完成，结果保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()