#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文件名称：refactoredModel/LinuxCLI/cli_app.py
所属类别：Linux CLI 部署代码 (Linux CLI Release)

功能描述：
    专为 Linux 服务器环境打包的命令行预测工具主程序，可在无图形界面的服务器上独立运行。
    提供以下三种模式：
    1. 交互模式 (Interactive Mode)：不带参数启动，逐步引导中文输入。
    2. 参数预测模式 (Arguments Prediction Mode)：在终端直接通过 `--turbidity`、`--flow` 等参数传入指标运行，并支持以 `--json` 格式输出。
    3. 批量 CSV 模式 (Batch CSV Mode)：提供 `--csv-in` 和 `--csv-out` 路径，批量读写并计算建议投药量和泵频率。

运行与使用方法：
    1. 交互模式：
       python cli_app.py
    2. 命令行参数预测：
       python cli_app.py --turbidity 21.5 --flow 10.0 --temp 18.0 --ph 7.2 --ammonia 0.1 --stroke 65 --model resnet
    3. 命令行参数预测（JSON 格式）：
       python cli_app.py --turbidity 21.5 --flow 10.0 --temp 18.0 --ph 7.2 --ammonia 0.1 --stroke 65 --model resnet --json
    4. 批量 CSV 模式：
       python cli_app.py --csv-in input.csv --csv-out output.csv

调用与依赖关系：
    - 导入并使用 `predictor_service.WaterPredictor` (from predictor_service) 来加载权重并推理。
    - 依赖并调用配套的 `utils.py` 获取变频泵计算公式。
    - 该目录是完全自包含的，可独立复制到 Linux 机器部署，无 customtkinter 图形依赖。

设计细节与关键备注：
    - 本文件与根目录的 cli_app.py 逻辑完全一致，为保障 LinuxCLI 目录能开箱即用，在此放置镜像。
    - 在运行 CSV 模式时，如果原水量列名匹配为 flow，输入默认按小时原水量 (km³/h) 进行换算，若是日原水量需要特殊核对。
"""

import os
import sys
import argparse
import datetime
import json
import traceback

# 确保当前目录在 sys.path 中，方便导入依赖
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import pandas as pd
except ImportError:
    pd = None

from predictor_service import WaterPredictor

# 终端彩色输出辅助
class Color:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_success(msg):
    print(f"{Color.GREEN}✔ {msg}{Color.END}")

def print_warning(msg):
    print(f"{Color.YELLOW}⚠ {msg}{Color.END}")

def print_error(msg):
    print(f"{Color.RED}✘ {msg}{Color.END}", file=sys.stderr)

def print_info(msg):
    print(f"{Color.BLUE}ℹ {msg}{Color.END}")


def get_default_model_dir(model_choice):
    """根据选择的模型名称返回对应的目录"""
    if model_choice == "xgboost":
        return "models/xgboost"
    else:
        return "models/resnet"


def calculate_metrics(daily_dosage, hourly_water_km3, stroke_val):
    """
    根据日投矾量计算泵频率和各种衍生流量指标，与 GUI 保持完全一致的计算公式。
    """
    # 确保 daily_dosage 是原生 Python float 类型，以防 numpy.float32 导致 JSON 序列化失败
    daily_dosage = float(daily_dosage)
    daily_water_km3 = float(hourly_water_km3 * 24.0)
    
    # 1. 计算单位投加量 (kg/m³) - 自动 /10 修正量级
    if daily_water_km3 > 0:
        unit_dosage = float((daily_dosage / daily_water_km3) / 10.0)
    else:
        unit_dosage = 0.0

    # 2. 泵频率及流量计算
    hourly_dosage = float(daily_dosage / 24.0)                     # kg/h
    pure_volume = float(hourly_dosage / 1.25)                      # L/h (密度 1.25 kg/L)
    diluted_volume = float(pure_volume * 4.0)                      # L/h (1份原液+4份水)
    rated_flow = 1000.0                                            # L/h @50Hz, 100%冲程
    rated_freq = 50.0                                              # Hz
    
    if stroke_val > 0:
        actual_freq = float(rated_freq * (diluted_volume / rated_flow) * (100.0 / stroke_val))
    else:
        actual_freq = 0.0
    actual_freq = max(0.0, actual_freq)                            # 确保非负
    
    return {
        "daily_water_km3": daily_water_km3,
        "unit_dosage": unit_dosage,
        "hourly_dosage": hourly_dosage,
        "pure_volume": pure_volume,
        "diluted_volume": diluted_volume,
        "actual_freq": actual_freq
    }


def execute_single_prediction(predictor, date_text, turbidity_val, hourly_water_km3, temp_val, ph_val, ammonia_val, stroke_val):
    """
    执行单次预测逻辑，返回预测结果和计算后的指标字典。
    """
    # 格式化输入数据结构，保持与 ui_app.py 一致
    daily_water_km3 = hourly_water_km3 * 24.0
    input_data = {
        "日期": date_text,
        "浑浊度（NTU）_0点": turbidity_val,
        "浑浊度（NTU）": turbidity_val,
        "原水量（Km³）": daily_water_km3,
        "供水量（Km³）": daily_water_km3,
        "温度（℃）_9点": temp_val,
        "温度（℃）": temp_val,
        "氨氮（mg/L）_9点": ammonia_val,
        "氨氮（mg/L）": ammonia_val,
        "pH值_9点": ph_val,
        "pH值": ph_val,
        "冲程": stroke_val,
        "小时原水量_km3": hourly_water_km3,
    }

    # 调用模型预测
    daily_dosage, warnings = predictor.predict(input_data)
    # 转换为 Python 原生 float
    daily_dosage = float(daily_dosage)
    
    # 计算泵频率等指标
    metrics = calculate_metrics(daily_dosage, hourly_water_km3, stroke_val)
    
    return daily_dosage, warnings, metrics


def run_interactive(predictor_model_choice):
    """
    控制台交互式运行模式，逐步引导用户输入工艺参数。
    """
    print("\n" + "=" * 50)
    print(f"{Color.BOLD}水厂投矾量预测系统 (命令行交互版){Color.END}")
    print("=" * 50)

    # 1. 自动定位或加载模型
    model_dir = get_default_model_dir(predictor_model_choice)
    print_info(f"正在加载预测模型: {model_dir} ...")
    try:
        predictor = WaterPredictor(model_dir=model_dir)
        print_success(f"模型加载成功！加载的模型类型为: {predictor.model_type}")
    except Exception as exc:
        print_error(f"加载模型失败: {exc}")
        sys.exit(1)

    # 2. 交互式输入
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # 日期
    date_text = input(f"🔹 请输入日期 (格式 YYYY-MM-DD, 默认 {today_str}): ").strip()
    if not date_text:
        date_text = today_str
    else:
        try:
            datetime.datetime.strptime(date_text, "%Y-%m-%d")
        except ValueError:
            print_error("错误: 日期格式必须为 YYYY-MM-DD！")
            return

    # 数值输入辅助函数
    def ask_float(prompt, default_val=None, val_range=None):
        range_str = f" 范围: {val_range[0]}~{val_range[1]}" if val_range else ""
        default_str = f", 默认: {default_val}" if default_val is not None else ""
        while True:
            val_in = input(f"🔹 请输入{prompt}{range_str}{default_str}: ").strip()
            if not val_in:
                if default_val is not None:
                    return default_val
                else:
                    print_error("错误: 该项为必填项！")
                    continue
            try:
                val = float(val_in)
                if val_range and not (val_range[0] <= val <= val_range[1]):
                    print_warning(f"警告: 输入的值不在合理建议范围 {val_range[0]}~{val_range[1]} 内，模型精度可能降低。")
                return val
            except ValueError:
                print_error("错误: 请输入有效的数字！")

    # 获取建议范围
    ranges = getattr(predictor, "feature_ranges", {}) or {}
    turbidity_range = ranges.get("浑浊度（NTU）", [0.0, 100.0])
    ph_range = ranges.get("pH值", [0.0, 14.0])
    temp_range = ranges.get("温度（℃）", [-10.0, 50.0])

    turbidity_val = ask_float("浑浊度 (NTU)", val_range=turbidity_range)
    hourly_water_km3 = ask_float("小时原水量 (km³/h, 例如 0.3)")
    temp_val = ask_float("温度 (℃)", val_range=temp_range)
    ph_val = ask_float("pH值", val_range=ph_range)
    ammonia_val = ask_float("氨氮浓度 (mg/L)")
    stroke_val = ask_float("冲程 (%)", default_val=65.0)

    # 3. 运行预测
    print("\n" + "-" * 30 + " 正在计算中... " + "-" * 30)
    try:
        daily_dosage, warnings, metrics = execute_single_prediction(
            predictor, date_text, turbidity_val, hourly_water_km3, temp_val, ph_val, ammonia_val, stroke_val
        )
        
        # 4. 显示结果
        print("\n" + "=" * 25 + " 预测结果 " + "=" * 25)
        print(f"📊 {Color.BOLD}日加矾量预测值:{Color.END} {Color.GREEN}{daily_dosage:.2f} kg/d{Color.END}")
        print(f"📊 {Color.BOLD}单位投加量:{Color.END} {metrics['unit_dosage']:.3f} kg/m³")
        print(f"📊 {Color.BOLD}小时加药量:{Color.END} {metrics['hourly_dosage']:.2f} kg/h")
        print(f"📊 {Color.BOLD}纯矾液流量:{Color.END} {metrics['pure_volume']:.2f} L/h (密度 1.25 kg/L)")
        print(f"📊 {Color.BOLD}稀释后流量:{Color.END} {metrics['diluted_volume']:.2f} L/h (1:4 稀释比)")
        print("-" * 60)
        print(f"⚡ {Color.BOLD}变频泵建议频率:{Color.END} {Color.YELLOW}{metrics['actual_freq']:.2f} Hz{Color.END}")
        print(f"ℹ️  冲程设置: {stroke_val:.1f}% | 目标流量: {metrics['diluted_volume']:.2f} L/h | 泵额定参数: 1000 L/h @50Hz")
        print("=" * 60)

        if warnings:
            print_warning(f"模型越界警告: {warnings}")
            
    except Exception as exc:
        print_error(f"预测计算失败: {exc}")
        traceback.print_exc()


def run_batch_csv(predictor, csv_in_path, csv_out_path):
    """
    批量 CSV 文件处理模式，读取 CSV 记录并追加预测列，写入新的 CSV。
    """
    if pd is None:
        print_error("错误: 运行批量 CSV 模式需要安装 pandas！请在 Linux 上运行: pip install pandas")
        sys.exit(1)

    if not os.path.exists(csv_in_path):
        print_error(f"错误: 输入的 CSV 文件不存在: {csv_in_path}")
        sys.exit(1)

    print_info(f"正在读取 CSV 文件: {csv_in_path} ...")
    try:
        df = pd.read_csv(csv_in_path)
    except Exception as e:
        print_error(f"读取 CSV 失败: {e}")
        sys.exit(1)

    # 识别列映射
    # 尽可能模糊匹配中英文列名
    col_mapping = {
        'date': ['日期', 'date', 'Date', 'time', '时间'],
        'turbidity': ['浑浊度', '浊度', 'turbidity', 'Turbidity', '浑浊度（NTU）', '浑浊度（NTU）_0点'],
        'flow': ['原水量', 'flow', 'Flow', '水量', '小时原水量_km3', '原水量（Km³）'],
        'temp': ['温度', 'temperature', 'Temperature', 'temp', 'Temp', '温度（℃）', '温度（℃）_9点'],
        'ph': ['pH', 'ph', 'PH', 'pH值', 'pH值_9点'],
        'ammonia': ['氨氮', 'ammonia', 'Ammonia', '氨氮（mg/L）', '氨氮（mg/L）_9点'],
        'stroke': ['冲程', 'stroke', 'Stroke', '冲程(%)', '冲程（%）']
    }

    found_cols = {}
    for standard_name, aliases in col_mapping.items():
        for alias in aliases:
            if alias in df.columns:
                found_cols[standard_name] = alias
                break
        if standard_name not in found_cols and standard_name != 'stroke':
            # stroke 允许缺失，其他必填
            print_error(f"错误: 无法在 CSV 中定位属性 '{standard_name}'，请确保 CSV 中包含以下备选列名之一: {aliases}")
            sys.exit(1)

    print_info(f"匹配到的列名映射:")
    for k, v in found_cols.items():
        print(f"  - {k} -> {v}")

    # 开始逐行预测
    pred_dosages = []
    unit_dosages = []
    hourly_dosages = []
    pure_volumes = []
    diluted_volumes = []
    suggested_freqs = []
    warning_list = []

    print_info(f"共检测到 {len(df)} 行数据，开始运行批量预测...")
    for idx, row in df.iterrows():
        try:
            # 提取参数值
            date_text = str(row[found_cols['date']])
            turbidity_val = float(row[found_cols['turbidity']])
            
            # 原水量需要区分是否已经换算
            flow_val = float(row[found_cols['flow']])
            # 如果原水量列名含有 "Km³" 或值普遍很大，并且没有 "小时" 字样，可能直接就是日原水量。
            # 这里统一规定：CSV 输入如果列匹配到的是 flow，应统一为小时原水量 (km³/h)。
            # 如果其列名为日原水量或数据很大，我们在此转换为 km³/h 运行预测。
            hourly_water_km3 = flow_val
            if "Km³" in found_cols['flow'] and "小时" not in found_cols['flow']:
                # 若是日原水量(km³/d) 则除以24.0转换为 km³/h
                hourly_water_km3 = flow_val / 24.0

            temp_val = float(row[found_cols['temp']])
            ph_val = float(row[found_cols['ph']])
            ammonia_val = float(row[found_cols['ammonia']])
            
            stroke_col = found_cols.get('stroke')
            stroke_val = float(row[stroke_col]) if stroke_col and not pd.isna(row[stroke_col]) else 65.0

            daily_dosage, warnings, metrics = execute_single_prediction(
                predictor, date_text, turbidity_val, hourly_water_km3, temp_val, ph_val, ammonia_val, stroke_val
            )

            pred_dosages.append(round(daily_dosage, 2))
            unit_dosages.append(round(metrics['unit_dosage'], 3))
            hourly_dosages.append(round(metrics['hourly_dosage'], 2))
            pure_volumes.append(round(metrics['pure_volume'], 2))
            diluted_volumes.append(round(metrics['diluted_volume'], 2))
            suggested_freqs.append(round(metrics['actual_freq'], 2))
            warning_list.append(warnings if warnings else "")

        except Exception as e:
            print_warning(f"行 {idx+1} 预测失败: {e}")
            pred_dosages.append(None)
            unit_dosages.append(None)
            hourly_dosages.append(None)
            pure_volumes.append(None)
            diluted_volumes.append(None)
            suggested_freqs.append(None)
            warning_list.append(f"预测出错: {e}")

    # 将结果写入 DataFrame
    df['预测投矾量(kg/d)'] = pred_dosages
    df['单位投加量(kg/m³)'] = unit_dosages
    df['小时投加(kg/h)'] = hourly_dosages
    df['纯矾流量(L/h)'] = pure_volumes
    df['稀释后流量(L/h)'] = diluted_volumes
    df['建议泵频率(Hz)'] = suggested_freqs
    df['模型警告'] = warning_list

    # 保存文件
    try:
        df.to_csv(csv_out_path, index=False, encoding='utf-8-sig')
        print_success(f"批量预测成功！结果已保存至: {csv_out_path}")
    except Exception as e:
        print_error(f"保存 CSV 结果失败: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="水厂投矾量预测系统及变频泵频率计算 (CLI版本)")
    
    # 模式控制参数
    parser.add_argument("-i", "--interactive", action="store_true", help="强制进入交互模式（依次在提示下输入参数）")
    parser.add_argument("--json", action="store_true", help="输出单次预测结果为 JSON 格式（配合参数模式使用）")
    
    # 预测输入参数
    parser.add_argument("--date", help="预测日期，格式 YYYY-MM-DD，默认使用今天")
    parser.add_argument("--turbidity", type=float, help="浑浊度 (NTU)")
    parser.add_argument("--flow", type=float, help="小时原水量 (km³/h)")
    parser.add_argument("--temp", type=float, help="温度 (℃)")
    parser.add_argument("--ph", type=float, help="pH值")
    parser.add_argument("--ammonia", type=float, help="氨氮 (mg/L)")
    parser.add_argument("--stroke", type=float, default=65.0, help="冲程 (%%)，默认 65%%")
    
    # 模型和输入输出文件参数
    parser.add_argument("--model", choices=["resnet", "xgboost"], default="resnet", help="预测所用的模型，默认使用 resnet (双头残差网络)")
    parser.add_argument("--csv-in", help="批量预测的输入 CSV 文件路径")
    parser.add_argument("--csv-out", help="保存批量预测结果的 CSV 文件路径")

    args = parser.parse_args()

    # 1. 决定是否进入交互模式
    # 如果指定了 --interactive，或者所有参数都空且未指定 --csv-in，则进入交互模式
    is_interactive = args.interactive or (
        args.turbidity is None and 
        args.flow is None and 
        args.temp is None and 
        args.ph is None and 
        args.ammonia is None and 
        args.csv_in is None
    )

    if is_interactive:
        run_interactive(args.model)
        return

    # 2. 批量 CSV 模式
    if args.csv_in:
        if not args.csv_out:
            print_error("错误: 运行批量 CSV 模式时，必须指定输出文件路径 --csv-out")
            sys.exit(1)
        
        # 加载模型
        model_dir = get_default_model_dir(args.model)
        try:
            predictor = WaterPredictor(model_dir=model_dir)
            run_batch_csv(predictor, args.csv_in, args.csv_out)
        except Exception as exc:
            print_error(f"加载模型或批量预测失败: {exc}")
            sys.exit(1)
        return

    # 3. 命令行参数直接预测模式
    # 检查必填字段
    missing = []
    if args.turbidity is None: missing.append("--turbidity")
    if args.flow is None: missing.append("--flow")
    if args.temp is None: missing.append("--temp")
    if args.ph is None: missing.append("--ph")
    if args.ammonia is None: missing.append("--ammonia")
    
    if missing:
        print_error(f"错误: 参数不足！直接参数预测模式下，必须提供以下所有参数: {', '.join(missing)}")
        print("或者您可以使用无参数方式启动，直接进入交互式输入模式。")
        sys.exit(1)

    # 日期默认值
    date_text = args.date if args.date else datetime.datetime.now().strftime("%Y-%m-%d")
    try:
        datetime.datetime.strptime(date_text, "%Y-%m-%d")
    except ValueError:
        print_error("错误: 日期格式必须为 YYYY-MM-DD")
        sys.exit(1)

    # 加载模型
    model_dir = get_default_model_dir(args.model)
    try:
        predictor = WaterPredictor(model_dir=model_dir)
    except Exception as exc:
        print_error(f"加载模型失败: {exc}")
        sys.exit(1)

    # 执行单次预测
    try:
        daily_dosage, warnings, metrics = execute_single_prediction(
            predictor, date_text, args.turbidity, args.flow, args.temp, args.ph, args.ammonia, args.stroke
        )

        if args.json:
            # 以 JSON 格式输出
            out_json = {
                "daily_dosage_kg_d": round(daily_dosage, 4),
                "unit_dosage_kg_m3": round(metrics["unit_dosage"], 4),
                "hourly_dosage_kg_h": round(metrics["hourly_dosage"], 4),
                "pure_volume_l_h": round(metrics["pure_volume"], 4),
                "diluted_volume_l_h": round(metrics["diluted_volume"], 4),
                "suggested_freq_hz": round(metrics["actual_freq"], 2),
                "warnings": warnings if warnings else None
            }
            print(json.dumps(out_json, ensure_ascii=False))
        else:
            # 友好文本输出
            print(f"--- 预测结果 ({predictor.model_type}) ---")
            print(f"日加矾量预测值: {daily_dosage:.2f} kg/d")
            print(f"单位投加量: {metrics['unit_dosage']:.3f} kg/m³")
            print(f"建议泵频率: {metrics['actual_freq']:.2f} Hz")
            if warnings:
                print_warning(f"警告: {warnings}")
                
    except Exception as exc:
        print_error(f"预测计算发生致命错误: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()
