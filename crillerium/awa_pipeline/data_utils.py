from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


MISSING_TOKENS = {"", "nan", "none", "null", "/", "\\", "-", "--", "—"}


@dataclass
class DatasetBundle:
    raw_water: pd.DataFrame
    chemical: pd.DataFrame
    merged: pd.DataFrame
    missing_summary: pd.DataFrame
    outlier_summary: pd.DataFrame


def normalize_text(value: object) -> str:
    text = str(value).replace("\n", "").replace("\r", "").strip()
    text = re.sub(r"\s+", "", text)
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("³", "3")
    return text


def sanitize_numeric(value: object) -> float | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if normalize_text(text).lower() in MISSING_TOKENS:
        return pd.NA
    text = (
        text.replace(",", "")
        .replace("，", "")
        .replace("＜", "<")
        .replace("≤", "")
        .replace("≥", "")
    )
    text = text.replace(" ", "")
    if text.startswith("<"):
        text = text[1:]
    match = re.search(r"-?\d+(\.\d+)?", text)
    if not match:
        return pd.NA
    try:
        return float(match.group())
    except ValueError:
        return pd.NA


def _deduplicate_columns(columns: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    result: list[str] = []
    for col in columns:
        base = col or "未命名字段"
        idx = counts.get(base, 0)
        counts[base] = idx + 1
        result.append(base if idx == 0 else f"{base}_{idx + 1}")
    return result


def build_raw_water_columns(frame: pd.DataFrame) -> list[str]:
    header0 = frame.iloc[0].ffill().tolist()
    header1 = frame.iloc[1].tolist()
    columns: list[str] = []
    for idx, (h0, h1) in enumerate(zip(header0, header1)):
        top = normalize_text(h0)
        sub = normalize_text(h1)
        if idx == 0 or top in {"日期", "时间"}:
            columns.append("日期")
        elif sub and sub != "时间":
            columns.append(f"{top}_{sub}")
        else:
            columns.append(top or f"列{idx + 1}")
    return _deduplicate_columns(columns)


def build_chemical_columns(frame: pd.DataFrame) -> list[str]:
    header = [normalize_text(item) for item in frame.iloc[1].tolist()]
    if header:
        header[0] = "日期"
    return _deduplicate_columns(header)


def _finalize_sheet_data(frame: pd.DataFrame, columns: list[str], start_row: int) -> pd.DataFrame:
    data = frame.iloc[start_row:].copy()
    data.columns = columns
    data = data.dropna(how="all")
    if "日期" not in data.columns:
        raise ValueError("无法识别日期列")
    data["日期"] = pd.to_datetime(data["日期"], errors="coerce")
    data = data[data["日期"].notna()].copy()
    for col in data.columns:
        if col == "日期":
            continue
        data[col] = data[col].map(sanitize_numeric).astype("Float64")
    numeric_cols = [col for col in data.columns if col != "日期"]
    if numeric_cols:
        data = data[data[numeric_cols].notna().sum(axis=1) > 0].copy()
    return data


def load_raw_water_file(path: Path) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    frames: list[pd.DataFrame] = []
    for sheet in workbook.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        if raw.shape[0] < 3:
            continue
        columns = build_raw_water_columns(raw)
        data = _finalize_sheet_data(raw, columns, start_row=2)
        if not data.empty:
            data["源文件"] = path.name
            data["工作表"] = sheet.strip()
            frames.append(data)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_chemical_file(path: Path) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    frames: list[pd.DataFrame] = []
    for sheet in workbook.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        if raw.shape[0] < 3:
            continue
        columns = build_chemical_columns(raw)
        data = _finalize_sheet_data(raw, columns, start_row=2)
        if not data.empty:
            data["源文件"] = path.name
            data["工作表"] = sheet.strip()
            frames.append(data)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def resolve_duplicate_dates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    non_meta_cols = [c for c in frame.columns if c not in {"日期", "源文件", "工作表"}]
    frame = frame.copy()
    frame["_non_null_count"] = frame[non_meta_cols].notna().sum(axis=1)
    frame = (
        frame.sort_values(["日期", "_non_null_count"], ascending=[True, False])
        .drop_duplicates(subset=["日期"], keep="first")
        .drop(columns="_non_null_count")
        .sort_values("日期")
        .reset_index(drop=True)
    )
    return frame


def impute_missing_values(frame: pd.DataFrame, high_missing_threshold: float = 0.35) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.sort_values("日期").reset_index(drop=True).copy()
    numeric_cols = [c for c in frame.columns if c not in {"日期", "源文件", "工作表"}]
    missing_rows = []
    drop_cols: list[str] = []
    for col in numeric_cols:
        missing_rate = float(frame[col].isna().mean())
        strategy = "保留"
        if missing_rate < 0.05:
            frame[col] = frame[col].interpolate(method="linear", limit_direction="both")
            strategy = "线性插值"
        elif missing_rate > high_missing_threshold:
            frame[f"{col}_高缺失标记"] = frame[col].isna().astype(int)
            drop_cols.append(col)
            strategy = "高缺失剔除并保留标记"
        else:
            frame[col] = frame[col].fillna(frame[col].median())
            strategy = "中位数填补"
        missing_rows.append({"字段": col, "缺失率": missing_rate, "处理策略": strategy})
    if drop_cols:
        frame = frame.drop(columns=drop_cols)
    return frame, pd.DataFrame(missing_rows).sort_values("缺失率", ascending=False)


def detect_and_handle_outliers(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    outlier_rows: list[dict[str, object]] = []
    numeric_cols = [c for c in frame.columns if c != "日期"]
    for col in numeric_cols:
        series = pd.to_numeric(frame[col], errors="coerce")
        if series.notna().sum() < 10:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr
        mean = series.mean()
        std = series.std()
        lower_sigma = mean - 3 * std
        upper_sigma = mean + 3 * std
        lower = max(lower_iqr, lower_sigma) if not np.isnan(std) else lower_iqr
        upper = min(upper_iqr, upper_sigma) if not np.isnan(std) else upper_iqr
        mask = (series < lower) | (series > upper)
        if "耗用矾量" in col or "矾(kg/Km3)" in col:
            mask = mask | (series < 0)
        if "浑浊度" in col:
            mask = mask | (series < 0)
        outlier_rows.append(
            {
                "字段": col,
                "异常值数量": int(mask.sum()),
                "异常值占比": float(mask.mean()),
                "下界": lower,
                "上界": upper,
            }
        )
        if mask.any():
            frame.loc[mask, col] = series.clip(lower=lower, upper=upper)[mask]
    return frame, pd.DataFrame(outlier_rows).sort_values("异常值占比", ascending=False)


def merge_datasets(raw_water: pd.DataFrame, chemical: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(raw_water, chemical, on="日期", how="inner", suffixes=("_原水", "_药耗"))
    merged = merged.sort_values("日期").reset_index(drop=True)
    merged["月份"] = merged["日期"].dt.month
    merged["年份"] = merged["日期"].dt.year
    merged["是否梅雨季"] = merged["月份"].between(4, 6).astype(int)
    merged["是否水库分层期"] = merged["月份"].between(6, 10).astype(int)
    return merged


def load_and_prepare_datasets(project_root: Path) -> DatasetBundle:
    raw_files = sorted(project_root.glob("原水数据*.xls"))
    chemical_files = sorted(project_root.glob("药耗数据*.xls"))
    raw_frames = [load_raw_water_file(path) for path in raw_files]
    chemical_frames = [load_chemical_file(path) for path in chemical_files]
    raw_water = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    chemical = pd.concat(chemical_frames, ignore_index=True) if chemical_frames else pd.DataFrame()
    raw_water = resolve_duplicate_dates(raw_water)
    chemical = resolve_duplicate_dates(chemical)
    raw_water, raw_missing = impute_missing_values(raw_water)
    chemical, chemical_missing = impute_missing_values(chemical)
    raw_water, raw_outliers = detect_and_handle_outliers(raw_water)
    chemical, chemical_outliers = detect_and_handle_outliers(chemical)
    merged = merge_datasets(raw_water, chemical)
    missing_summary = pd.concat(
        [
            raw_missing.assign(数据源="原水数据"),
            chemical_missing.assign(数据源="药耗数据"),
        ],
        ignore_index=True,
    )
    outlier_summary = pd.concat(
        [
            raw_outliers.assign(数据源="原水数据"),
            chemical_outliers.assign(数据源="药耗数据"),
        ],
        ignore_index=True,
    )
    return DatasetBundle(
        raw_water=raw_water,
        chemical=chemical,
        merged=merged,
        missing_summary=missing_summary,
        outlier_summary=outlier_summary,
    )
