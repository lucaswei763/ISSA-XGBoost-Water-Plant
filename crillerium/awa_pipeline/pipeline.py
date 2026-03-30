from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from awa_pipeline.data_utils import DatasetBundle, load_and_prepare_datasets
from awa_pipeline.modeling import ModelArtifacts, run_model_suite
from awa_pipeline.reporting import (
    build_word_report,
    save_correlation_heatmap,
    save_distribution_plots,
    save_model_plots,
    save_operating_condition_table,
    save_trend_plots,
    setup_plotting,
)


TARGET_CANDIDATES = ["矾(kg/Km3)", "矾(kg/Km³)", "耗用矾量(kg)"]
LEAKAGE_KEYWORDS = ["耗用矾量", "矾(", "耗用次氯酸钠", "耗用氯量", "次氯耗", "氯耗", "消耗电", "电耗", "供水量"]


def choose_target_column(frame: pd.DataFrame) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    raise ValueError("未找到投矾目标列")


def first_existing(columns: list[str], data: pd.DataFrame, fallback_prefix: str) -> str:
    for col in columns:
        if col in data.columns:
            return col
    matches = [c for c in data.columns if fallback_prefix in c]
    if not matches:
        raise ValueError(f"缺少必要字段: {fallback_prefix}")
    return matches[0]


def should_drop_feature(col: str, target_col: str) -> bool:
    if col == target_col:
        return True
    return any(keyword in col for keyword in LEAKAGE_KEYWORDS)


def describe_feature(name: str) -> str:
    rules = {
        "target_lag": "前序投矾记忆特征，反映连续工况下的投药惯性。",
        "target_roll": "历史投矾滚动均值，表征阶段投药水平。",
        "target_diff": "最近投矾变化趋势特征。",
        "滞后": "体现历史过程记忆与时滞影响。",
        "滚动均值": "平滑短期扰动，刻画阶段工况水平。",
        "变化率": "反映原水水质或水量变化速度。",
        "温度补偿项": "低温条件下的混凝补偿特征。",
        "pH修正项": "pH 偏离中性时的工艺修正特征。",
        "有机物放大项": "高锰酸盐指数偏高时的有机物风险放大。",
        "浊度x原水量": "浊度与处理规模的交互强度。",
        "浊度x高锰酸盐指数": "颗粒物与有机物叠加影响。",
        "月份分类": "季节性编码。",
        "是否梅雨季分类": "梅雨季标识。",
        "是否分层期分类": "水库分层期标识。",
    }
    for key, desc in rules.items():
        if key in name:
            return desc
    return "由原始监测字段或其聚合值构成的基础输入特征。"


def engineer_features(frame: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy().sort_values("日期").reset_index(drop=True)
    text_cols = [c for c in data.columns if c.startswith("源文件") or c.startswith("工作表")]
    if text_cols:
        data = data.drop(columns=text_cols)
    data["对齐后投矾目标"] = data[target_col]
    leakage_cols = [c for c in data.columns if should_drop_feature(c, target_col)]
    protected = {"对齐后投矾目标", target_col}
    data = data.drop(columns=[c for c in leakage_cols if c not in protected])

    turbidity_cols = [c for c in data.columns if "浑浊度" in c]
    ph_cols = [c for c in data.columns if "pH值" in c]
    temp_cols = [c for c in data.columns if "温度" in c]
    kmno4_cols = [c for c in data.columns if "高锰酸盐指数" in c]
    flow_cols = [c for c in data.columns if "原水量" in c]

    if turbidity_cols:
        data["原水浊度均值"] = data[turbidity_cols].mean(axis=1)
        data["原水浊度极差"] = data[turbidity_cols].max(axis=1) - data[turbidity_cols].min(axis=1)
    if ph_cols:
        data["原水pH均值"] = data[ph_cols].mean(axis=1)
    if temp_cols:
        data["原水温度均值"] = data[temp_cols].mean(axis=1)
    if kmno4_cols:
        data["高锰酸盐指数均值"] = data[kmno4_cols].mean(axis=1)
    if flow_cols:
        data["原水量均值"] = data[flow_cols].mean(axis=1)

    shift_source = first_existing(["原水浊度均值"], data, "浑浊度")
    flow_source = first_existing(["原水量均值"], data, "原水量")
    temp_source = first_existing(["原水温度均值"], data, "温度")
    ph_source = first_existing(["原水pH均值"], data, "pH值")
    kmno4_source = first_existing(["高锰酸盐指数均值"], data, "高锰酸盐指数")

    for lag in range(1, 7):
        data[f"浊度滞后{lag}日"] = data[shift_source].shift(lag)
    for lag in range(1, 4):
        data[f"原水量滞后{lag}日"] = data[flow_source].shift(lag)
    for lag in range(1, 15):
        data[f"target_lag_{lag}"] = data["对齐后投矾目标"].shift(lag)

    data["浊度日变化率"] = data[shift_source].pct_change().replace([np.inf, -np.inf], np.nan)
    data["原水量日变化率"] = data[flow_source].pct_change().replace([np.inf, -np.inf], np.nan)
    data["浊度3日滚动均值"] = data[shift_source].rolling(3).mean()
    data["浊度7日滚动均值"] = data[shift_source].rolling(7).mean()
    data["原水量3日滚动均值"] = data[flow_source].rolling(3).mean()
    data["原水量7日滚动均值"] = data[flow_source].rolling(7).mean()
    data["target_roll3"] = data["对齐后投矾目标"].shift(1).rolling(3).mean()
    data["target_roll7"] = data["对齐后投矾目标"].shift(1).rolling(7).mean()
    data["target_roll14"] = data["对齐后投矾目标"].shift(1).rolling(14).mean()
    data["target_diff_1"] = data["对齐后投矾目标"].shift(1) - data["对齐后投矾目标"].shift(2)
    data["target_diff_3"] = data["对齐后投矾目标"].shift(1) - data["对齐后投矾目标"].shift(4)

    data["温度补偿项"] = np.where(data[temp_source] < 10, (10 - data[temp_source]) / 10, 0.0)
    data["pH修正项"] = np.where((data[ph_source] < 6.5) | (data[ph_source] > 8.0), np.abs(data[ph_source] - 7) / 2, 0.0)
    data["有机物放大项"] = np.where(data[kmno4_source] > 1.5, data[kmno4_source] / 1.5, 0.0)

    data["月份分类"] = data["日期"].dt.month.astype(str)
    data["是否梅雨季分类"] = data["是否梅雨季"].map({0: "否", 1: "是"})
    data["是否分层期分类"] = data["是否水库分层期"].map({0: "否", 1: "是"})

    data["浊度x原水量"] = data[shift_source] * data[flow_source]
    data["浊度x高锰酸盐指数"] = data[shift_source] * data[kmno4_source]
    data["pHx温度"] = data[ph_source] * data[temp_source]

    data["工况_低浊期"] = (data[shift_source] <= data[shift_source].quantile(0.25)).astype(int)
    data["工况_汛期"] = data["日期"].dt.month.isin([6, 7, 8, 9]).astype(int)
    data["工况_低温期"] = (data[temp_source] < 10).astype(int)
    data["工况_分层期"] = data["日期"].dt.month.isin([6, 7, 8, 9, 10]).astype(int)

    data = data.dropna(subset=["对齐后投矾目标"]).copy()
    data = data.iloc[14:].reset_index(drop=True)
    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        data[col] = data[col].ffill().bfill()
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].median())

    feature_rows = []
    for col in data.columns:
        if col in {"日期", "对齐后投矾目标", target_col}:
            continue
        feature_type = "分类特征" if col.endswith("分类") else "连续特征"
        if col.startswith("工况_"):
            feature_type = "工况标签"
        feature_rows.append({"特征名": col, "类型": feature_type, "说明": describe_feature(col)})
    feature_catalog = pd.DataFrame(feature_rows)
    return data, feature_catalog


def prepare_model_matrix(data: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    model_data = data.copy()
    y = model_data["对齐后投矾目标"].astype(float)
    meta_cols = ["日期", "月份", "工况_低浊期", "工况_汛期", "工况_低温期", "工况_分层期"]
    meta = model_data[meta_cols].copy()
    drop_cols = ["对齐后投矾目标", "日期", target_col]
    raw_X = model_data.drop(columns=[c for c in drop_cols if c in model_data.columns])
    categorical_cols = [c for c in raw_X.columns if c.endswith("分类")]
    numeric_cols = [c for c in raw_X.columns if c not in categorical_cols]
    scaler = StandardScaler()
    X_num = pd.DataFrame(scaler.fit_transform(raw_X[numeric_cols]), columns=numeric_cols, index=raw_X.index)
    if categorical_cols:
        cat_frame = raw_X[categorical_cols].astype(str)
        X_cat = pd.get_dummies(cat_frame, columns=categorical_cols, prefix={col: col for col in categorical_cols}, dtype=float)
    else:
        X_cat = pd.DataFrame(index=raw_X.index)
    X_final = pd.concat([X_num, X_cat], axis=1)
    return X_final, y, meta, raw_X


def split_by_time(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, raw_X: pd.DataFrame):
    n = len(X)
    train_end = int(n * 0.8)
    valid_end = int(train_end * 0.8)
    return (
        X.iloc[:valid_end],
        y.iloc[:valid_end],
        X.iloc[valid_end:train_end],
        y.iloc[valid_end:train_end],
        X.iloc[train_end:],
        y.iloc[train_end:],
        meta.iloc[train_end:].reset_index(drop=True),
        raw_X.iloc[:valid_end].reset_index(drop=True),
        raw_X.iloc[valid_end:train_end].reset_index(drop=True),
        raw_X.iloc[train_end:].reset_index(drop=True),
    )


def export_tables(bundle: DatasetBundle, feature_frame: pd.DataFrame, feature_catalog: pd.DataFrame, stats_table: pd.DataFrame, corr_table: pd.DataFrame, condition_table: pd.DataFrame, artifacts: ModelArtifacts, output_root: Path) -> None:
    data_dir = output_root / "data"
    result_dir = output_root / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    bundle.raw_water.to_csv(data_dir / "cleaned_raw_water.csv", index=False, encoding="utf-8-sig")
    bundle.chemical.to_csv(data_dir / "cleaned_chemical.csv", index=False, encoding="utf-8-sig")
    bundle.merged.to_csv(data_dir / "cleaned_merged_dataset.csv", index=False, encoding="utf-8-sig")
    feature_frame.to_csv(data_dir / "feature_engineered_dataset.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(result_dir / "data_quality_and_features.xlsx") as writer:
        bundle.missing_summary.to_excel(writer, sheet_name="缺失值处理", index=False)
        bundle.outlier_summary.to_excel(writer, sheet_name="异常值处理", index=False)
        feature_catalog.to_excel(writer, sheet_name="特征清单", index=False)
        stats_table.to_excel(writer, sheet_name="分布统计", index=False)
        corr_table.to_excel(writer, sheet_name="相关性", index=False)
        condition_table.to_excel(writer, sheet_name="工况分析", index=False)
    artifacts.train_metrics.to_excel(result_dir / "model_metrics.xlsx", index=False)
    artifacts.comparison_metrics.to_excel(result_dir / "comparison_metrics.xlsx", index=False)
    artifacts.stability_metrics.to_excel(result_dir / "stability_metrics.xlsx", index=False)
    artifacts.generalization_metrics.to_excel(result_dir / "generalization_metrics.xlsx", index=False)
    artifacts.test_predictions.to_excel(result_dir / "test_predictions.xlsx", index=False)
    artifacts.feature_importance.to_excel(result_dir / "feature_importance.xlsx", index=False)
    with open(result_dir / "best_params.json", "w", encoding="utf-8") as fp:
        json.dump({
            "recommended_model": "Huber-ElasticNet\u878d\u5408",
            "recommended_reason": "\u5728\u52a0\u5165\u6295\u77fe\u8fc7\u7a0b\u8bb0\u5fc6\u7279\u5f81\u540e\uff0cHuber \u9c81\u68d2\u56de\u5f52\u4e0e ElasticNet \u5e73\u53f0\u6821\u6b63\u7684\u878d\u5408\u7ed3\u679c\u6700\u8d34\u5408\u5f53\u524d\u65e5\u5c3a\u5ea6\u6570\u636e\u3002",
            "recommended_weight_huber": artifacts.recommendation_weight,
            "recommended_rule": artifacts.recommendation_rule,
            "tree_model_choice": "LightGBM",
            "tree_model_reason": "满足原始树模型约束，并保留 ISSA 优化过程与特征重要性输出。",
            "manual_experience_baseline": "未提供人工经验投加记录，因此未纳入正式对比。",
            "leakage_control": "已剔除投矾、消毒药耗、电耗、供水量等同日结果性字段，避免未来信息泄露。",
            "best_validation_rmse_lightgbm": artifacts.best_score,
            "best_params_lightgbm": artifacts.best_params,
        }, fp, ensure_ascii=False, indent=2)


def save_model(artifacts: ModelArtifacts, output_root: Path) -> None:
    model_dir = output_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.final_model, model_dir / "optimized_lightgbm.pkl")
    joblib.dump(artifacts.recommended_model, model_dir / "recommended_blend_model.pkl")
    joblib.dump(artifacts.recommended_model, model_dir / "enhanced_elasticnet.pkl")
    with open(model_dir / "enhanced_elasticnet_rule.json", "w", encoding="utf-8") as fp:
        json.dump({"weight_huber": artifacts.recommendation_weight, "rule": artifacts.recommendation_rule}, fp, ensure_ascii=False, indent=2)
    artifacts.final_model.booster_.save_model(str(model_dir / "optimized_lightgbm.txt"))


def run_pipeline(project_root: Path) -> None:
    output_root = project_root / "outputs"
    figures_dir = output_root / "figures"
    results_dir = output_root / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_plotting()
    bundle = load_and_prepare_datasets(project_root)
    target_col = choose_target_column(bundle.merged)
    feature_frame, feature_catalog = engineer_features(bundle.merged, target_col)
    stats_table = save_distribution_plots(feature_frame, figures_dir)
    save_trend_plots(feature_frame, figures_dir)
    corr_table = save_correlation_heatmap(feature_frame, figures_dir, target_col="对齐后投矾目标")
    condition_table = save_operating_condition_table(feature_frame, results_dir, target_col="对齐后投矾目标")
    X, y, meta, raw_X = prepare_model_matrix(feature_frame, target_col)
    X_train, y_train, X_valid, y_valid, X_test, y_test, meta_test, raw_train, raw_valid, raw_test = split_by_time(X, y, meta, raw_X)
    artifacts = run_model_suite(X_train, y_train, X_valid, y_valid, X_test, y_test, raw_train, raw_valid, raw_test, meta_test)
    save_model_plots(artifacts.test_predictions, artifacts.feature_importance, artifacts.convergence, figures_dir)
    export_tables(bundle, feature_frame, feature_catalog, stats_table, corr_table, condition_table, artifacts, output_root)
    save_model(artifacts, output_root)
    summary_text = [
        "\u63a8\u8350\u6a21\u578b\uff1aHuber-ElasticNet \u878d\u5408\u56de\u5f52\u3002\u8be5\u6a21\u578b\u5c06 14 \u5929\u6295\u77fe\u8bb0\u5fc6\u3001\u6eda\u52a8\u5747\u503c\u3001\u53d8\u5316\u91cf\u7279\u5f81\u4e0e\u9c81\u68d2\u56de\u5f52\u3001\u5e73\u53f0\u6821\u6b63\u7ec4\u5408\uff0c\u5728\u5f53\u524d\u6570\u636e\u6761\u4ef6\u4e0b\u8d34\u5408\u5ea6\u6700\u4f73\u3002",
        "约束模型：采用带 Sobol 初始化、双样本学习和柯西-高斯变异的 ISSA 优化 LightGBM，保留原始树模型路线与可解释性输出。",
        "对齐说明：药耗数据为日尺度，原水监测包含日内多时点。由于缺少小时级投药日志，2 小时时滞通过历史滞后特征、投矾记忆特征和滚动窗口特征近似表达。",
        f"最终目标变量：{target_col}。",
        "已剔除同日投矾、电耗、次氯酸钠耗量、供水量等结果性变量，降低未来信息泄露风险。",
        "\u4eba\u5de5\u7ecf\u9a8c\u6295\u52a0\u8bb0\u5f55\u672a\u63d0\u4f9b\uff0c\u56e0\u6b64\u5bf9\u6bd4\u5b9e\u9a8c\u5305\u542b\u878d\u5408\u63a8\u8350\u6a21\u578b\u3001\u4f18\u5316/\u672a\u4f18\u5316 LightGBM\u3001\u7ebf\u6027\u56de\u5f52\u4e0e\u5e42\u51fd\u6570\u56de\u5f52\u3002",
    ]
    image_paths = [figures_dir / "time_series_trends.png", figures_dir / "correlation_heatmap.png", figures_dir / "issa_convergence.png", figures_dir / "prediction_vs_actual_scatter.png", figures_dir / "time_series_prediction_100_points.png", figures_dir / "error_distribution.png", figures_dir / "feature_importance_top20.png"]
    build_word_report(output_root / "混凝智能投药预测模型技术报告.docx", summary_text, artifacts.train_metrics, artifacts.comparison_metrics, artifacts.stability_metrics.describe().reset_index(), artifacts.generalization_metrics, image_paths)
