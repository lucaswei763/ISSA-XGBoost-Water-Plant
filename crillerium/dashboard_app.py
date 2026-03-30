from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from awa_pipeline.modeling import apply_recommendation_rule
from awa_pipeline.pipeline import prepare_model_matrix


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUT_ROOT / "results"
DATA_DIR = OUTPUT_ROOT / "data"
MODEL_DIR = OUTPUT_ROOT / "model"

COL_DATE = "\u65e5\u671f"
COL_ACTUAL = "\u771f\u5b9e\u503c"
COL_ERROR = "\u8bef\u5dee"
COL_RECOMMENDED = "\u63a8\u8350\u6a21\u578b\u9884\u6d4b"
COL_REPLAY = "\u5386\u53f2\u56de\u653e\u9884\u6d4b"
COL_TARGET = "\u5bf9\u9f50\u540e\u6295\u77fe\u76ee\u6807"
COL_LGBM = "\u4f18\u5316LightGBM\u9884\u6d4b"
COL_LGBM_DEFAULT = "\u672a\u4f18\u5316LightGBM\u9884\u6d4b"
COL_LINEAR = "\u7ebf\u6027\u56de\u5f52\u9884\u6d4b"
COL_POWER = "\u5e42\u51fd\u6570\u56de\u5f52\u9884\u6d4b"
COL_MODEL = "\u6a21\u578b"
COL_FEATURE = "\u7279\u5f81"
COL_BLEND_IMPORTANCE = "\u878d\u5408\u7ebf\u6027\u7cfb\u6570\u7edd\u5bf9\u503c"
COL_ENET_IMPORTANCE = "ElasticNet\u7cfb\u6570\u7edd\u5bf9\u503c"
COL_HUBER_IMPORTANCE = "Huber\u7cfb\u6570\u7edd\u5bf9\u503c"
COL_LGBM_IMPORTANCE = "LightGBM\u91cd\u8981\u6027"
NAME_RECOMMENDED = "Huber-ElasticNet\u878d\u5408(\u63a8\u8350)"
TARGET_CANDIDATES = ["\u77fe(kg/Km3)", "\u77fe(kg/Km\u00b3)", "\u8017\u7528\u77fe\u91cf(kg)"]
HISTORY_FEATURE_CANDIDATES = [
    "\u5bf9\u9f50\u540e\u6295\u77fe\u76ee\u6807",
    "\u539f\u6c34\u6d4a\u5ea6\u5747\u503c",
    "\u539f\u6c34pH\u5747\u503c",
    "\u539f\u6c34\u6e29\u5ea6\u5747\u503c",
    "\u9ad8\u9530\u9178\u76d0\u6307\u6570\u5747\u503c",
    "\u539f\u6c34\u91cf\u5747\u503c",
]

st.set_page_config(page_title="\u6df7\u51dd\u667a\u80fd\u6295\u836f\u6a21\u578b\u53ef\u89c6\u5316\u770b\u677f", page_icon="\U0001F4C8", layout="wide")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    df = pd.read_excel(RESULTS_DIR / "test_predictions.xlsx")
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    return df.sort_values(COL_DATE).reset_index(drop=True)


@st.cache_data
def load_metrics() -> pd.DataFrame:
    return pd.read_excel(RESULTS_DIR / "model_metrics.xlsx")


@st.cache_data
def load_comparison() -> pd.DataFrame:
    return pd.read_excel(RESULTS_DIR / "comparison_metrics.xlsx")


@st.cache_data
def load_generalization() -> pd.DataFrame:
    path = RESULTS_DIR / "generalization_metrics.xlsx"
    return pd.read_excel(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_operating_conditions() -> pd.DataFrame:
    path = RESULTS_DIR / "operating_conditions.xlsx"
    return pd.read_excel(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_importance() -> pd.DataFrame:
    return pd.read_excel(RESULTS_DIR / "feature_importance.xlsx")


@st.cache_data
def load_feature_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "feature_engineered_dataset.csv")
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    return df


@st.cache_data
def load_cleaned_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "cleaned_merged_dataset.csv")
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    return df


@st.cache_data
def load_best_params_text() -> str:
    return (RESULTS_DIR / "best_params.json").read_text(encoding="utf-8")


@st.cache_data
def load_replay_predictions() -> pd.DataFrame:
    feature_df = load_feature_data().copy()
    target_col = next((c for c in TARGET_CANDIDATES if c in feature_df.columns), None)
    if target_col is None:
        raise ValueError("\u672a\u5728\u7279\u5f81\u5de5\u7a0b\u6570\u636e\u4e2d\u627e\u5230\u6295\u77fe\u76ee\u6807\u5217")
    X_all, y_all, meta_all, raw_X = prepare_model_matrix(feature_df, target_col)
    model_path = MODEL_DIR / "recommended_blend_model.pkl"
    if not model_path.exists():
        model_path = MODEL_DIR / "enhanced_elasticnet.pkl"
    bundle = joblib.load(model_path)
    enet = bundle["elasticnet"]
    huber = bundle["huber"]
    weight = float(bundle["weight_huber"])
    rule = bundle["rule"]
    enet_pred = enet.predict(X_all)
    huber_pred = huber.predict(X_all)
    replay_pred = weight * huber_pred + (1 - weight) * apply_recommendation_rule(enet_pred, raw_X, rule)
    replay = pd.DataFrame({
        COL_DATE: meta_all[COL_DATE].values,
        COL_ACTUAL: y_all.values,
        COL_REPLAY: replay_pred,
    }).sort_values(COL_DATE).reset_index(drop=True)
    replay[COL_ERROR] = replay[COL_REPLAY] - replay[COL_ACTUAL]
    return replay


def build_time_series_figure(df: pd.DataFrame, pred_col: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[COL_DATE], y=df[COL_ACTUAL], mode="lines", name="\u5b9e\u9645\u503c", line=dict(color="#1f77b4", width=2.5)))
    fig.add_trace(go.Scatter(x=df[COL_DATE], y=df[pred_col], mode="lines", name="\u9884\u6d4b\u503c", line=dict(color="#d62728", width=2)))
    fig.update_layout(title=title, xaxis_title="\u65e5\u671f", yaxis_title="\u6295\u77fe\u503c", hovermode="x unified", height=480)
    return fig


def build_scatter_figure(df: pd.DataFrame, pred_col: str, title: str) -> go.Figure:
    low = min(df[COL_ACTUAL].min(), df[pred_col].min())
    high = max(df[COL_ACTUAL].max(), df[pred_col].max())
    fig = px.scatter(df, x=COL_ACTUAL, y=pred_col, color_discrete_sequence=["#ff7f0e"], opacity=0.7, title=title)
    fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="y=x \u53c2\u8003\u7ebf", line=dict(color="#2ca02c", dash="dash")))
    fig.update_layout(height=420)
    return fig


def build_error_figure(df: pd.DataFrame, pred_col: str, title: str) -> go.Figure:
    temp = df.copy()
    temp[COL_ERROR] = temp[pred_col] - temp[COL_ACTUAL]
    fig = px.histogram(temp, x=COL_ERROR, nbins=30, color_discrete_sequence=["#9467bd"], title=title)
    fig.update_layout(height=420)
    return fig


def build_importance_figure(importance_df: pd.DataFrame, metric_col: str, top_n: int) -> go.Figure:
    plot_df = importance_df.head(top_n).sort_values(metric_col, ascending=True)
    fig = px.bar(plot_df, x=metric_col, y=COL_FEATURE, orientation="h", color=metric_col, color_continuous_scale="Blues", title=f"\u7279\u5f81\u91cd\u8981\u6027 Top {top_n}")
    fig.update_layout(height=540, coloraxis_showscale=False)
    return fig


def build_history_figure(df: pd.DataFrame, columns: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for idx, col in enumerate(columns):
        fig.add_trace(go.Scatter(x=df[COL_DATE], y=df[col], mode="lines", name=col, line=dict(color=palette[idx % len(palette)], width=2)))
    fig.update_layout(title=title, xaxis_title="\u65e5\u671f", yaxis_title="\u6570\u503c", hovermode="x unified", height=500)
    return fig


def ensure_outputs_exist() -> None:
    required = [
        RESULTS_DIR / "test_predictions.xlsx",
        RESULTS_DIR / "model_metrics.xlsx",
        RESULTS_DIR / "comparison_metrics.xlsx",
        RESULTS_DIR / "feature_importance.xlsx",
        DATA_DIR / "cleaned_merged_dataset.csv",
        DATA_DIR / "feature_engineered_dataset.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        st.error("\u672a\u627e\u5230\u5b8c\u6574\u8f93\u51fa\u7ed3\u679c\uff0c\u8bf7\u5148\u5728\u9879\u76ee\u76ee\u5f55\u6267\u884c `python run_pipeline.py`\u3002")
        st.code("python run_pipeline.py")
        st.write(missing)
        st.stop()


ensure_outputs_exist()
predictions = load_predictions()
metrics_df = load_metrics()
comparison_df = load_comparison()
importance_df = load_importance()
generalization_df = load_generalization()
operating_df = load_operating_conditions()
feature_df = load_feature_data()
cleaned_df = load_cleaned_data()
replay_df = load_replay_predictions()
params_text = load_best_params_text()

prediction_options = {
    "Huber-ElasticNet \u878d\u5408\uff08\u63a8\u8350\uff09": COL_RECOMMENDED,
    "\u4f18\u5316 LightGBM": COL_LGBM,
    "\u672a\u4f18\u5316 LightGBM": COL_LGBM_DEFAULT,
    "\u7ebf\u6027\u56de\u5f52": COL_LINEAR,
    "\u5e42\u51fd\u6570\u56de\u5f52": COL_POWER,
}
importance_options = {
    "\u878d\u5408\u7ebf\u6027\u7cfb\u6570": COL_BLEND_IMPORTANCE,
    "ElasticNet \u7cfb\u6570": COL_ENET_IMPORTANCE,
    "Huber \u7cfb\u6570": COL_HUBER_IMPORTANCE,
    "LightGBM \u91cd\u8981\u6027": COL_LGBM_IMPORTANCE,
}
history_source_options = {
    "\u7279\u5f81\u5de5\u7a0b\u6570\u636e": feature_df,
    "\u6e05\u6d17\u540e\u5408\u5e76\u6570\u636e": cleaned_df,
}

st.title("\u6df7\u51dd\u667a\u80fd\u6295\u836f\u6a21\u578b\u53ef\u89c6\u5316\u770b\u677f")
st.caption("\u770b\u677f\u73b0\u5728\u540c\u65f6\u652f\u6301\u201c\u6d4b\u8bd5\u96c6\u9884\u6d4b\u6548\u679c\u201d\u548c\u201c2021-2026 \u5168\u5386\u53f2\u56de\u653e\u201d\u4e24\u79cd\u89c6\u56fe\u3002")

history_date_min = min(cleaned_df[COL_DATE].min().date(), replay_df[COL_DATE].min().date())
history_date_max = max(cleaned_df[COL_DATE].max().date(), replay_df[COL_DATE].max().date())
pred_date_min = predictions[COL_DATE].min().date()
pred_date_max = predictions[COL_DATE].max().date()

with st.sidebar:
    st.header("\u63a7\u5236\u9762\u677f")
    selected_model_label = st.selectbox("\u9009\u62e9\u5c55\u793a\u6a21\u578b", list(prediction_options.keys()), index=0)
    selected_pred_col = prediction_options[selected_model_label]
    selected_importance_label = st.selectbox("\u7279\u5f81\u89e3\u91ca\u53e3\u5f84", list(importance_options.keys()), index=0)
    importance_col = importance_options[selected_importance_label]
    history_source_label = st.selectbox("\u5386\u53f2\u6570\u636e\u6e90", list(history_source_options.keys()), index=0)
    history_source_df = history_source_options[history_source_label]
    history_candidates = [c for c in HISTORY_FEATURE_CANDIDATES if c in history_source_df.columns]
    if not history_candidates:
        history_candidates = [c for c in history_source_df.columns if c != COL_DATE and pd.api.types.is_numeric_dtype(history_source_df[c])][:6]
    selected_history_cols = st.multiselect("\u5168\u5386\u53f2\u8d8b\u52bf\u5b57\u6bb5", history_candidates, default=history_candidates[: min(4, len(history_candidates))])
    top_n = st.slider("\u7279\u5f81\u91cd\u8981\u6027\u663e\u793a\u6570\u91cf", 5, 30, 15, 1)
    window_size = st.slider("\u6d4b\u8bd5\u96c6\u65f6\u5e8f\u7a97\u53e3\u70b9\u6570", 30, min(300, len(predictions)), min(120, len(predictions)), 10)
    default_history_range = st.session_state.get("history_date_range_v2", (history_date_min, history_date_max))
    if not isinstance(default_history_range, tuple) or len(default_history_range) != 2:
        default_history_range = (history_date_min, history_date_max)
    start_default = max(history_date_min, min(default_history_range[0], history_date_max))
    end_default = max(start_default, min(default_history_range[1], history_date_max))
    date_range = st.date_input(
        "\u65e5\u671f\u8303\u56f4\uff08\u5168\u5386\u53f2\uff09",
        value=(start_default, end_default),
        min_value=history_date_min,
        max_value=history_date_max,
        key="history_date_range_v2",
    )
    st.caption(f"\u5168\u5386\u53f2\u53ef\u9009\u533a\u95f4\uff1a{history_date_min} ~ {history_date_max}")
    st.caption(f"\u72ec\u7acb\u6d4b\u8bd5\u96c6\u9884\u6d4b\u53ef\u7528\u533a\u95f4\uff1a{pred_date_min} ~ {pred_date_max}")
    st.markdown("---")
    st.code("python run_pipeline.py")
    st.code("streamlit run dashboard_app.py")

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = history_date_min, history_date_max
start_date = max(history_date_min, min(start_date, history_date_max))
end_date = max(start_date, min(end_date, history_date_max))

history_filtered = history_source_df[(history_source_df[COL_DATE].dt.date >= start_date) & (history_source_df[COL_DATE].dt.date <= end_date)].copy()
replay_filtered = replay_df[(replay_df[COL_DATE].dt.date >= start_date) & (replay_df[COL_DATE].dt.date <= end_date)].copy()
prediction_filtered = predictions[(predictions[COL_DATE].dt.date >= start_date) & (predictions[COL_DATE].dt.date <= end_date)].copy()
if not prediction_filtered.empty:
    prediction_filtered = prediction_filtered.tail(window_size)
    prediction_filtered[COL_ERROR] = prediction_filtered[selected_pred_col] - prediction_filtered[COL_ACTUAL]

recommended_row = comparison_df.loc[comparison_df[COL_MODEL] == NAME_RECOMMENDED]
if recommended_row.empty:
    recommended_row = comparison_df.iloc[[0]]
recommended_row = recommended_row.iloc[0]

metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("\u5f53\u524d\u5c55\u793a\u6a21\u578b", selected_model_label)
with metric_cols[1]:
    st.metric("\u63a8\u8350\u6a21\u578b RMSE", f"{float(recommended_row['RMSE']):.4f}")
with metric_cols[2]:
    st.metric("\u63a8\u8350\u6a21\u578b MAE", f"{float(recommended_row['MAE']):.4f}")
with metric_cols[3]:
    st.metric("\u63a8\u8350\u6a21\u578b R?", f"{float(recommended_row['R2']):.4f}")

summary_tab, replay_tab, compare_tab, feature_tab, data_tab = st.tabs([
    "\u6d4b\u8bd5\u96c6\u9884\u6d4b\u6548\u679c",
    "\u5168\u5386\u53f2\u56de\u653e",
    "\u5bf9\u6bd4\u5206\u6790",
    "\u7279\u5f81\u4e0e\u5de5\u51b5",
    "\u6570\u636e\u6d4f\u89c8",
])

with summary_tab:
    st.caption(f"\u8fd9\u4e00\u9875\u53ea\u5c55\u793a\u72ec\u7acb\u6d4b\u8bd5\u96c6\u9884\u6d4b\uff0c\u56e0\u6b64\u53ef\u7528\u65f6\u95f4\u662f {pred_date_min} ~ {pred_date_max}\u3002")
    if prediction_filtered.empty:
        st.info("\u5f53\u524d\u9009\u62e9\u7684\u65e5\u671f\u8303\u56f4\u6ca1\u6709\u72ec\u7acb\u6d4b\u8bd5\u96c6\u9884\u6d4b\u7ed3\u679c\u3002\u8bf7\u5207\u5230\u201c\u5168\u5386\u53f2\u56de\u653e\u201d\u67e5\u770b 2021-2026 \u7684\u6574\u4f53\u66f2\u7ebf\u3002")
    else:
        left, right = st.columns([1.6, 1])
        with left:
            st.plotly_chart(build_time_series_figure(prediction_filtered, selected_pred_col, "\u6d4b\u8bd5\u96c6\u9884\u6d4b\u503c\u4e0e\u5b9e\u9645\u503c\u65f6\u5e8f\u5bf9\u6bd4"), use_container_width=True)
        with right:
            st.plotly_chart(build_scatter_figure(prediction_filtered, selected_pred_col, "\u6d4b\u8bd5\u96c6\u9884\u6d4b\u503c vs \u5b9e\u9645\u503c\u6563\u70b9\u56fe"), use_container_width=True)
        lower_left, lower_right = st.columns([1, 1])
        with lower_left:
            st.plotly_chart(build_error_figure(prediction_filtered, selected_pred_col, "\u6d4b\u8bd5\u96c6\u9884\u6d4b\u8bef\u5dee\u5206\u5e03"), use_container_width=True)
        with lower_right:
            preview = prediction_filtered[[COL_DATE, COL_ACTUAL, selected_pred_col, COL_ERROR]].rename(columns={selected_pred_col: "\u9884\u6d4b\u503c"})
            st.subheader("\u5f53\u524d\u6d4b\u8bd5\u7a97\u53e3\u6570\u636e")
            st.dataframe(preview, use_container_width=True, hide_index=True)

with replay_tab:
    st.caption("\u8fd9\u4e00\u9875\u662f\u5168\u5386\u53f2\u56de\u653e\uff0c\u8986\u76d6 2021-2026 \u6574\u4e2a\u65f6\u95f4\u6bb5\u3002\u7531\u4e8e\u5305\u542b\u8bad\u7ec3\u671f\uff0c\u7528\u4e8e\u89c2\u5bdf\u66f2\u7ebf\u8d34\u5408\u5ea6\uff0c\u4e0d\u4f5c\u4e3a\u72ec\u7acb\u6d4b\u8bd5\u6307\u6807\u3002")
    if replay_filtered.empty:
        st.info("\u5f53\u524d\u9009\u62e9\u7684\u65e5\u671f\u8303\u56f4\u6ca1\u6709\u5386\u53f2\u56de\u653e\u6570\u636e\u3002")
    else:
        left, right = st.columns([1.6, 1])
        with left:
            st.plotly_chart(build_time_series_figure(replay_filtered, COL_REPLAY, "\u5168\u5386\u53f2\u5b9e\u9645\u503c\u4e0e\u56de\u653e\u9884\u6d4b\u5bf9\u6bd4"), use_container_width=True)
        with right:
            st.plotly_chart(build_scatter_figure(replay_filtered, COL_REPLAY, "\u5168\u5386\u53f2\u56de\u653e\u6563\u70b9\u56fe"), use_container_width=True)
        st.subheader("\u5168\u5386\u53f2\u5de5\u827a\u6307\u6807\u8d8b\u52bf")
        if selected_history_cols:
            st.plotly_chart(build_history_figure(history_filtered, selected_history_cols, f"{history_source_label}\u65f6\u95f4\u8d8b\u52bf"), use_container_width=True)
        else:
            st.info("\u8bf7\u5728\u5de6\u4fa7\u9009\u62e9\u81f3\u5c11\u4e00\u4e2a\u5168\u5386\u53f2\u8d8b\u52bf\u5b57\u6bb5\u3002")
        replay_preview = replay_filtered[[COL_DATE, COL_ACTUAL, COL_REPLAY, COL_ERROR]].rename(columns={COL_REPLAY: "\u56de\u653e\u9884\u6d4b\u503c"})
        st.subheader("\u5168\u5386\u53f2\u56de\u653e\u6570\u636e")
        st.dataframe(replay_preview.tail(200), use_container_width=True, hide_index=True)

with compare_tab:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("\u6a21\u578b\u8bc4\u4f30\u6307\u6807")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    with col2:
        comp_plot = px.bar(comparison_df.sort_values("RMSE"), x=COL_MODEL, y=["RMSE", "MAE"], barmode="group", title="RMSE / MAE \u5bf9\u6bd4")
        st.plotly_chart(comp_plot, use_container_width=True)
    if not generalization_df.empty:
        st.subheader("\u5de5\u51b5\u6cdb\u5316\u8868\u73b0")
        gen_plot = px.bar(generalization_df, x="\u5de5\u51b5", y=["RMSE", "MAE", "R2"], barmode="group", title="\u4e0d\u540c\u5de5\u51b5\u4e0b\u7684\u63a8\u8350\u6a21\u578b\u8868\u73b0")
        st.plotly_chart(gen_plot, use_container_width=True)
        st.dataframe(generalization_df, use_container_width=True, hide_index=True)
    st.subheader("\u6a21\u578b\u914d\u7f6e\u6458\u8981")
    st.code(params_text, language="json")

with feature_tab:
    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(build_importance_figure(importance_df, importance_col, top_n=top_n), use_container_width=True)
    with right:
        if not operating_df.empty:
            st.subheader("\u5de5\u51b5\u7edf\u8ba1")
            st.dataframe(operating_df, use_container_width=True, hide_index=True)
            operating_plot = px.bar(operating_df, x="\u5de5\u51b5", y="\u5e73\u5747\u6295\u77fe", color="\u5e73\u5747\u6d4a\u5ea6", title="\u4e0d\u540c\u5de5\u51b5\u5e73\u5747\u6295\u77fe", color_continuous_scale="YlOrRd")
            st.plotly_chart(operating_plot, use_container_width=True)
    st.subheader("\u7279\u5f81\u91cd\u8981\u6027\u660e\u7ec6")
    st.dataframe(importance_df, use_container_width=True, hide_index=True)

with data_tab:
    st.subheader("\u6e05\u6d17\u540e\u5408\u5e76\u6570\u636e")
    st.dataframe(cleaned_df.head(200), use_container_width=True, hide_index=True)
    st.subheader("\u7279\u5f81\u5de5\u7a0b\u6570\u636e")
    st.dataframe(feature_df.head(200), use_container_width=True, hide_index=True)
    st.subheader("\u8fd9\u4e9b\u6570\u636e\u600e\u4e48\u7528")
    st.markdown(
        """
        1. `outputs/data/cleaned_merged_dataset.csv`?????????????
        2. `outputs/data/feature_engineered_dataset.csv`?????????????????????????
        3. `outputs/results/test_predictions.xlsx`??????????????????????
        4. `outputs/results/feature_importance.xlsx`???????????????????
        5. `outputs/model/recommended_blend_model.pkl`????????????????
        6. `outputs/model/optimized_lightgbm.pkl`????????????????
        7. ???????????? 2021-2026 ????????????????????
        """
    )

st.markdown("---")
st.info("\u5982\u679c\u4f60\u91cd\u65b0\u8bad\u7ec3\u6a21\u578b\uff0c\u53ea\u9700\u518d\u6b21\u6267\u884c `python run_pipeline.py`\uff0c\u5237\u65b0\u754c\u9762\u540e\u5373\u53ef\u770b\u5230\u65b0\u7684\u9884\u6d4b\u6548\u679c\u3002")
