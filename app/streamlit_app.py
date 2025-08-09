# app/streamlit_app.py
# --- make imports work locally and on Streamlit Cloud ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.config import DATA_DIR, DEFAULT_INPUT, DEFAULT_CLEANED
from src.data_loader import load_csv
from src.cleaning import standardize_columns, fill_missing

st.set_page_config(page_title="Marketing Campaign Analysis (No ROI)", page_icon="📊", layout="wide")
st.title("📊 Marketing Campaign Analysis — Non‑ROI Metrics")

# ----------------------------------
# Sidebar controls
# ----------------------------------
with st.sidebar:
    st.header("Data Source")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"])

    st.markdown("---")
    st.header("Cleaning")
    missing_strategy = st.selectbox(
        "Missing value strategy (numeric)",
        ["median", "mean", "zero"],
        index=0
    )

    st.markdown("---")
    st.header("Column Mapping")
    st.caption("Adjust to match your cleaned column names (lowercase, underscores).")
    clicks_col = st.text_input("Clicks column", value="clicks")
    imps_col = st.text_input("Impressions column", value="impressions")
    cost_col = st.text_input("Cost column", value="acquisition_cost")
    conv_rate_col = st.text_input("Conversion rate column", value="conversion_rate")
    engagement_col = st.text_input("Engagement score column", value="engagement_score")
    channel_col = st.text_input("Channel column", value="channel_used")
    campaign_col = st.text_input("Campaign column", value="campaign_type")
    date_col = st.text_input("Date column (optional)", value="date")

# ----------------------------------
# Load data
# ----------------------------------
if uploaded is not None:
    df = pd.read_csv(uploaded)
    source_msg = "Uploaded file"
else:
    default_clean = DATA_DIR / DEFAULT_CLEANED
    default_raw = DATA_DIR / DEFAULT_INPUT
    if default_clean.exists():
        df = load_csv(default_clean)
        source_msg = f"Default file: {default_clean}"
    elif default_raw.exists():
        df = load_csv(default_raw)
        source_msg = f"Default file: {default_raw}"
    else:
        st.warning("No file uploaded and no default CSV found. Upload a CSV using the sidebar.")
        st.stop()

st.caption(source_msg)

# ----------------------------------
# Preview
# ----------------------------------
st.subheader("Preview")
st.dataframe(df.head(), use_container_width=True)

# ----------------------------------
# Cleaning
# ----------------------------------
with st.expander("Cleaning", expanded=False):
    st.write("Column names will be standardized (lowercase, underscores).")
    df = standardize_columns(df)

    strategy_key = "median" if missing_strategy in ("median", "mean") else "zero"
    if missing_strategy == "mean":
        strategy_key = "mean"
    df = fill_missing(df, strategy=strategy_key)

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# ----------------------------------
# Derived metrics (NO ROI)
# ----------------------------------
# Create CTR (%), CPC, CPM if the necessary columns exist
if clicks_col in df.columns and imps_col in df.columns:
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ctr_pct"] = (
            pd.to_numeric(df[clicks_col], errors="coerce")
            / pd.to_numeric(df[imps_col], errors="coerce")
        ) * 100.0

if cost_col in df.columns and clicks_col in df.columns:
    with np.errstate(divide="ignore", invalid="ignore"):
        df["cpc"] = (
            pd.to_numeric(df[cost_col], errors="coerce")
            / pd.to_numeric(df[clicks_col], errors="coerce")
        )
        df.loc[~np.isfinite(df["cpc"]), "cpc"] = np.nan

if cost_col in df.columns and imps_col in df.columns:
    with np.errstate(divide="ignore", invalid="ignore"):
        df["cpm"] = (
            pd.to_numeric(df[cost_col], errors="coerce")
            / (pd.to_numeric(df[imps_col], errors="coerce") / 1000.0)
        )
        df.loc[~np.isfinite(df["cpm"]), "cpm"] = np.nan

# ----------------------------------
# Filters
# ----------------------------------
st.subheader("Filters")

def unique_vals(col):
    return sorted([v for v in df[col].dropna().unique().tolist()]) if col in df.columns else []

left, mid, right = st.columns(3)

with left:
    channel_sel = st.multiselect(
        "Channel",
        options=unique_vals(channel_col),
        default=unique_vals(channel_col)[:5] if len(unique_vals(channel_col)) > 5 else unique_vals(channel_col),
    )
with mid:
    campaign_sel = st.multiselect(
        "Campaign",
        options=unique_vals(campaign_col),
        default=unique_vals(campaign_col)[:5] if len(unique_vals(campaign_col)) > 5 else unique_vals(campaign_col),
    )
with right:
    if date_col in df.columns:
        min_d = pd.to_datetime(df[date_col], errors="coerce").min()
        max_d = pd.to_datetime(df[date_col], errors="coerce").max()
        date_range = st.date_input("Date range", value=(min_d, max_d))
    else:
        date_range = None

df_filt = df.copy()
if channel_sel and channel_col in df_filt.columns:
    df_filt = df_filt[df_filt[channel_col].isin(channel_sel)]
if campaign_sel and campaign_col in df_filt.columns:
    df_filt = df_filt[df_filt[campaign_col].isin(campaign_sel)]
if date_range and date_col in df_filt.columns:
    dmin = pd.to_datetime(date_range[0])
    dmax = pd.to_datetime(date_range[1])
    df_filt = df_filt[
        (pd.to_datetime(df_filt[date_col], errors="coerce") >= dmin)
        & (pd.to_datetime(df_filt[date_col], errors="coerce") <= dmax)
    ]

st.caption(f"Filtered rows: {len(df_filt):,} (of {len(df):,})")

# ----------------------------------
# KPIs (filtered)
# ----------------------------------
st.subheader("Key Metrics")

def sum_safe_local(data, col):
    return float(pd.to_numeric(data[col], errors="coerce").sum()) if col in data.columns else 0.0

def mean_safe_local(data, col):
    s = pd.to_numeric(data[col], errors="coerce") if col in data.columns else pd.Series(dtype=float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.mean()) if not s.empty else 0.0

total_rows = len(df_filt)
total_cost = sum_safe_local(df_filt, cost_col)
total_clicks = sum_safe_local(df_filt, clicks_col)
total_imps = sum_safe_local(df_filt, imps_col)
avg_ctr = mean_safe_local(df_filt, "ctr_pct")
avg_cpc = mean_safe_local(df_filt, "cpc")
avg_cpm = mean_safe_local(df_filt, "cpm")
avg_conv_rate = mean_safe_local(df_filt, conv_rate_col)
avg_engagement = mean_safe_local(df_filt, engagement_col)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Filtered Rows", f"{total_rows:,}")
k2.metric("Total Cost", f"{total_cost:,.0f}")
k3.metric("Total Clicks", f"{total_clicks:,.0f}")
k4.metric("Total Impressions", f"{total_imps:,.0f}")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Avg CTR (%)", f"{avg_ctr:.2f}")
k6.metric("Avg CPC", f"{avg_cpc:.2f}")
k7.metric("Avg CPM", f"{avg_cpm:.2f}")
k8.metric("Avg Conversion Rate (%)", f"{avg_conv_rate:.2f}")

if engagement_col in df_filt.columns:
    k9, _ = st.columns([1, 3])
    k9.metric("Avg Engagement Score", f"{avg_engagement:.2f}")

# ----------------------------------
# Visualizations (use df_filt)
# ----------------------------------
st.subheader("Visualizations")

# CTR distribution
if "ctr_pct" in df_filt.columns:
    fig = px.histogram(df_filt, x="ctr_pct", nbins=30, title="CTR (%) Distribution")
    s = pd.to_numeric(df_filt["ctr_pct"], errors="coerce").dropna()
    if not s.empty:
        q1, q99 = s.quantile([0.01, 0.99])
        if q1 < q99:
            fig.update_xaxes(range=[q1, q99])
    fig.update_layout(xaxis_title="CTR (%)", yaxis_title="Count", margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

# Cost by Category
cat_candidates = [c for c in df_filt.columns if df_filt[c].dtype == "object" or str(df_filt[c].dtype).startswith("category")]
if cat_candidates and cost_col in df_filt.columns:
    st.markdown("##### Cost by Category")
    c1, c2 = st.columns([2, 1])
    with c1:
        cat_col = st.selectbox("Category", options=sorted(cat_candidates), index=cat_candidates.index(channel_col) if channel_col in cat_candidates else 0)
    with c2:
        agg_cost = st.selectbox("Aggregation", options=["sum", "mean", "median"], index=0)
    grp = df_filt.groupby(cat_col, dropna=False)[cost_col]
    agg_map = {"sum": grp.sum(), "mean": grp.mean(), "median": grp.median()}
    data = agg_map[agg_cost].sort_values(ascending=False).reset_index()
    fig = px.bar(data, x=cat_col, y=cost_col, title=f"{cost_col} ({agg_cost}) by {cat_col}", text=cost_col)
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title=cat_col, yaxis_title=cost_col, margin=dict(l=10, r=10, t=50, b=60))
    st.plotly_chart(fig, use_container_width=True)

# CTR by Category
if "ctr_pct" in df_filt.columns and cat_candidates:
    st.markdown("##### CTR by Category")
    cat2 = st.selectbox("Category for CTR", options=sorted(cat_candidates), index=cat_candidates.index(channel_col) if channel_col in cat_candidates else 0, key="ctr_cat")
    data = df_filt.groupby(cat2, dropna=False)["ctr_pct"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(data, x=cat2, y="ctr_pct", title=f"CTR (%) by {cat2}", text="ctr_pct")
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title=cat2, yaxis_title="CTR (%)", margin=dict(l=10, r=10, t=50, b=60))
    st.plotly_chart(fig, use_container_width=True)

# Cost vs Clicks scatter (bubble = Impressions)
if all(c in df_filt.columns for c in [cost_col, clicks_col]) and imps_col in df_filt.columns:
    st.markdown("##### Cost vs Clicks (bubble = Impressions)")
    fig = px.scatter(
        df_filt,
        x=cost_col,
        y=clicks_col,
        size=imps_col,
        hover_data=[c for c in [campaign_col, channel_col, "company"] if c in df_filt.columns],
        title="Cost vs Clicks",
    )
    fig.update_layout(xaxis_title=cost_col, yaxis_title=clicks_col, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

# Time Series
num_candidates = [c for c in df_filt.columns if pd.api.types.is_numeric_dtype(df_filt[c])]
ts_options = [c for c in [imps_col, clicks_col, cost_col, "ctr_pct", engagement_col, conv_rate_col] if c in num_candidates]
if date_col in df_filt.columns and ts_options:
    st.markdown("##### Time Series")
    t1, t2, t3 = st.columns([2, 2, 1])
    with t1:
        ts_metric = st.selectbox("Metric", options=ts_options, index=0)
    with t2:
        freq = st.selectbox("Resample", options=[None, "D", "W", "M", "Q"], index=2)
    tmp = df_filt[[date_col, ts_metric]].dropna()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    if not tmp.empty:
        if freq:
            tmp = tmp.set_index(date_col).resample(freq)[ts_metric].sum().reset_index()
        fig = px.line(tmp, x=date_col, y=ts_metric, title=f"{ts_metric} over time")
        fig.update_layout(xaxis_title="Date", yaxis_title=ts_metric, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------
# Rankings: Top/Bottom tables
# ----------------------------------
st.subheader("Rankings")

def show_top_bottom(metric_col: str, label: str, higher_is_better: bool = True, n: int = 10):
    if metric_col not in df_filt.columns:
        return
    s = pd.to_numeric(df_filt[metric_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    tmp = df_filt.assign(_metric=s).dropna(subset=["_metric"]).copy()
    if tmp.empty:
        return

    id_cols = [c for c in ["campaign_type", "channel_used", "company"] if c in tmp.columns]
    display_cols = id_cols + [metric_col]
    for c in [cost_col, clicks_col, imps_col]:
        if c in tmp.columns and c not in display_cols:
            display_cols.append(c)

    if higher_is_better:
        top = tmp.sort_values("_metric", ascending=False).head(n)[display_cols]
        bottom = tmp.sort_values("_metric", ascending=True).head(n)[display_cols]
    else:
        top = tmp.sort_values("_metric", ascending=True).head(n)[display_cols]
        bottom = tmp.sort_values("_metric", ascending=False).head(n)[display_cols]

    c1, c2 = st.columns(2)
    c1.markdown(f"**Top {n} — {label}**")
    c1.dataframe(top.reset_index(drop=True), use_container_width=True)
    c2.markdown(f"**Bottom {n} — {label}**")
    c2.dataframe(bottom.reset_index(drop=True), use_container_width=True)

if "ctr_pct" in df_filt.columns:
    show_top_bottom("ctr_pct", "CTR (%)", higher_is_better=True, n=10)
if "cpc" in df_filt.columns:
    show_top_bottom("cpc", "CPC", higher_is_better=False, n=10)
if "cpm" in df_filt.columns:
    show_top_bottom("cpm", "CPM", higher_is_better=False, n=10)
if engagement_col in df_filt.columns:
    show_top_bottom(engagement_col, "Engagement Score", higher_is_better=True, n=10)

# ----------------------------------
# Downloads
# ----------------------------------
st.markdown("---")
colA, colB = st.columns(2)
with colA:
    st.download_button(
        "Download PROCESSED CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="processed_no_roi.csv",
        mime="text/csv",
    )
with colB:
    st.download_button(
        "Download FILTERED CSV",
        df_filt.to_csv(index=False).encode("utf-8"),
        file_name="filtered_metrics.csv",
        mime="text/csv",
    )

