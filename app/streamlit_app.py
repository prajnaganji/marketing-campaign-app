# app/streamlit_app.py
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

st.set_page_config(page_title="Marketing Campaign Analysis", page_icon="📊", layout="wide")
st.title("📊 Marketing Campaign Analysis")

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Data Source")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"])

    st.markdown("---")
    st.header("Cleaning")
    missing_strategy = st.selectbox("Missing value strategy (numeric)", ["median", "mean", "zero"], index=0)

    st.markdown("---")
    st.header("Column Mapping")
    clicks_col = st.text_input("Clicks column", value="clicks")
    imps_col = st.text_input("Impressions column", value="impressions")
    cost_col = st.text_input("Cost column", value="acquisition_cost")
    conv_rate_col = st.text_input("Conversion rate column", value="conversion_rate")
    engagement_col = st.text_input("Engagement score column", value="engagement_score")
    channel_col = st.text_input("Channel column", value="channel_used")
    campaign_col = st.text_input("Campaign column", value="campaign_type")
    date_col = st.text_input("Date column (optional)", value="date")

# ---------------------------
# Load Data
# ---------------------------
if uploaded is not None:
    df = pd.read_csv(uploaded)
    source_msg = "Uploaded file"
else:
    default_clean = DATA_DIR / DEFAULT_CLEANED
    default_raw = DATA_DIR / DEFAULT_INPUT
    if default_clean.exists():
        df = load_csv(default_clean); source_msg = f"Default file: {default_clean}"
    elif default_raw.exists():
        df = load_csv(default_raw); source_msg = f"Default file: {default_raw}"
    else:
        st.warning("No file uploaded and no default CSV found."); st.stop()

st.caption(source_msg)

# ---------------------------
# Preview
# ---------------------------
st.subheader("Preview")
st.dataframe(df.head(), use_container_width=True)

# ---------------------------
# Cleaning
# ---------------------------
with st.expander("Cleaning", expanded=False):
    df = standardize_columns(df)
    strategy_key = "median" if missing_strategy in ("median", "mean") else "zero"
    df = fill_missing(df, strategy=strategy_key)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# ---------------------------
# Derived Metrics (no ROI)
# ---------------------------
if clicks_col in df.columns and imps_col in df.columns:
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ctr_pct"] = pd.to_numeric(df[clicks_col], errors="coerce") / pd.to_numeric(df[imps_col], errors="coerce") * 100

if cost_col in df.columns and clicks_col in df.columns:
    with np.errstate(divide="ignore", invalid="ignore"):
        df["cpc"] = pd.to_numeric(df[cost_col], errors="coerce") / pd.to_numeric(df[clicks_col], errors="coerce")
        df.loc[~np.isfinite(df["cpc"]), "cpc"] = np.nan

if cost_col in df.columns and imps_col in df.columns:
    with np.errstate(divide="ignore", invalid="ignore"):
        df["cpm"] = pd.to_numeric(df[cost_col], errors="coerce") / (pd.to_numeric(df[imps_col], errors="coerce") / 1000)
        df.loc[~np.isfinite(df["cpm"]), "cpm"] = np.nan

# ---------------------------
# Filters
# ---------------------------
st.subheader("Filters")
def unique_vals(col):
    return sorted(df[col].dropna().unique()) if col in df.columns else []

left, mid, right = st.columns(3)
with left:
    channel_sel = st.multiselect("Channel", options=unique_vals(channel_col))
with mid:
    campaign_sel = st.multiselect("Campaign", options=unique_vals(campaign_col))
with right:
    if date_col in df.columns:
        min_d, max_d = pd.to_datetime(df[date_col]).min(), pd.to_datetime(df[date_col]).max()
        date_range = st.date_input("Date range", value=(min_d, max_d))
    else:
        date_range = None

df_filt = df.copy()
if channel_sel and channel_col in df_filt.columns:
    df_filt = df_filt[df_filt[channel_col].isin(channel_sel)]
if campaign_sel and campaign_col in df_filt.columns:
    df_filt = df_filt[df_filt[campaign_col].isin(campaign_sel)]
if date_range and date_col in df_filt.columns:
    dmin, dmax = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df_filt = df_filt[(pd.to_datetime(df_filt[date_col]) >= dmin) & (pd.to_datetime(df_filt[date_col]) <= dmax)]

# ---------------------------
# KPIs (pretty formatting)
# ---------------------------
st.subheader("Key Metrics")

def sum_safe(col):
    return float(pd.to_numeric(df_filt[col], errors="coerce").sum()) if col in df_filt.columns else 0.0

def mean_safe(col):
    s = pd.to_numeric(df_filt[col], errors="coerce") if col in df_filt.columns else pd.Series(dtype=float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.mean()) if not s.empty else 0.0

def format_number(v: float) -> str:
    v = float(v)
    if abs(v) >= 1_000_000_000: return f"{v/1_000_000_000:.2f}B"
    if abs(v) >= 1_000_000:     return f"{v/1_000_000:.2f}M"
    if abs(v) >= 1_000:         return f"{v/1_000:.2f}K"
    return f"{v:,.0f}"

def format_decimal(v: float) -> str:
    return f"{float(v):,.2f}"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Cost", format_number(sum_safe(cost_col)))
c2.metric("Total Clicks", format_number(sum_safe(clicks_col)))
c3.metric("Total Impressions", format_number(sum_safe(imps_col)))
c4.metric("Avg CTR (%)", format_decimal(mean_safe("ctr_pct")))

c5, c6, c7 = st.columns(3)
c5.metric("Avg CPC", format_decimal(mean_safe("cpc")))
c6.metric("Avg CPM", format_decimal(mean_safe("cpm")))
c7.metric("Avg Conversion Rate (%)", format_decimal(mean_safe(conv_rate_col)))

# ---------------------------
# Visualizations
# ---------------------------
st.subheader("Visualizations")

# CTR distribution (zoom to 1–99 percentile to avoid flat bars)
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
cat_cols = [c for c in df_filt.columns if df_filt[c].dtype == "object"]
if cost_col in df_filt.columns and cat_cols:
    cat1 = st.selectbox("Category for Cost", options=cat_cols, index=0)
    data = df_filt.groupby(cat1, dropna=False)[cost_col].mean().reset_index()
    fig = px.bar(data, x=cat1, y=cost_col, title=f"Average {cost_col} by {cat1}", text=cost_col)
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title=cat1, yaxis_title=cost_col, margin=dict(l=10, r=10, t=50, b=60))
    st.plotly_chart(fig, use_container_width=True)

# CTR by Category
if "ctr_pct" in df_filt.columns and cat_cols:
    cat2 = st.selectbox("Category for CTR", options=cat_cols, index=0, key="ctr_cat")
    data = df_filt.groupby(cat2, dropna=False)["ctr_pct"].mean().reset_index()
    fig = px.bar(data, x=cat2, y="ctr_pct", title=f"Average CTR (%) by {cat2}", text="ctr_pct")
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title=cat2, yaxis_title="CTR (%)", margin=dict(l=10, r=10, t=50, b=60))
    st.plotly_chart(fig, use_container_width=True)

# Time Series
num_candidates = [c for c in [imps_col, clicks_col, cost_col, "ctr_pct", engagement_col, conv_rate_col] if c in df_filt.columns and pd.api.types.is_numeric_dtype(df_filt[c])]
if date_col in df_filt.columns and num_candidates:
    metric_choice = st.selectbox("Time‑series metric", options=num_candidates, index=0)
    freq = st.selectbox("Resample", options=[None, "D", "W", "M", "Q"], index=2)
    tmp = df_filt[[date_col, metric_choice]].dropna()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    if not tmp.empty:
        if freq:
            tmp = tmp.set_index(date_col).resample(freq)[metric_choice].sum().reset_index()
        fig = px.line(tmp, x=date_col, y=metric_choice, title=f"{metric_choice} over time")
        fig.update_layout(xaxis_title="Date", yaxis_title=metric_choice, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Download
# ---------------------------
st.markdown("---")
st.download_button(
    "Download Filtered Data",
    df_filt.to_csv(index=False).encode("utf-8"),
    "filtered_data.csv",
    "text/csv",
)



