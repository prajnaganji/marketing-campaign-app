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
    missing_strategy = st.selectbox(
        "Missing value strategy (numeric)", ["median", "mean", "zero"], index=0
    )

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
        df = load_csv(default_clean)
        source_msg = f"Default file: {default_clean}"
    elif default_raw.exists():
        df = load_csv(default_raw)
        source_msg = f"Default file: {default_raw}"
    else:
        st.warning("No file uploaded and no default CSV found.")
        st.stop()

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
# Derived Metrics
# ---------------------------
if clicks_col in df.columns and imps_col in df.columns:
    df["ctr_pct"] = (
        pd.to_numeric(df[clicks_col], errors="coerce")
        / pd.to_numeric(df[imps_col], errors="coerce")
    ) * 100

if cost_col in df.columns and clicks_col in df.columns:
    df["cpc"] = (
        pd.to_numeric(df[cost_col], errors="coerce")
        / pd.to_numeric(df[clicks_col], errors="coerce")
    )

if cost_col in df.columns and imps_col in df.columns:
    df["cpm"] = (
        pd.to_numeric(df[cost_col], errors="coerce")
        / (pd.to_numeric(df[imps_col], errors="coerce") / 1000)
    )

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
        min_d, max_d = df[date_col].min(), df[date_col].max()
        date_range = st.date_input("Date range", value=(min_d, max_d))
    else:
        date_range = None

df_filt = df.copy()
if channel_sel and channel_col in df_filt.columns:
    df_filt = df_filt[df_filt[channel_col].isin(channel_sel)]
if campaign_sel and campaign_col in df_filt.columns:
    df_filt = df_filt[df_filt[campaign_col].isin(campaign_sel)]
if date_range and date_col in df_filt.columns:
    df_filt = df_filt[
        (df_filt[date_col] >= pd.to_datetime(date_range[0]))
        & (df_filt[date_col] <= pd.to_datetime(date_range[1]))
    ]

# ---------------------------
# KPIs
# ---------------------------
st.subheader("Key Metrics")
def sum_safe(col): return pd.to_numeric(df_filt[col], errors="coerce").sum() if col in df_filt.columns else 0
def mean_safe(col): return pd.to_numeric(df_filt[col], errors="coerce").mean() if col in df_filt.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Cost", f"{sum_safe(cost_col):,.0f}")
c2.metric("Total Clicks", f"{sum_safe(clicks_col):,.0f}")
c3.metric("Total Impressions", f"{sum_safe(imps_col):,.0f}")
c4.metric("Avg CTR (%)", f"{mean_safe('ctr_pct'):.2f}")

c5, c6, c7 = st.columns(3)
c5.metric("Avg CPC", f"{mean_safe('cpc'):.2f}")
c6.metric("Avg CPM", f"{mean_safe('cpm'):.2f}")
c7.metric("Avg Conversion Rate (%)", f"{mean_safe(conv_rate_col):.2f}")

# ---------------------------
# Visualizations
# ---------------------------
st.subheader("Visualizations")

# CTR distribution
if "ctr_pct" in df_filt.columns:
    fig = px.histogram(df_filt, x="ctr_pct", nbins=30, title="CTR (%) Distribution")
    st.plotly_chart(fig, use_container_width=True)

# Cost by Category
cat_cols = [c for c in df_filt.columns if df_filt[c].dtype == "object"]
if cost_col in df_filt.columns and cat_cols:
    cat_col = st.selectbox("Category", options=cat_cols, index=0)
    data = df_filt.groupby(cat_col)[cost_col].mean().reset_index()
    fig = px.bar(data, x=cat_col, y=cost_col, title=f"Avg {cost_col} by {cat_col}")
    st.plotly_chart(fig, use_container_width=True)

# CTR by Category
if "ctr_pct" in df_filt.columns and cat_cols:
    cat_col = st.selectbox("Category for CTR", options=cat_cols, index=0, key="ctr_cat")
    data = df_filt.groupby(cat_col)["ctr_pct"].mean().reset_index()
    fig = px.bar(data, x=cat_col, y="ctr_pct", title=f"Avg CTR (%) by {cat_col}")
    st.plotly_chart(fig, use_container_width=True)

# Time Series
if date_col in df_filt.columns:
    metric_choice = st.selectbox("Metric for time series", options=["ctr_pct", cost_col, clicks_col, imps_col])
    ts = df_filt.groupby(date_col)[metric_choice].mean().reset_index()
    fig = px.line(ts, x=date_col, y=metric_choice, title=f"{metric_choice} over Time")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Download
# ---------------------------
st.download_button(
    "Download Filtered Data",
    df_filt.to_csv(index=False).encode("utf-8"),
    "filtered_data.csv",
    "text/csv",
)


