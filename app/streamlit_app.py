# app/streamlit_app.py

# --- make imports work locally and on Streamlit Cloud ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd

from src.config import DATA_DIR, DEFAULT_INPUT, DEFAULT_CLEANED
from src.data_loader import load_csv
from src.cleaning import standardize_columns, fill_missing
from src.features import compute_roi
from src.viz import (
    roi_histogram,
    roi_by_category_bar,
    roi_vs_cost_scatter,
    time_series_metric,
    kpi_summary,
)

st.set_page_config(page_title="Marketing Campaign Analysis", page_icon="📊", layout="wide")
st.title("📊 Marketing Campaign Analysis")

# ---------------------------
# Sidebar controls
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
    st.header("ROI Settings")
    revenue_col_input = st.text_input("Revenue column", value="revenue")
    cost_col_input = st.text_input("Cost column", value="cost")

# ---------------------------
# Load data
# ---------------------------
if uploaded is not None:
    df = pd.read_csv(uploaded)
    source_msg = "Uploaded file"
else:
    # Try cleaned dataset first, then raw as a fallback
    default_clean = DATA_DIR / DEFAULT_CLEANED
    default_raw = DATA_DIR / DEFAULT_INPUT
    if default_clean.exists():
        df = load_csv(default_clean)
        source_msg = f"Default file: {default_clean}"
    elif default_raw.exists():
        df = load_csv(default_raw)
        source_msg = f"Default file: {default_raw}"
    else:
        st.warning(
            "No file uploaded and no default CSV found. "
            "Upload a CSV using the sidebar."
        )
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
    st.write("Column names will be standardized (lowercase, underscores).")
    df = standardize_columns(df)

    strategy_key = "median" if missing_strategy in ("median", "mean") else "zero"
    if missing_strategy == "mean":
        # fill_missing maps "mean" too, this just keeps the UI text tidy
        strategy_key = "mean"
    df = fill_missing(df, strategy=strategy_key)

# ---------------------------
# Feature engineering (ROI)
# ---------------------------
with st.expander("Features", expanded=False):
    st.write("Compute ROI = ((revenue - cost) / cost) × 100.")
    df = compute_roi(df, revenue_col=revenue_col_input, cost_col=cost_col_input, out_col="roi")
    if "roi" in df.columns:
        st.success("ROI computed and added as column 'roi'.")
    else:
        st.info("ROI not computed. Check the revenue and cost column names.")

# ---------------------------
# KPIs
# ---------------------------
st.subheader("Key Metrics")
kpis = kpi_summary(df, roi_col="roi", revenue_col=revenue_col_input, cost_col=cost_col_input)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg ROI (%)", f"{kpis.get('roi_avg', 0):.2f}")
c2.metric("Total Revenue", f"{kpis.get('revenue_sum', 0):,.0f}")
c3.metric("Total Cost", f"{kpis.get('cost_sum', 0):,.0f}")
c4.metric("Total ROI (%)", f"{kpis.get('roi_total', 0):.2f}")

# ---------------------------
# Visualizations
# ---------------------------
st.subheader("Visualizations")

# ROI histogram
fig = roi_histogram(df, roi_col="roi", nbins=40)
if fig:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Add an ROI column to see the ROI histogram.")

# ROI by category (choose any categorical column)
cat_cols = sorted(
    [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
)
if cat_cols:
    st.markdown("##### ROI by Category")
    ccol1, ccol2 = st.columns([2, 1])
    with ccol1:
        chosen_cat = st.selectbox("Category column", options=cat_cols, index=0, key="cat_col")
    with ccol2:
        agg = st.selectbox("Aggregation", options=["mean", "median", "sum"], index=0, key="agg")
    fig = roi_by_category_bar(df, category_col=chosen_cat, roi_col="roi", agg=agg, top_n=20)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

# ROI vs Cost scatter
num_cols = sorted([c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
if {"roi", cost_col_input}.issubset(df.columns):
    st.markdown("##### ROI vs Cost")
    fig = roi_vs_cost_scatter(
        df,
        cost_col=cost_col_input,
        roi_col="roi",
        hover_cols=[c for c in ["campaign", "channel"] if c in df.columns],
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True)

# Optional time series (only if a date column exists)
date_cols = [
    c for c in df.columns
    if pd.api.types.is_datetime64_any_dtype(df[c])
    or pd.api.types.is_object_dtype(df[c])  # allow parseable strings
]
if date_cols:
    st.markdown("##### Time Series")
    t1, t2, t3 = st.columns([2, 2, 1])
    with t1:
        date_col = st.selectbox("Date column", options=date_cols, index=0)
    with t2:
        value_col = st.selectbox(
            "Metric", options=[revenue_col_input, cost_col_input, "roi"] + [c for c in num_cols if c not in ["roi"]], index=0
        )
    with t3:
        freq = st.selectbox("Resample", options=[None, "D", "W", "M", "Q"], index=2)
    fig = time_series_metric(df, date_col=date_col, value_col=value_col, freq=freq, agg="sum")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Download
# ---------------------------
st.markdown("---")
st.download_button(
    "Download processed CSV",
    df.to_csv(index=False).encode("utf-8"),
    file_name="processed.csv",
    mime="text/csv",
)
