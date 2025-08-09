
import streamlit as st
from pathlib import Path
import pandas as pd

from src.config import DATA_DIR, DEFAULT_INPUT, DEFAULT_CLEANED
from src.data_loader import load_csv
from src.cleaning import standardize_columns, fill_missing
from src.features import compute_roi
from src.viz import roi_histogram

st.set_page_config(page_title="Marketing Campaign Analysis", page_icon="📊", layout="wide")
st.title("📊 Marketing Campaign Analysis")

# Upload or use default
uploaded = st.file_uploader("Upload a CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    default_path = DATA_DIR / DEFAULT_CLEANED
    if default_path.exists():
        df = load_csv(default_path)
        st.info(f"Loaded default file: {default_path}")
    else:
        st.warning("No file uploaded and default file not found. Please upload a CSV.")
        st.stop()

st.subheader("Preview")
st.dataframe(df.head())

# Cleaning options
with st.expander("Cleaning"):
    df = standardize_columns(df)
    strategy = st.selectbox("Missing value strategy", ["median", "mean", "zero"], index=0)
    df = fill_missing(df, strategy="median" if strategy in ["median", "mean"] else "zero")

# Feature engineering
with st.expander("Features"):
    revenue_col = st.text_input("Revenue column", "revenue")
    cost_col = st.text_input("Cost column", "cost")
    df = compute_roi(df, revenue_col=revenue_col, cost_col=cost_col, out_col="roi")
    if "roi" in df.columns:
        st.success("ROI computed.")

# Visualization
st.subheader("Visualizations")
fig = roi_histogram(df, "roi")
if fig:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Add an ROI column to see the histogram.")

# Download
st.download_button("Download processed CSV", df.to_csv(index=False).encode("utf-8"), "processed.csv", "text/csv")
