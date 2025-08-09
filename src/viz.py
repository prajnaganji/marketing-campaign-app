# src/viz.py
"""
Visualization utilities for the Marketing Campaign app.

All functions return Plotly Figure objects (or None if inputs are invalid).
Keep the plotting logic here so the Streamlit app stays thin and readable.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ------------ helpers ------------

def _ensure_columns(df: pd.DataFrame, cols: Sequence[str]) -> bool:
    """Return True if all columns exist, else False."""
    return all(c in df.columns for c in cols)


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Coerce to numeric and drop NaNs/inf so plots don't blow up."""
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return s.dropna()


# ------------ KPI helper ------------

def kpi_summary(
    df: pd.DataFrame,
    roi_col: str = "roi",
    revenue_col: str = "revenue",
    cost_col: str = "cost",
) -> dict:
    """
    Compute quick KPI numbers. Returns a dict you can feed into st.metric().
    Missing columns are handled gracefully.
    """
    out = {}
    if roi_col in df.columns:
        s = _clean_numeric(df[roi_col])
        if not s.empty:
            out["roi_avg"] = float(s.mean())
            out["roi_min"] = float(s.min())
            out["roi_max"] = float(s.max())

    if revenue_col in df.columns:
        out["revenue_sum"] = float(_clean_numeric(df[revenue_col]).sum())
    if cost_col in df.columns:
        out["cost_sum"] = float(_clean_numeric(df[cost_col]).sum())

    if "revenue_sum" in out and "cost_sum" in out and out["cost_sum"] != 0:
        out["roi_total"] = (out["revenue_sum"] - out["cost_sum"]) / out["cost_sum"] * 100.0

    return out


# ------------ charts ------------

def roi_histogram(
    df: pd.DataFrame,
    roi_col: str = "roi",
    nbins: Optional[int] = 30,
    title: str = "ROI Distribution",
) -> Optional[go.Figure]:
    """
    Histogram of ROI values (assumed to be percent numbers, e.g., 12.5 = 12.5%).
    """
    if not _ensure_columns(df, [roi_col]):
        return None

    s = _clean_numeric(df[roi_col])
    if s.empty:
        return None

    fig = px.histogram(df, x=roi_col, nbins=nbins, title=title)
    fig.update_layout(
        xaxis_title="ROI (%)",
        yaxis_title="Count",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    # Show reasonable x‑axis ticks (avoid 1 giant bar if outliers exist)
    q1, q99 = s.quantile([0.01, 0.99])
    if q1 < q99:
        fig.update_xaxes(range=[q1, q99])
    return fig


def roi_by_category_bar(
    df: pd.DataFrame,
    category_col: str,
    roi_col: str = "roi",
    agg: str = "mean",
    top_n: int = 20,
    title: Optional[str] = None,
) -> Optional[go.Figure]:
    """
    Bar chart of ROI aggregated by a category (e.g., channel, campaign, region).
    agg can be 'mean', 'median', or 'sum'. Only top_n categories are shown.
    """
    if not _ensure_columns(df, [category_col, roi_col]):
        return None

    tmp = df[[category_col, roi_col]].copy()
    tmp[roi_col] = _clean_numeric(tmp[roi_col])

    if tmp[roi_col].empty:
        return None

    if agg == "sum":
        g = tmp.groupby(category_col, dropna=False)[roi_col].sum()
    elif agg == "median":
        g = tmp.groupby(category_col, dropna=False)[roi_col].median()
    else:
        g = tmp.groupby(category_col, dropna=False)[roi_col].mean()

    ordered = g.sort_values(ascending=False).head(top_n).reset_index()
    plotted_title = title or f"ROI ({agg}) by {category_col}"

    fig = px.bar(
        ordered,
        x=category_col,
        y=roi_col,
        title=plotted_title,
        text=roi_col,
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(
        xaxis_title=category_col,
        yaxis_title="ROI (%)",
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        margin=dict(l=10, r=10, t=50, b=60),
    )
    return fig


def roi_vs_cost_scatter(
    df: pd.DataFrame,
    cost_col: str = "cost",
    roi_col: str = "roi",
    hover_cols: Optional[Iterable[str]] = None,
    title: str = "ROI vs Cost",
) -> Optional[go.Figure]:
    """
    Scatter plot of ROI (%) vs Cost. Helpful to see diminishing returns or outliers.
    """
    needed = [cost_col, roi_col]
    if not _ensure_columns(df, needed):
        return None

    tmp = df.copy()
    tmp[cost_col] = _clean_numeric(tmp[cost_col])
    tmp[roi_col] = _clean_numeric(tmp[roi_col])
    tmp = tmp.dropna(subset=[cost_col, roi_col])
    if tmp.empty:
        return None

    hover_data = list(hover_cols) if hover_cols else None

    fig = px.scatter(
        tmp,
        x=cost_col,
        y=roi_col,
        hover_data=hover_data,
        trendline="ols",
        title=title,
    )
    fig.update_layout(
        xaxis_title="Cost",
        yaxis_title="ROI (%)",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def time_series_metric(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: Optional[str] = None,
    agg: str = "sum",
    title: Optional[str] = None,
) -> Optional[go.Figure]:
    """
    Simple time‑series line chart. Optionally resample by freq (e.g., 'D','W','M').
    agg: 'sum' | 'mean' | 'median'
    """
    if not _ensure_columns(df, [date_col, value_col]):
        return None

    tmp = df[[date_col, value_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp[value_col] = _clean_numeric(tmp[value_col])
    tmp = tmp.dropna(subset=[date_col, value_col])

    if tmp.empty:
        return None

    if freq:
        if agg == "mean":
            tmp = tmp.set_index(date_col).resample(freq)[value_col].mean().reset_index()
        elif agg == "median":
            tmp = tmp.set_index(date_col).resample(freq)[value_col].median().reset_index()
        else:
            tmp = tmp.set_index(date_col).resample(freq)[value_col].sum().reset_index()

    plot_title = title or f"{value_col} over time"
    fig = px.line(tmp, x=date_col, y=value_col, title=plot_title)
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=value_col,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig

