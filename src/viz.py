
import pandas as pd
import plotly.express as px

def roi_histogram(df: pd.DataFrame, roi_col: str = "roi"):
    if roi_col not in df.columns:
        return None
    fig = px.histogram(df, x=roi_col, nbins=30, title="ROI Distribution")
    return fig
