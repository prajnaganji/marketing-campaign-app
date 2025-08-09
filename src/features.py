
import pandas as pd
from .logging_conf import logger

def compute_roi(df: pd.DataFrame, revenue_col: str = "revenue", cost_col: str = "cost", out_col: str = "roi") -> pd.DataFrame:
    df = df.copy()
    if revenue_col not in df.columns or cost_col not in df.columns:
        logger.warning("Missing revenue or cost columns; skipping ROI computation")
        return df
    logger.info("Computing ROI")
    df[out_col] = ((df[revenue_col] - df[cost_col]) / df[cost_col]).replace([float('inf'), -float('inf')], None) * 100
    return df
