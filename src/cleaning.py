
import pandas as pd
from .logging_conf import logger

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Standardizing column names")
    df = df.copy()
    df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]
    return df

def fill_missing(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    logger.info(f"Filling missing values using strategy={strategy}")
    df = df.copy()
    numeric = df.select_dtypes(include="number").columns
    if strategy == "median":
        df[numeric] = df[numeric].fillna(df[numeric].median())
    elif strategy == "mean":
        df[numeric] = df[numeric].fillna(df[numeric].mean())
    else:
        df[numeric] = df[numeric].fillna(0)
    return df
