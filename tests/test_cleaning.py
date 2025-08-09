
import pandas as pd
from src.cleaning import standardize_columns, fill_missing

def test_standardize_columns():
    df = pd.DataFrame({'A Col ': [1,2], 'B Col': [3,4]})
    out = standardize_columns(df)
    assert list(out.columns) == ['a_col', 'b_col']

def test_fill_missing_median():
    df = pd.DataFrame({'x': [1, None, 3]})
    out = fill_missing(df, strategy="median")
    assert out['x'].isna().sum() == 0
