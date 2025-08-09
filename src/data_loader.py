
from pathlib import Path
import pandas as pd
from .logging_conf import logger

def load_csv(path: Path) -> pd.DataFrame:
    try:
        logger.info(f"Loading CSV: {path}")
        df = pd.read_csv(path)
        logger.info(f"Loaded shape: {df.shape}")
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found: {path}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"No data: {path}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error loading {path}: {e}")
        raise
