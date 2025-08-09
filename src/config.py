
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DEFAULT_INPUT = os.getenv("DEFAULT_INPUT", "Advertisment-checkpoint.csv")
DEFAULT_CLEANED = os.getenv("DEFAULT_CLEANED", "cleaned_marketing_campaign_dataset.csv")
