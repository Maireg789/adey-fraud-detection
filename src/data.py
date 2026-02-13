import pandas as pd
from typing import Tuple
from src.utils import get_logger

logger = get_logger("DataModule")

def load_data(filepath: str) -> pd.DataFrame:
    """Loads CSV data from a path."""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic cleaning: drops duplicates, handles missing values.
    Refactor this based on your specific notebook logic.
    """
    initial_rows = len(df)
    df = df.drop_duplicates()
    
    # Example: Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    logger.info(f"Cleaned data. Dropped {initial_rows - len(df)} duplicates.")
    return df