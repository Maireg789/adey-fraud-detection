import pandas as pd
import numpy as np
import io
import bisect
from src.utils import get_logger

logger = get_logger("DataProcessing")

def load_ip_data(filepath: str) -> pd.DataFrame:
    """Loads and prepares IP Country data."""
    logger.info(f"Loading IP data from {filepath}...")
    ip_df = pd.read_csv(filepath)
    # Ensure lower and upper bounds are integers
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(int)
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(int)
    return ip_df

def get_country_fast(ip_address: float, ip_df: pd.DataFrame, ip_lowers: list, ip_countries: list) -> str:
    """Fast look-up of country using binary search (bisect)."""
    # IP is float in the main dataset, convert to int
    ip = int(ip_address)
    
    # Find insertion point
    idx = bisect.bisect_right(ip_lowers, ip) - 1
    
    if idx >= 0:
        # Check if it's within the upper bound
        row = ip_df.iloc[idx]
        if row['lower_bound_ip_address'] <= ip <= row['upper_bound_ip_address']:
            return ip_countries[idx]
    return "Unknown"

def process_fraud_data(fraud_path: str, ip_path: str) -> pd.DataFrame:
    """
    Loads Fraud_Data and IP_Address_to_Country, merges them, and processes features.
    """
    # 1. Load Data
    logger.info("Loading datasets...")
    df = pd.read_csv(fraud_path)
    ip_df = load_ip_data(ip_path)
    
    # 2. Map IP to Country (Optimized)
    logger.info("Mapping IP addresses to countries (this may take a moment)...")
    
    # Pre-compute lists for binary search
    ip_lowers = ip_df['lower_bound_ip_address'].tolist()
    ip_countries = ip_df['country'].tolist()
    
    # Apply the mapping (Use a subset for testing if too slow, e.g., df.head(10000))
    # For full run:
    df['country'] = df['ip_address'].apply(lambda x: get_country_fast(x, ip_df, ip_lowers, ip_countries))
    
    # 3. Time Features
    logger.info("Engineering time features...")
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    # Velocity: Seconds between signup and purchase
    df['time_diff'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # 4. Cleanup
    cols_to_drop = ['user_id', 'device_id', 'signup_time', 'purchase_time', 'ip_address']
    df = df.drop(columns=cols_to_drop)
    
    # 5. One-Hot Encoding (Country, Source, Browser, Sex)
    logger.info("Encoding categorical variables...")
    
    # Limit countries to Top 10 to prevent 200+ columns (Engineering Decision)
    top_countries = df['country'].value_counts().nlargest(10).index
    df['country'] = df['country'].apply(lambda x: x if x in top_countries else 'Other')
    
    categorical_cols = ['source', 'browser', 'sex', 'country']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    logger.info(f"Processing complete. Shape: {df.shape}")
    return df