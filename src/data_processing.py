import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def load_data(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
        return None

def clean_data(df):
    """Handles missing values, duplicates, and dtypes."""
    if df is None: return None
    df = df.drop_duplicates()
    
    # Missing values: Numeric (median), Categorical (mode)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
            
    # Dtype fixes
    if 'signup_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    return df

def map_ip_to_country(fraud_df, ip_df):
    fraud_df = fraud_df.sort_values('ip_address')
    ip_df = ip_df.sort_values('lower_bound_ip_address')
    merged = pd.merge_asof(fraud_df, ip_df, left_on='ip_address', right_on='lower_bound_ip_address')
    merged['country'] = np.where(
        (merged['ip_address'] >= merged['lower_bound_ip_address']) & 
        (merged['ip_address'] <= merged['upper_bound_ip_address']), 
        merged['country'], 'Unknown'
    )
    return merged

def engineer_features(df):
    if 'signup_time' in df.columns:
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
    if 'device_id' in df.columns:
        df['device_usage_count'] = df.groupby('device_id')['device_id'].transform('count')
    return df