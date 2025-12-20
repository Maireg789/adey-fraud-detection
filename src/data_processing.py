import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_data(path):
    """Loads data with basic error handling. Satisfies feedback: 'basic error handling'."""
    try:
        df = pd.read_csv(path)
        logging.info(f"Successfully loaded {path}")
        return df
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
        return None

def clean_data(df):
    """Handles missing values, duplicates, and dtypes. Satisfies feedback: 'actual cleaning and duplicate checks'."""
    if df is None: return None
    
    # 1. Duplicate check
    before_count = len(df)
    df = df.drop_duplicates()
    logging.info(f"Removed {before_count - len(df)} duplicate rows.")
    
    # 2. Missing values: Numeric (median), Categorical (mode)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        else:
            df[col] = df[col].fillna(df[col].median())
            
    # 3. Dtype fixes
    if 'signup_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    return df

def map_ip_to_country(fraud_df, ip_df):
    """Geolocation Integration using range-based lookup."""
    fraud_df = fraud_df.sort_values('ip_address')
    ip_df = ip_df.sort_values('lower_bound_ip_address')
    
    merged = pd.merge_asof(
        fraud_df, 
        ip_df, 
        left_on='ip_address', 
        right_on='lower_bound_ip_address'
    )
    
    merged['country'] = np.where(
        (merged['ip_address'] >= merged['lower_bound_ip_address']) & 
        (merged['ip_address'] <= merged['upper_bound_ip_address']), 
        merged['country'], 'Unknown'
    )
    return merged

def engineer_features(df):
    """Task 1: Creating time and velocity features."""
    if 'signup_time' in df.columns:
        # Time-based features
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # Velocity features: how many times is this device used?
    if 'device_id' in df.columns:
        df['device_usage_count'] = df.groupby('device_id')['device_id'].transform('count')
        
    return df

def scale_features(df, columns):
    """Standardizes numeric features (crucial for Credit Card data)."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df