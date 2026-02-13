import pytest
import pandas as pd
import numpy as np
from src.data import clean_data

def test_clean_data_removes_duplicates():
    # Create dummy data with duplicate
    data = {
        'amount': [100.0, 200.0, 100.0],
        'fraud': [0, 1, 0]
    }
    df = pd.DataFrame(data)
    
    cleaned_df = clean_data(df)
    
    # Should have 2 rows (duplicate removed)
    assert len(cleaned_df) == 2
    assert cleaned_df.shape[0] == 2

def test_clean_data_fills_missing():
    # Create dummy data with NaN
    data = {
        'amount': [100.0, np.nan, 300.0],
        'fraud': [0, 0, 1]
    }
    df = pd.DataFrame(data)
    
    cleaned_df = clean_data(df)
    
    # Should not have any nulls
    assert cleaned_df['amount'].isnull().sum() == 0
    # Median of 100 and 300 is 200
    assert cleaned_df.iloc[1]['amount'] == 200.0
        