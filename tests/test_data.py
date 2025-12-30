import pandas as pd
import pytest

def test_columns_present():
    df = pd.read_csv('data/processed/fraud_data_processed.csv')
    required_columns = ['purchase_value', 'class']
    for col in required_columns:
        assert col in df.columns, f"Missing column {col}"