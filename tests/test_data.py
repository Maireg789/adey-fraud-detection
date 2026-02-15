import pytest
import pandas as pd
import numpy as np
from src.data_processing import get_country_fast

def test_get_country_fast_found():
    """Test that the IP mapping logic works for a known range."""
    # Create a dummy IP dataframe
    data = {
        'lower_bound_ip_address': [100, 200],
        'upper_bound_ip_address': [150, 250],
        'country': ['TestLand', 'DebugRepublic']
    }
    ip_df = pd.DataFrame(data)
    
    ip_lowers = ip_df['lower_bound_ip_address'].tolist()
    ip_countries = ip_df['country'].tolist()
    
    # Test an IP inside the range [100, 150]
    result = get_country_fast(120, ip_df, ip_lowers, ip_countries)
    assert result == 'TestLand'

def test_get_country_fast_not_found():
    """Test that the IP mapping handles unknown IPs gracefully."""
    data = {
        'lower_bound_ip_address': [100],
        'upper_bound_ip_address': [150],
        'country': ['TestLand']
    }
    ip_df = pd.DataFrame(data)
    ip_lowers = ip_df['lower_bound_ip_address'].tolist()
    ip_countries = ip_df['country'].tolist()
    
    # Test an IP outside the range
    result = get_country_fast(999, ip_df, ip_lowers, ip_countries)
    assert result == 'Unknown'

def test_dataframe_structure():
    """Simple smoke test to ensure libraries are installed."""
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    assert not df.empty
    assert df.shape == (2, 2)