import pandas as pd
import numpy as np

def test_environment_setup():
    """Smoke test to verify libraries are installed correctly."""
    assert pd.__version__ is not None
    assert np.__version__ is not None

def test_dataframe_creation():
    """Test that pandas can create dataframes (System Health Check)."""
    data = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data)
    assert df.shape == (2, 2)
    assert not df.empty

def test_basic_math():
    """Basic logic test."""
    assert 1 + 1 == 2