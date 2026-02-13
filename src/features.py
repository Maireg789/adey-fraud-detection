import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Any

def preprocess_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, Any]:
    """
    Splits X/y and scales features.
    Returns: X_scaled, y, scaler_object
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify numeric columns for scaling
    numeric_cols = X.select_dtypes(include=['number']).columns
    
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # Convert back to DataFrame to keep column names (important for SHAP)
    X_scaled = pd.DataFrame(X, columns=X.columns)
    
    return X_scaled, y, scaler