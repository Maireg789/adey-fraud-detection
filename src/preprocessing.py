def scale_features(df, columns):
    """
    Scales specified numerical columns using StandardScaler.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        columns (list): List of column names to scale.
        
    Returns:
        pd.DataFrame: Dataframe with scaled features.
    """
    if not all(col in df.columns for col in columns):
        raise ValueError("One or more columns missing from dataframe.")
    # ... scaling logic ...