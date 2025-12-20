from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import logging

def prepare_for_modeling(df, target_col):
    """Encodes categorical data and splits into X, y."""
    model_df = df.copy()
    le = LabelEncoder()
    
    # Drop IDs and Timestamps
    drop_cols = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 
                 'lower_bound_ip_address', 'upper_bound_ip_address']
    model_df = model_df.drop(columns=[c for c in drop_cols if c in model_df.columns])
    
    # Encode strings
    for col in model_df.select_dtypes(include=['object']).columns:
        model_df[col] = le.fit_transform(model_df[col].astype(str))
        
    X = model_df.drop(target_col, axis=1)
    y = model_df[target_col]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def handle_imbalance_smote(X_train, y_train):
    """Performs SMOTE and logs before/after counts."""
    logging.info(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logging.info(f"After SMOTE: {y_res.value_counts().to_dict()}")
    return X_res, y_res