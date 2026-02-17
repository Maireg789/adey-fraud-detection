import pandas as pd
import joblib
from src.data_processing import process_fraud_data
from src.features import preprocess_features
from src.model import train_model

def run_training():
    fraud_path = "data/raw/Fraud_Data.csv"
    ip_path = "data/raw/IpAddress_to_Country.csv"
    
    print("🚀 Loading and Processing ALL Data (This may take a minute)...")
    try:
        df = process_fraud_data(fraud_path, ip_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # Handle target column
    if 'class' in df.columns:
        target = 'class'
    elif 'Class' in df.columns:
        target = 'Class'
    else:
        print("❌ Error: Target column 'class' not found.")
        return

    print("⚖️ Scaling features for the entire dataset...")
    X_scaled, y, scaler = preprocess_features(df, target_col=target)
    
    print("🤖 Training Model...")
    model = train_model(X_scaled, y)
    
    print("💾 Saving artifacts...")
    joblib.dump(model, "models/xgb_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(X_scaled.columns.tolist(), "models/features.pkl")
    
    # --- CHANGE: Save the FULL processed dataset ---
    # We add the target 'class' back temporarily so we can see the ground truth in the dashboard
    X_scaled['class'] = y.values
    
    # Save to a new file (warning: this will be ~20-30MB)
    print("💾 Saving FULL processed dataset to 'models/all_processed_data.csv'...")
    X_scaled.to_csv("models/all_processed_data.csv", index=False)
    
    print("🎉 Pipeline finished! Full dataset ready for Dashboard.")

if __name__ == "__main__":
    run_training()