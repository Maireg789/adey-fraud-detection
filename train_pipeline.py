import pandas as pd
import joblib
from src.data_processing import process_fraud_data
from src.features import preprocess_features
from src.model import train_model

def run_training():
    # Define paths
    fraud_path = "data/raw/Fraud_Data.csv"
    ip_path = "data/raw/IpAddress_to_Country.csv"
    
    # 1. Process Data
    print("🚀 Starting Data Processing...")
    try:
        # We assume target column is 'class' based on typical 10 Academy dataset
        df = process_fraud_data(fraud_path, ip_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # Check for correct target column name
    if 'class' in df.columns:
        target = 'class'
    elif 'Class' in df.columns:
        target = 'Class'
    else:
        print("❌ Error: Target column 'class' not found.")
        return

    # 2. Split & Scale
    print("⚖️ Scaling features...")
    X_scaled, y, scaler = preprocess_features(df, target_col=target)
    
    # 3. Train
    print("🤖 Training Model...")
    model = train_model(X_scaled, y)
    
    # 4. Save Artifacts
    print("💾 Saving artifacts to 'models/'...")
    joblib.dump(model, "models/xgb_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(X_scaled.columns.tolist(), "models/features.pkl")
    
    # --- CRITICAL: Save a sample of REAL data for the Dashboard ---
    # We save 50 rows of processed X data (with no target)
    sample_data = X_scaled.sample(50)
    sample_data.to_csv("models/sample_data.csv", index=False)
    print("✅ Saved 'models/sample_data.csv' for the dashboard.")

    print("🎉 Pipeline finished successfully!")

if __name__ == "__main__":
    run_training()