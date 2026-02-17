import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Adey Fraud Detection", layout="wide")

@st.cache_resource
def load_assets():
    try:
        model = joblib.load("models/xgb_model.pkl")
        features = joblib.load("models/features.pkl")
        # Load the FULL dataset
        data = pd.read_csv("models/all_processed_data.csv")
        return model, features, data
    except FileNotFoundError:
        return None, None, None

model, features, data = load_assets()

if model is None:
    st.error("🚨 Data not found. Run 'python train_pipeline.py' first!")
else:
    st.title("🛡️ Adey Fraud Detection System")
    st.markdown(f"### Analyzing {len(data):,} Real Transactions")

    # --- Sidebar: Select ANY Transaction ---
    st.sidebar.header("Select Transaction")
    
    # Input box for ID (0 to 150,000)
    max_id = len(data) - 1
    txn_id = st.sidebar.number_input(f"Enter Transaction ID (0 - {max_id})", min_value=0, max_value=max_id, value=123, step=1)
    
    # Extract the row
    row_data = data.iloc[[txn_id]].copy()
    
    # Separate Features (X) and Target (y) if 'class' exists
    if 'class' in row_data.columns:
        actual_class = row_data['class'].values[0]
        # Drop class so we can feed it to the model
        row_features = row_data.drop(columns=['class'])
    else:
        actual_class = None
        row_features = row_data

    # --- Display Data ---
    st.write("#### 1. Transaction Features")
    st.dataframe(row_features)
    
    if actual_class is not None:
        status = "FRAUD" if actual_class == 1 else "LEGIT"
        color = "red" if actual_class == 1 else "green"
        st.caption(f"Actual Historical Status: :{color}[{status}]")

    if st.button("Analyze Risk"):
        # Predict
        prob = model.predict_proba(row_features)[0][1]
        
        # Display Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Fraud Probability", f"{prob:.2%}")
        
        risk_label = "HIGH RISK" if prob > 0.5 else "Low Risk"
        col2.metric("Model Prediction", risk_label, delta="-Block" if prob > 0.5 else "+Approve")
        
        if actual_class is not None:
            match = (int(prob > 0.5) == actual_class)
            col3.metric("Prediction vs Actual", "Correct" if match else "Incorrect", delta_color="normal")
        
        # --- FIXED SHAP PLOT ---
        st.write("#### 2. Why this prediction?")
        with st.spinner("Calculating SHAP Waterfall..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(row_features)
            
            # Create a Matplotlib Figure explicitly
            fig = plt.figure(figsize=(10, 5))
            
            # Draw the Waterfall plot
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            
            # Display in Streamlit
            st.pyplot(fig, clear_figure=True)
            
            st.info("Red bars push the risk score UP. Blue bars push it DOWN.")