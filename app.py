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
        # Load the REAL sample data we saved during training
        sample_data = pd.read_csv("models/sample_data.csv")
        return model, features, sample_data
    except FileNotFoundError:
        return None, None, None

model, features, sample_data = load_assets()

if model is None:
    st.error("🚨 Models not found. Please run 'python train_pipeline.py' first!")
else:
    st.title("🛡️ Adey Fraud Detection System")
    st.markdown("### Real-Time Transaction Analysis")

    # Sidebar
    st.sidebar.header("Select Transaction")
    
    # Allow user to pick a row from the real data
    # We use the index as a proxy for Transaction ID
    txn_id = st.sidebar.selectbox("Choose a Transaction ID (Sample)", sample_data.index)
    
    # Get the row data
    row_data = sample_data.loc[[txn_id]]
    
    st.write("#### 1. Transaction Features (Processed)")
    st.dataframe(row_data)

    if st.button("Analyze Risk"):
        # Predict
        prob = model.predict_proba(row_data)[0][1]
        prediction = int(prob > 0.5)
        
        # Display
        col1, col2, col3 = st.columns(3)
        col1.metric("Fraud Probability", f"{prob:.2%}")
        col2.metric("Risk Status", "HIGH RISK" if prob > 0.5 else "Safe", 
                    delta="-Block" if prob > 0.5 else "+Approve")
        
        # Explainability
        st.write("#### 2. Why this prediction?")
        with st.spinner("Calculating SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(row_data)
            
            fig, ax = plt.subplots(figsize=(10, 3))
            shap.plots.force(
                shap_values[0].base_values, 
                shap_values[0].values, 
                row_data.iloc[0], 
                matplotlib=True,
                show=False
            )
            st.pyplot(fig)
            
            st.info("Red bars push risk HIGHER. Blue bars push risk LOWER.")