import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.utils import get_logger

logger = get_logger("Explainability")

def setup_shap(model, X_train):
    """Initializes the SHAP explainer."""
    logger.info("Initializing SHAP explainer...")
    # For XGBoost, TreeExplainer is best
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    return explainer, shap_values

def plot_shap_summary(explainer, shap_values, X_train):
    """Generates the global summary plot."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, show=False)
    return plt.gcf()

def plot_force_plot(explainer, shap_values, X_input, index=0):
    """
    Generates a force plot for a specific observation.
    Note: Streamlit uses shap.plots.waterfall or force mostly.
    """
    # For a single prediction
    explanation = shap.Explanation(
        values=shap_values[index], 
        base_values=explainer.expected_value, 
        data=X_input.iloc[index], 
        feature_names=X_input.columns
    )
    return explanation