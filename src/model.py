import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from typing import Dict, Any
from src.utils import get_logger

logger = get_logger("ModelModule")

def train_model(X_train, y_train, params: Dict[str, Any] = None) -> xgb.XGBClassifier:
    """Trains an XGBoost model."""
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
    
    logger.info("Starting model training...")
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model

def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """Returns a dictionary of evaluation metrics."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs)
    }
    
    logger.info(f"Evaluation Metrics: {metrics}")
    return metrics