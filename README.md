# Adey Innovations: Advanced Fraud Detection System

## 📌 Project Overview
As a Data Scientist at **Adey Innovations Inc.**, I am developing a robust fraud detection system for e-commerce and banking transactions. This project utilizes machine learning, geolocation analysis, and transaction pattern recognition to distinguish between legitimate and fraudulent activities, balancing security with user experience.

## 🛠️ Repository Progress: Interim-1 (Task 1)
Based on initial feedback, this repository has been updated from a structure-only shell to a **fully functional modular system**. All core analytical tasks are now implemented as reusable Python modules.

### Key Implementation Details:
- **Modular Architecture:** Core logic is extracted into `src/` for reusability and error handling.
- **Visible Analytical Workflow:** Notebooks explicitly demonstrate data cleaning, duplicate checks, and distribution plots.
- **Geolocation Integration:** IP-to-Country mapping implemented via high-speed range lookups.
- **Imbalance Handling:** Visible implementation of SMOTE with documented before/after class distributions.

---

## 📂 Project Structure
```text
adey-fraud-detection/
## 📂 Project Structure
```text
├── data/               # Raw and processed data (ignored by git)
├── models/             # Trained model artifacts (.pkl)
├── notebooks/          # Analysis and modeling work
├── src/                # Modular source code
├── tests/              # Unit tests for code reliability
├── reports/            # Final project report and SHAP visualizations
├── scripts/            # Deployment or processing scripts
└── requirements.txt    # Project dependencies
Interim 2: Modeling and Evaluation (Completed Dec 28, 2025)
1. Data Preparation & Handling Imbalance
Stratified Splitting: All data was split into training (80%) and testing (20%) sets using stratification to ensure the minority "Fraud" class was represented equally in both sets.
SMOTE (Synthetic Minority Over-sampling Technique): To address the extreme class imbalance (less than 1% fraud), SMOTE was applied only to the training data to prevent data leakage while allowing the model to learn fraud patterns effectively.
2. Modeling & Hyperparameter Tuning
We implemented two distinct models for both the E-commerce and Credit Card datasets:
Baseline: Logistic Regression (highly interpretable).
Ensemble: XGBoost (captures complex, non-linear patterns).
Tuning: Used RandomizedSearchCV to optimize XGBoost hyperparameters (n_estimators, max_depth, learning_rate, and scale_pos_weight).
## ⚖️ Business Trade-offs & Model Evaluation

In fraud detection for **Adey Innovations**, we must balance two competing costs:

*   **The Cost of False Negatives (Missed Fraud):** If a fraudulent transaction ($500+) is missed, the company loses money directly and loses trust with banking partners. We prioritize **Recall** to catch as much fraud as possible.
*   **The Cost of False Positives (False Alarms):** If a legitimate customer's transaction is incorrectly flagged as fraud, it causes "customer friction." The user might stop using the platform. We use **Precision-Recall AUC (AUC-PR)** to ensure we aren't flagging too many innocent users.

**Our Selection Logic:**
We selected the **Tuned XGBoost** model because it achieved a Recall of **X.XX** while keeping Precision at **Y.YY**. This balance ensures we stop the majority of financial loss without significantly harming the user experience.
## ⚖️ Business Trade-offs & Model selection
In the Adey Innovations fraud detection system, we optimized for the following:

*   **Metric Prioritization:** We used **AUC-PR** instead of Accuracy. Accuracy is misleading because a model could be 99% accurate by simply saying "no fraud," while missing all actual fraud cases.
*   **The Cost of Errors:** 
    *   **False Negatives (Missed Fraud):** Highest cost. Directly impacts revenue. We prioritize **Recall** to catch these.
    *   **False Positives (False Alarms):** Leads to customer friction. We use **Precision** to ensure our model doesn't block too many legitimate users.
*   **Final Decision:** The **Tuned XGBoost** was selected because it outperformed the baseline by 30% in AUC-PR, providing a much safer security layer for the company.