🛡️ Fraud Detection for E-commerce and Bank Transactions
Data Science Project for Adey Innovations Inc.
📋 Project Overview
This project aims to enhance the security of e-commerce and banking transactions by building sophisticated fraud detection models. By leveraging geolocation analysis, transaction patterns, and advanced machine learning (XGBoost), we identify fraudulent activities while maintaining a smooth user experience.
A core focus of this project is managing the trade-off between Security (catching fraud) and User Experience (minimizing false alarms).
📂 Repository Structure
Aligned with production-ready best practices:
code
Text
├── data/               # Raw and processed data (excluded from git)
├── models/             # Trained model artifacts (.pkl)
├── notebooks/          # EDA, Feature Engineering, Modeling, and SHAP
├── src/                # Modular source code for preprocessing
├── tests/              # Unit tests for data and model validation
├── reports/            # SHAP plots and built-in feature importance
├── scripts/            # Deployment and automation scripts
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
⚙️ Workflow & Tasks
Task 1: Data Analysis and Preprocessing
IP Mapping: Converted IP addresses to integer format and merged with IpAddress_to_Country.csv to identify geographic fraud patterns.
Feature Engineering: Created high-impact features such as time_since_signup (difference between signup and purchase) and transaction frequency.
Handling Imbalance: Applied SMOTE (Synthetic Minority Over-sampling Technique) to the training set to address the extreme class imbalance (0.17% fraud in banking data).
Task 2: Modeling & Statistical Rigor
We implemented a baseline Logistic Regression and a Tuned XGBoost Ensemble.
Hyperparameter Tuning: Used GridSearchCV to optimize max_depth, learning_rate, and scale_pos_weight.
5-Fold Cross-Validation: Ensured model stability across different data subsets.
Model Comparison Table
Model	Dataset	AUC-PR	F1-Score	Recall	Justification
Logistic Regression	E-commerce	0.52	0.45	0.51	Baseline: Low recall, misses many fraud cases.
Tuned XGBoost	E-commerce	0.81	0.78	0.74	Selected: Strongest balance of precision/recall.
Logistic Regression	Bank	0.69	0.62	0.58	Baseline: Struggles with extreme imbalance.
Tuned XGBoost	Bank	0.89	0.84	0.82	Selected: Robust performance on PCA features.
Note: Metrics prioritized AUC-PR and Recall due to the high business cost of False Negatives (missed fraud).
Task 3: Model Explainability (SHAP)
Using SHAP (SHapley Additive exPlanations), we deconstructed the "black-box" XGBoost model to understand fraud drivers:
Global Importance: time_since_signup was the #1 predictor—fraudsters often act immediately after account creation.
Local Importance: Force plots revealed that high purchase_value in specific countries significantly pushes the risk score higher.
💡 Business Recommendations
Based on SHAP and EDA results, we recommend the following for Adey Innovations:
Implement a "Cool-off" Period: Accounts making purchases within 10 minutes of signup should trigger a mandatory 24-hour hold or manual identity verification.
Adaptive MFA: Transactions exceeding $200 from countries identified as "High Risk" in our SHAP analysis should require Multi-Factor Authentication (OTP).
Velocity Thresholds: Implement automated blocks for any Device ID linked to more than 3 unique User IDs within a 24-hour window, as this pattern strongly correlates with fraudulent activity.
🛠️ Installation & Setup
Clone the Repo:
code
Bash
git clone https://github.com/Maireg789/adey-fraud-detection.git
Setup Environment:
code
Bash
pip install -r requirements.txt
Run Tests:
code
Bash
pytest tests/test_data.py
👨‍💻 Branching & PR Workflow
To maintain code quality, this project followed a feature-branch workflow:
Tasks were developed in feature/ branches.
Simulated Pull Requests were used to merge into the main branch.
Commit messages follow descriptive, professional standards.