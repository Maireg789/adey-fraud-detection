# 🛡️ Adey Fraud Detection System

![CI Status](https://github.com/Maireg789/adey-fraud-detection/actions/workflows/main.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Business Overview
A production-grade fraud detection pipeline designed for the finance sector. It identifies high-risk transactions with **XGBoost** and explains decisions using **SHAP**, enabling fraud analysts to make data-driven, audit-ready decisions.

## 🚀 Key Features
- **Real-Time Inference:** API-ready pipeline for scoring transactions.
- **Explainability:** Integrated SHAP waterfall plots for regulatory compliance.
- **Robust Engineering:** Automated testing (`pytest`) and CI/CD (GitHub Actions).
- **Interactive Dashboard:** Streamlit app for real-time investigation.

## 🛠️ Project Structure
```bash
├── .github/workflows  # CI/CD Pipeline
├── data/              # Raw and Processed Data
├── models/            # Serialized XGBoost Models
├── src/               # Source Code
│   ├── data_processing.py  # ETL & Feature Engineering
│   ├── model.py            # Training Logic
│   └── utils.py            # Logger Config
├── tests/             # Unit Tests
├── app.py             # Streamlit Dashboard
└── requirements.txt   # Dependencies
```
## 💻 Quick Start
1. Clone the repo:
code
Bash
git clone https://github.com/Maireg789/adey-fraud-detection.git
cd adey-fraud-detection
2. Install dependencies:
code
Bash
pip install -r requirements.txt
3. Run the Dashboard:
code
Bash
streamlit run app.py
## 📊 Results
Model: XGBoost Classifier
Metric: AUC-PR: 0.89
Key Insight: "Time between signup and purchase" is the #1 predictor of fraud.