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
├── data/
│   ├── raw/                # Original Fraud_Data, Creditcard, and IP datasets
│   └── processed/          # Data post-cleaning and feature engineering
├── notebooks/
│   ├── eda-fraud-data.ipynb     # Visible cleaning, EDA, and class analysis
│   ├── eda-creditcard.ipynb     # Bank transaction analysis
│   └── feature-engineering.ipynb # Modular feature creation & SMOTE
├── src/                    # Modular Source Files (Reusable Logic)
│   ├── data_processing.py  # Cleaning, Mapping, and Feature Engineering
│   ├── eda_utils.py        # Reusable visualization functions
│   └── model_utils.py      # Encoding and Imbalance handling (SMOTE)
├── requirements.txt        # Pinned dependencies for environment stability
├── Interim_Report_1.md     # Detailed summary of Task 1 findings
└── README.md