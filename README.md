# UPI Customer Churn Prediction

Binary classification project to predict customer churn on a UPI payment platform. Includes EDA, KNN imputation, feature engineering, and benchmarking of 7 ML models with K-Fold CV and ROC-AUC evaluation.

## Overview

Customer acquisition is costlier than retention. This project builds a machine learning pipeline to identify UPI users likely to churn, enabling businesses to take proactive action.

**Target Variable:** `is_churned` (1 = churned, 0 = retained)  
**Dataset Size:** ~104,000 users | 17 features

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Jupyter Notebook


## Workflow

1. **EDA** — distributions, correlations, class balance
2. **Data Cleaning** — type corrections, outlier treatment
3. **Missing Value Treatment** — mode/median fill, KNN imputation for city
4. **Feature Engineering** — city-tier mapping, device grouping
5. **Encoding & Scaling** — dummy encoding, StandardScaler
6. **Modeling** — Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, AdaBoost, XGBoost
7. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, K-Fold Cross Validation

## Models Benchmarked

| Model | Evaluation |
|---|---|
| Logistic Regression | ROC-AUC, K-Fold CV |
| Decision Tree | ROC-AUC, K-Fold CV |
| Random Forest | ROC-AUC, K-Fold CV |
| K-Nearest Neighbors | ROC-AUC, K-Fold CV |
| Naive Bayes | ROC-AUC, K-Fold CV |
| AdaBoost | ROC-AUC, K-Fold CV |
| XGBoost | ROC-AUC, K-Fold CV |

## Future Scope

- Handle class imbalance with SMOTE
- Hyperparameter tuning for XGBoost
- Deploy as an interactive Streamlit app for real-time churn probability prediction
