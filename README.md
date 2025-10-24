# Customer Churn Prediction Pipeline

This repository contains an end-to-end machine learning pipeline for customer churn prediction, focusing on advanced temporal feature engineering from transaction logs.

## Project Overview

This pipeline transforms raw customer activity into a rich feature set to train a high-performance CatBoost model. The core strategy is capturing behavioral trends, activity decline, and inactivity patterns.

## Core Components

### **ChurnDataLoader**
Loads and prepares raw customer and transaction data

### **ChurnFeatureEngineer**
The core of the project. Creates 141 features from raw data, including:
- Temporal windows (Last 1, 3, 6 months)
- Trend analysis (`decline_acceleration`)
- Inactivity & Recency (RFM scores, `months_since_last`)
- Composite risk scores (`temporal_churn_risk`)

### **CatBoostChurnTrainer**
Trains a CatBoostClassifier using 5-fold StratifiedKFold cross-validation

## Performance & Validation

The model is validated using competition-specific metrics (Gini, Recall@k, Lift@k, AUC) to ensure robust performance. The final submission is an average of the 5-fold models.

- **Features Created**: 141
- **Training Customers**: 133,287
- **Test Customers**: 43,006
- **OOF Custom Metric**: 1.186404
## For a detailed breakdown of the analysis and feature engineering process, see the [Full Project Report](reports/Full_Report.md).

## To Run

1. Place datasets in the `/datasets/` folder
2. Execute the Jupyter Notebook cells in order
