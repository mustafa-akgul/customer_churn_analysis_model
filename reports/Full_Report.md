# Customer Churn Prediction Pipeline

This repository contains an end-to-end machine learning pipeline developed for a datathon, which performs in-depth feature engineering on customer transaction logs to build a high-performance churn prediction model.

The project focuses on capturing temporal behavioral patterns and activity decline trends from customer transaction records.

## Project Components

The pipeline consists of three main sequential components:

### **ChurnDataLoader** (Data Loader)
- Loads raw CSV files (customer demographics and transaction history)
- Splits train and test sets
- Prepares data in a clean format for feature engineering

### **ChurnFeatureEngineer** (Feature Engineer)
- The core of the project
- Takes raw transaction data and creates a rich feature matrix for modeling
- Prevents data leakage using scikit-learn compatible `.fit_transform()` and `.transform()` methodology

### **CatBoostChurnTrainer** (Model Trainer)
- Uses the feature matrix to train CatBoostClassifier with StratifiedKFold (5-fold) cross-validation
- Evaluates performance using custom competition metrics (Gini, Recall@k, AUC)
- Generates predictions for the test set

## 1. Exploratory Data Analysis (EDA)

Pre-modeling analysis revealed significant behavioral differences between churned and non-churned customers:

### **Activity Decline**
Churned customers showed a significant drop in transaction counts and amounts in the 1-3 months before churning, compared to previous periods (e.g., 4-12 months prior).

### **Increased Inactivity**
Churned customers had statistically significant higher metrics for:
- "Number of months with zero transactions"
- "Time since last transaction (recency)"

### **Volatility Spike**
Some churned customers showed a sudden increase in transaction behavior "volatility" just before reducing their activity.

These findings formed the foundation of our feature engineering strategy.

## 2. Feature Engineering

This is the most critical part of the pipeline, deriving 141 high-signal features from raw transaction data.

### **Generated Feature Groups:**

#### **Aggregate Statistics**
Basic statistics across entire customer history (sum, mean, std, max) - e.g., `mobile_eft_all_cnt_sum`

#### **Temporal Window Features**
Analyzes customer behavior across different time periods:
- Last 1 month
- Last 3 months  
- Last 6 months
- Last 12 months
- Previous period (e.g., 4-12 months ago)

#### **Trend and Decline Features**
Compares different time windows to capture decline/acceleration trends - e.g., `decline_acceleration`, `mobile_decline_1to3`

#### **Inactivity and Recency Features**
Measures periods when customers were "inactive" - e.g., `zero_mobile_months`, `months_since_last`, `zero_ratio_last3m`

#### **RFM Scores**
Classical Recency, Frequency, Monetary metrics to measure customer value, including composite `rfm_total_score`

#### **Behavioral Risk Scores**
Composite scores combining multiple signals - e.g., `temporal_churn_risk`, `inactivity_risk_score`

#### **Demographic and Category Encoding**
Demographic data like `gender`, `province` processed with LabelEncoder

## 3. Modeling and Validation

### **Model Selection**
CatBoostClassifier was chosen for its:
- Native handling of categorical features
- High performance
- Speed

### **Validation Strategy**
Due to the inherently imbalanced nature of churn data, **StratifiedKFold (n_splits=5)** cross-validation was used to reliably measure model performance. This ensures the churn rate is preserved in train/validation sets across all folds.

### **Metrics**
Model performance was evaluated using a comprehensive metric set aligned with competition goals:

- **AUC (Area Under Curve)**: Measures model ranking capability
- **Gini Coefficient**: Normalized version of AUC via `(2 * AUC) - 1`
- **Recall@k (Top 10%)**: Measures how many churned customers we capture in the top 10% riskiest segment
- **Lift@k (Top 10%)**: Indicates how much better the model performs compared to random selection
- **Datathon Custom Metric**: Competition-specific scoring formula

## Results

The Out-of-Fold (OOF) metrics from 5-fold cross-validation demonstrate the model's stability and strength:

- **Total Training Data**: 133,287 Customers
- **Total Test Data**: 43,006 Customers  
- **Features Created**: 141
- **OOF Custom Metric (Average)**: 1.186404

The final `submission_catboost.csv` file was created by averaging predictions from all 5 fold models on the test data.

## How to Run

1. Install required libraries from `requirements.txt`
2. Place competition data (`customer_history.csv`, `customers.csv`, etc.) in the `/datasets/` folder
3. Execute the Jupyter Notebook cells sequentially:
   - **Step 1**: Data Loading (ChurnDataLoader)
   - **Step 2**: Feature Engineering (ChurnFeatureEngineer)  
   - **Steps 3-7**: Model Training & Submission (CatBoostChurnTrainer)

## Generated Outputs

- `submission_catboost.csv`: Prediction file for competition submission
- `catboost_feature_importance.csv`: List of the model's most important features
- `/models/`: 5 trained `.cbm` model files