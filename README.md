# 🏋️ Gym Customer Churn Prediction

## 🌟 Project Summary

This project aims to predict **customer churn** (attrition) for a gym chain using historical usage and contract data. We used a **Random Forest Classifier** to identify high-risk members and determine the most influential factors driving churn.

## 📊 Data Overview

- **File:** `gym_churn_us.csv`
- **Total Records:** 4,000
- **Target Variable:** `Churn` (1: Churned, 0: Did not churn)
- **Churn Rate:** ~26.5%

## 🛠️ Methodology

1.  **Data Loading & Cleaning:** Loaded the dataset; confirmed no missing values.
2.  **Exploratory Data Analysis (EDA):** Analyzed descriptive statistics and feature correlations (Heatmap generated).
3.  **Model:** A **Random Forest Classifier** was trained on an 80% split of the data.

## 🚀 Model Performance

The model achieved strong performance on the test set (20% of data).

| Metric | Score |
| :--- | :--- |
| **Accuracy** | $\mathbf{0.92}$ |
| **Precision (Class 1)** | $0.87$ |
| **Recall (Class 1)** | $0.80$ |
| **F1-Score (Class 1)** | $0.83$ |

### Confusion Matrix

| | Predicted: No Churn (0) | Predicted: Churn (1) |
| :--- | :--- | :--- |
| **Actual: No Churn (0)** | 573 | 25 |
| **Actual: Churn (1)** | 40 | **162** |

## 🔑 Main Insights (Feature Importance)

Based on the feature importance plot, the most critical factors influencing churn prediction are:

1.  **$\mathbf{Lifetime}$ (Membership Duration):** The longest-standing members are the most stable.
2.  **$\mathbf{Avg\_class\_frequency\_current\_month}$ (Recent Attendance):** A drop in current month usage is a strong, immediate indicator of impending churn.
3.  **$\mathbf{Age}$:** Age is a significant predictor.
4.  **$\mathbf{Avg\_class\_frequency\_total}$ (Total Attendance) and $\mathbf{Avg\_additional\_charges\_total}$ (Extra Spending):** These factors also play a major role in customer stability.

## 💻 Setup and Execution

### Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
