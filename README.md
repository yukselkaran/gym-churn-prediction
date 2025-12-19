## 🏋️ Gym Customer Churn Prediction

## 🌟 Project Overview

This project focuses on predicting **customer churn (attrition)** for a gym chain using historical membership, usage, and spending data.  
A **Random Forest Classifier** is trained to identify members with a high risk of churn and to uncover the most influential factors behind customer attrition.

---

## 📊 Dataset Information

- **File:** `gym_churn_us.csv`
- **Total Records:** 4,000
- **Target Variable:** `Churn`
  - `1` → Churned
  - `0` → Retained
- **Churn Rate:** ~26.5%

---

## 🛠️ Methodology

1. **Data Loading & Cleaning**
   - Dataset loaded using Pandas
   - No missing values detected

2. **Exploratory Data Analysis (EDA)**
   - Descriptive statistics
   - Correlation analysis
   - Heatmap visualization for feature relationships

3. **Model Training**
   - Algorithm: **Random Forest Classifier**
   - Train-test split: **80% / 20%**
   - Default hyperparameters applied

---

## 🚀 Model Performance

The model demonstrates strong performance on the test dataset.

| Metric | Score |
|------|------|
| **Accuracy** | **0.92** |
| **Precision (Churn = 1)** | 0.87 |
| **Recall (Churn = 1)** | 0.80 |
| **F1-Score (Churn = 1)** | 0.83 |

---

## 🔍 Confusion Matrix

| | Predicted: No Churn (0) | Predicted: Churn (1) |
|---|---|---|
| **Actual: No Churn (0)** | 573 | 25 |
| **Actual: Churn (1)** | 40 | 162 |

---

## 🔑 Key Insights (Feature Importance)

The most important features influencing churn prediction are:

1. **Lifetime**  
   - Long-term members are significantly more stable.

2. **Avg_class_frequency_current_month**  
   - A decrease in recent attendance is a strong churn signal.

3. **Age**  
   - Age plays a meaningful role in customer retention behavior.

4. **Avg_class_frequency_total** and **Avg_additional_charges_total**  
   - Higher engagement and extra spending indicate stronger loyalty.

---

## 💻 Installation & Usage

### Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
