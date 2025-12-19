# 🏋️ Gym Customer Churn Prediction

This project aims to predict **customer churn (membership cancellation)** for a gym chain using machine learning techniques.  
The goal is to identify high-risk members early and support **data-driven retention strategies**.

---

## 🚀 Project Overview

- **Problem Type:** Binary Classification (Churn / No Churn)
- **Dataset Size:** 4,000 customers
- **Target Variable:** `Churn`  
  - `1` → Customer churned  
  - `0` → Customer retained
- **Churn Rate:** ~26.5%

---

## 📊 Dataset Information

**File:** `gym_churn_us.csv`

### Features:
- Demographic data (Age, Gender)
- Contract information (Contract period, Remaining months)
- Usage behavior (Class frequency, Group visits)
- Engagement indicators (Lifetime, Additional charges)

All features are **numerical**, and the dataset contains **no missing values**.

---

## 🛠️ Methodology

### 1️⃣ Exploratory Data Analysis (EDA)
- Descriptive statistics
- Churn distribution analysis
- **Correlation heatmap** to explore feature relationships

---

### 2️⃣ Model Training

#### ✅ Random Forest Classifier
- Baseline ensemble model
- Strong overall performance
- Feature importance extracted

#### ✅ XGBoost Classifier
- Gradient boosting model
- Handles class imbalance effectively
- Achieved superior performance compared to Random Forest

---

### 3️⃣ Threshold Optimization (Critical for Churn Problems)

Instead of using the default `0.5` threshold:
- Optimized decision threshold to **maximize recall**
- Goal: catch as many churn customers as possible
- Final threshold used: **0.35**

---

### 4️⃣ Model Explainability (SHAP)
- SHAP values used for **global and local interpretability**
- Identified key churn drivers:
  - Membership lifetime
  - Recent attendance frequency
  - Contract duration
  - Age

---

## 📈 Model Performance Comparison

| Model | Recall (Churn=1) | Precision (Churn=1) | F1-score | ROC-AUC |
|------|------------------|---------------------|----------|---------|
| Random Forest | 0.80 | 0.87 | 0.83 | 0.96 |
| XGBoost (Default) | 0.86 | 0.86 | 0.86 | 0.97 |
| **XGBoost (Threshold Optimized)** | **0.90** | 0.84 | **0.87** | **0.97** |

📌 **Threshold-optimized XGBoost** was selected as the final model.

---

## 🧠 Key Insights

- Long-term members are significantly less likely to churn
- A sudden drop in recent gym attendance is the strongest churn signal
- Short contract duration strongly correlates with churn risk
- Threshold optimization dramatically improves business impact

---

## 💻 Deployment (Streamlit App)

The final model is deployed using **Streamlit**, allowing:
- Manual customer input
- Real-time churn probability prediction
- Risk classification using optimized threshold

### Run the app:
```bash
streamlit run app.py
