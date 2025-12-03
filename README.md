# 🏋️ Gym Churn Prediction using Random Forest

## 🎯 Project Goal

The main objective of this project is to **predict customer churn** (attrition) for a U.S. fitness center chain using machine learning. By analyzing customer behavior and contract data, we aim to:
1. Identify the **key factors** that drive or prevent churn.
2. Develop a robust predictive model (Random Forest) to flag high-risk customers.
3. Provide **actionable insights** for the management team to design effective retention campaigns.

## 📊 Data Overview

The analysis uses the **`gym_churn_us.csv`** dataset, which contains 4,000 customer entries and 14 features. The data is clean with no missing values.

### Key Features

| Feature | Description | Type |
| :--- | :--- | :--- |
| $\text{Churn}$ | **Target Variable** (1: Churned, 0: Did not churn) | Binary |
| $\text{Lifetime}$ | Months since the customer first visited the gym. | Numerical |
| $\text{Avg\_class\_frequency\_current\_month}$ | Average weekly class attendance in the current month. | Numerical |
| $\text{Contract\_period}$ | Contract duration in months (1, 6, 12). | Ordinal |
| $\text{Age}$ | Age of the customer. | Numerical |
| $\text{Group\_visits}$ | Participation in group classes (1: Yes, 0: No). | Binary |

**Baseline Churn Rate:** Approximately **26.5%** of the customers in the dataset churned.

## 🛠️ Methodology and Modeling

### 1. Data Preprocessing
1.  **Feature Selection:** All available features were used to train the model.
2.  **Splitting:** The data was split into **Training** (75%) and **Testing** (25%) sets, utilizing **stratification** to ensure balanced class representation.
3.  **Scaling:** Numerical features were scaled using **StandardScaler** to normalize their distribution (though less critical for tree-based models like Random Forest, it's a good practice).

### 2. Random Forest Classifier

A **Random Forest Classifier** was chosen for its high accuracy, robustness to noisy data, and ability to provide explicit **Feature Importance** scores, which are vital for business interpretation.

## 🚀 Model Performance

The Random Forest model demonstrated strong predictive power on the unseen test data.

| Metric | Random Forest Score | Interpretation |
| :--- | :--- | :--- |
| **ROC AUC Score** | $\mathbf{0.9713}$ | Excellent ability to discriminate between churn and non-churn customers. |
| **Accuracy** | $0.9170$ | $91.7\%$ of all predictions were correct. |
| **Recall (Sensitivity)** | $0.7962$ | The model correctly identified $79.62\%$ of all customers who actually churned. |

### Confusion Matrix

| | Predicted: No Churn (0) | Predicted: Churn (1) |
| :--- | :--- | :--- |
| **Actual: No Churn (0)** | 706 (True Negatives) | 29 (False Positives) |
| **Actual: Churn (1)** | 54 (False Negatives) | **211 (True Positives)** |

## 🔍 Key Business Insights (Feature Importance)

The model's Feature Importance scores highlight the most critical variables for retention:

| Feature | Importance Score | Business Impact |
| :--- | :--- | :--- |
| **$\mathbf{Lifetime}$** | $\mathbf{0.2860}$ | **Most Critical Factor:** Predicts tenure. Customers with a short lifetime (new members) are at the highest risk. |
| **$\mathbf{Avg\_class\_frequency\_current\_month}$** | $\mathbf{0.1741}$ | **Behavioral Signal:** A sudden drop in class attendance in the current month is a strong, immediate precursor to churn. |
| **$\mathbf{Age}$** | $0.1226$ | Older members are more stable; younger customers are a higher churn risk segment. |
| **$\mathbf{Contract\_period}$** | $0.0603$ | Longer contracts (6 or 12 months) significantly reduce churn probability. |
| **$\mathbf{Group\_visits}$** | $0.0165$ | Group participation fosters community, acting as a small but beneficial protective factor. |

### 💡 Strategic Takeaways

Retention efforts should be prioritized for:
1.  **New Members:** Focus on engagement within the first 1-3 months ($\text{Lifetime}$).
2.  **Disengaged Members:** Immediately intervene when the $\text{Avg\_class\_frequency\_current\_month}$ drops below a healthy threshold.
3.  **Contract Structure:** Encourage customers to commit to longer-term contracts.

---

## 💻 How to Run the Analysis

### Requirements (Dependencies)

To execute the script, ensure you have the necessary libraries installed:

```bash
pip install pandas scikit-learn numpy
