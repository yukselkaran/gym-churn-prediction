import streamlit as st
import pandas as pd
import pickle
import shap

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Gym Churn Prediction",
    layout="wide"
)

st.title("🏋️ Gym Churn Prediction App")

# -------------------------------------------------
# Load models & artifacts
# -------------------------------------------------
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)   # model (Random Forest)

with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# -------------------------------------------------
# Sidebar – User Input
# -------------------------------------------------
st.sidebar.header("Customer Information")

user_input = {}

# Binary features (0 / 1)
binary_features = [
    "gender",
    "Near_Location",
    "Partner",
    "Promo_friends",
    "Phone",
    "Group_visits"
]

for col in binary_features:
    user_input[col] = st.sidebar.selectbox(col, [0, 1])

# Numerical features
numerical_features = [
    "Contract_period",
    "Age",
    "Avg_additional_charges_total",
    "Month_to_end_contract",
    "Lifetime",
    "Avg_class_frequency_total",
    "Avg_class_frequency_current_month"
]

for col in numerical_features:
    user_input[col] = st.sidebar.number_input(col, value=0.0)

# Create DataFrame & ensure column order
input_df = pd.DataFrame([user_input])
input_df = input_df[feature_columns]

# -------------------------------------------------
# Model selection
# -------------------------------------------------
st.sidebar.header("Model Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "XGBoost"]
)

threshold = 0.5
if model_choice == "XGBoost":
    threshold = st.sidebar.slider(
        "Churn Threshold",
        0.1, 0.9, 0.35
    )

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if model_choice == "Random Forest":
    proba = rf_model.predict_proba(input_df)[0][1]
else:
    proba = xgb_model.predict_proba(input_df)[0][1]

prediction = int(proba >= threshold)
# -------------------------------------------------
# Text-based Explanation (Rule-based)
# -------------------------------------------------
reasons_positive = []  # churn riskini artıranlar
reasons_negative = []  # churn riskini azaltanlar

if input_df.loc[0, "Contract_period"] <= 3:
    reasons_positive.append("short contract period")

if input_df.loc[0, "Month_to_end_contract"] <= 1:
    reasons_positive.append("contract is about to end")

if input_df.loc[0, "Avg_class_frequency_current_month"] < 1:
    reasons_positive.append("low recent class attendance")

if input_df.loc[0, "Lifetime"] < 6:
    reasons_positive.append("short customer lifetime")

if input_df.loc[0, "Age"] < 30:
    reasons_positive.append("young age segment")

# Protective factors
if input_df.loc[0, "Contract_period"] >= 6:
    reasons_negative.append("long-term contract")

if input_df.loc[0, "Avg_class_frequency_current_month"] >= 3:
    reasons_negative.append("high class attendance")

if input_df.loc[0, "Partner"] == 1:
    reasons_negative.append("has a partner membership")

if input_df.loc[0, "Promo_friends"] == 1:
    reasons_negative.append("joined via friend promotion")

# -------------------------------------------------
# Results
# -------------------------------------------------
st.subheader("📊 Prediction Result")

col1, col2 = st.columns(2)

with col2:
    if prediction == 1:
        st.error("🚨 Customer WILL CHURN")

        if reasons_positive:
            st.markdown(
                "**Main reasons:** " +
                ", ".join(reasons_positive[:3])
            )
    else:
        st.success("✅ Customer WILL STAY")

        if reasons_negative:
            st.markdown(
                "**Decision based on:** " +
                ", ".join(reasons_negative[:3])
            )

# -------------------------------------------------
# SHAP Explanation (XGBoost only)
# -------------------------------------------------
show_shap = st.checkbox("🔍 Show explanation (SHAP)")

if show_shap and model_choice == "XGBoost":
    st.subheader("🔍 SHAP Explanation")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(input_df)

    # Binary classification güvenliği
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]

    shap_html = shap.force_plot(
        expected_value,
        shap_values[0],
        input_df.iloc[0]
    ).html()

    st.components.v1.html(shap_html, height=350)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Gym Churn Prediction • Random Forest & XGBoost • SHAP Explainability")
