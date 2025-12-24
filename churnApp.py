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
# Session state (UX control)
# -------------------------------------------------
if "has_input" not in st.session_state:
    st.session_state.has_input = False

# -------------------------------------------------
# Load models & artifacts
# -------------------------------------------------
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# -------------------------------------------------
# Sidebar – User Input
# -------------------------------------------------
st.sidebar.header("Customer Information")

user_input = {}

binary_features = [
    "gender",
    "Near_Location",
    "Partner",
    "Promo_friends",
    "Phone",
    "Group_visits"
]

for col in binary_features:
    user_input[col] = st.sidebar.selectbox(
        col, [0, 1],
        key=col,
        on_change=lambda: st.session_state.update({"has_input": True})
    )

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
    user_input[col] = st.sidebar.number_input(
        col,
        value=0.0,
        key=col,
        on_change=lambda: st.session_state.update({"has_input": True})
    )

input_df = pd.DataFrame([user_input])
input_df = input_df[feature_columns]

# -------------------------------------------------
# Sidebar – Model Settings
# -------------------------------------------------
st.sidebar.header("Model Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "XGBoost"],
    on_change=lambda: st.session_state.update({"has_input": True})
)

threshold = 0.5
if model_choice == "XGBoost":
    threshold = st.sidebar.slider(
        "Churn Threshold",
        0.1, 0.9, 0.35,
        on_change=lambda: st.session_state.update({"has_input": True})
    )

# -------------------------------------------------
# Initial Info Screen
# -------------------------------------------------
if not st.session_state.has_input:
    st.info(
        "👈 **Please enter customer information from the left sidebar.**\n\n"
        "This application predicts **gym customer churn risk** and provides "
        "**actionable insights** to help retain customers.\n\n"
        "**After entering data, you will see:**\n"
        "- Churn probability\n"
        "- Stay / Churn decision\n"
        "- Key factors behind the decision\n"
        "- Retention improvement suggestions\n"
        "- Optional SHAP explanation"
    )

else:
    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    if model_choice == "Random Forest":
        proba = rf_model.predict_proba(input_df)[0][1]
    else:
        proba = xgb_model.predict_proba(input_df)[0][1]

    prediction = int(proba >= threshold)

    # -------------------------------------------------
    # Rule-based explanation
    # -------------------------------------------------
    reasons_positive = []
    reasons_negative = []

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

    if input_df.loc[0, "Contract_period"] >= 6:
        reasons_negative.append("long-term contract")

    if input_df.loc[0, "Avg_class_frequency_current_month"] >= 3:
        reasons_negative.append("high class attendance")

    if input_df.loc[0, "Partner"] == 1:
        reasons_negative.append("partner membership")

    if input_df.loc[0, "Promo_friends"] == 1:
        reasons_negative.append("friend promotion")

    # -------------------------------------------------
    # Results
    # -------------------------------------------------
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{proba:.2%}")

    with col2:
        if prediction == 1:
            st.error("🚨 Customer WILL CHURN")
            if reasons_positive:
                st.markdown("**Main reasons:** " + ", ".join(reasons_positive[:3]))
        else:
            st.success("✅ Customer WILL STAY")
            if reasons_negative:
                st.markdown("**Decision based on:** " + ", ".join(reasons_negative[:3]))

    # -------------------------------------------------
    # Retention Improvement Analysis
    # -------------------------------------------------
    ideal_values = {
        "Avg_class_frequency_current_month": 3,
        "Avg_class_frequency_total": 3,
        "Lifetime": 12,
        "Contract_period": 6
    }

    weak_features = []

    for feature, ideal in ideal_values.items():
        actual = input_df.loc[0, feature]
        if actual < ideal:
            weak_features.append({
                "Feature": feature,
                "Current Value": actual,
                "Recommended Value": ideal
            })

    weak_df = pd.DataFrame(weak_features)

    if prediction == 1 and not weak_df.empty:
        st.subheader("📉 Retention Improvement Analysis")

        st.markdown(
            "The following features are **below recommended levels**. "
            "Improving them may reduce churn risk."
        )

        st.bar_chart(
            weak_df.set_index("Feature")[["Current Value", "Recommended Value"]]
        )

        st.markdown("### 🔧 Suggested Actions")

        for _, row in weak_df.iterrows():
            if row["Feature"] == "Avg_class_frequency_current_month":
                st.markdown("- 📅 Encourage more class attendance (free classes, reminders, PT offers)")
            elif row["Feature"] == "Lifetime":
                st.markdown("- 🎁 Loyalty campaigns to increase customer lifetime")
            elif row["Feature"] == "Contract_period":
                st.markdown("- 📄 Offer discounted long-term contracts")

    # -------------------------------------------------
    # SHAP Explanation (optional)
    # -------------------------------------------------
    show_shap = st.checkbox("🔍 Show explanation (SHAP)")

    if show_shap and model_choice == "XGBoost":
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(input_df)

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
st.caption("Gym Churn Prediction • Random Forest & XGBoost • Explainable AI")
