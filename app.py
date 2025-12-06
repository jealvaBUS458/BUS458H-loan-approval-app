import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------- Load model artifacts --------
@st.cache_resource
def load_model():
    # Make sure this matches your filename in the repo
    model_data = joblib.load("my_model.pkl")
    return model_data

model_data = load_model()
model = model_data["model"]
scaler = model_data["scaler"]
feature_columns = model_data["feature_columns"]

st.title("Loan Approval Prediction App")

st.write("""
This app uses a Logistic Regression model trained in BUS458 to estimate the
probability that a loan application will be approved.
Fill in the fields below to get a prediction.
""")

# -------- User Inputs --------
reason = st.selectbox(
    "Reason for loan",
    [
        "cover_an_unexpected_cost",
        "credit_card_refinancing",
        "debt_conslidation",
        "home_improvement",
        "major_purchase",
        "other",
    ],
)

fico_group = st.selectbox(
    "FICO score group",
    ["poor", "fair", "good", "very_good", "excellent"],
)

employment_status = st.selectbox(
    "Employment status",
    ["full_time", "part_time", "unemployed"],
)

employment_sector = st.text_input(
    "Employment sector",
    value="consumer_discretionary",
)

lender = st.selectbox(
    "Lender",
    ["A", "B", "C"],
)

requested_amount = st.number_input(
    "Requested loan amount",
    min_value=5000.0,
    max_value=2500000.0,
    value=40000.0,
    step=1000.0,
)

fico_score = st.number_input(
    "FICO score",
    min_value=300.0,
    max_value=850.0,
    value=650.0,
    step=1.0,
)

monthly_income = st.number_input(
    "Monthly gross income",
    min_value=0.0,
    max_value=20000.0,
    value=5000.0,
    step=100.0,
)

housing_payment = st.number_input(
    "Monthly housing payment",
    min_value=0.0,
    max_value=50000.0,
    value=1500.0,
    step=50.0,
)

ever_bankrupt = st.selectbox(
    "Ever bankrupt or foreclosed?",
    ["No", "Yes"],
)
ever_bankrupt_flag = 1 if ever_bankrupt == "Yes" else 0

# -------- Build model input row --------
input_dict = {
    "Reason": reason,
    "Requested_Loan_Amount": requested_amount,
    "FICO_score": fico_score,
    "Fico_Score_group": fico_group,
    "Employment_Status": employment_status,
    "Employment_Sector": employment_sector,
    "Monthly_Gross_Income": monthly_income,
    "Monthly_Housing_Payment": housing_payment,
    "Ever_Bankrupt_or_Foreclose": ever_bankrupt_flag,
    "Lender": lender,
}

input_df = pd.DataFrame([input_dict])

# Apply same one-hot encoding as training
input_dummies = pd.get_dummies(input_df, drop_first=True)

# Align columns with training matrix (missing cols filled with 0)
input_aligned = input_dummies.reindex(columns=feature_columns, fill_value=0)

# Scale numeric features with the same scaler
input_scaled = scaler.transform(input_aligned)

# -------- Prediction --------
if st.button("Predict approval probability"):
    proba = model.predict_proba(input_scaled)[0, 1]
    st.metric("Predicted probability of approval", f"{proba:.2%}")

    threshold = 0.7  # match what you used in the notebook
    decision = "✅ Recommend" if proba >= threshold else "❌ Do not recommend"
    st.write(f"Decision at threshold {threshold:.2f}: **{decision}**")
