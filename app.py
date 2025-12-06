import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load model artifacts
# -------------------------------

@st.cache_resource
def load_model():
    model_data = joblib.load("my_model.pkl")
    return model_data

model_data = load_model()
model = model_data["model"]
scaler = model_data["scaler"]
feature_columns = model_data["feature_columns"]

# -------------------------------
# Pretty label → value mappings
# -------------------------------

REASON_OPTIONS = {
    "Cover an Unexpected Cost": "cover_an_unexpected_cost",
    "Credit Card Refinancing": "credit_card_refinancing",
    "Debt Consolidation": "debt_conslidation",
    "Home Improvement": "home_improvement",
    "Major Purchase": "major_purchase",
    "Other": "other",
}

FICO_GROUP_OPTIONS = {
    "Poor": "poor",
    "Fair": "fair",
    "Good": "good",
    "Very Good": "very_good",
    "Excellent": "excellent",
}

EMPLOYMENT_STATUS_OPTIONS = {
    "Full-Time": "full_time",
    "Part-Time": "part_time",
    "Unemployed": "unemployed",
}

# -------------------------------
# App Layout
# -------------------------------

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💳",
    layout="centered",
)

st.title("💳 Loan Approval Prediction App")

st.write(
    """
This app estimates the **probability that a loan application will be approved**  
using a Logistic Regression model built for your BUS458 final project.
"""
)

with st.sidebar:
    st.header("About this app")
    st.markdown(
        """
- Built as part of **BUS458 – From Data to Decisions**  
- Model: Logistic Regression  
- Target: Loan **Approved (0/1)**  
- Cutoff: **0.70** for a conservative recommendation

You can adjust the inputs and see how the predicted approval probability changes.
"""
    )
    st.markdown("---")
    st.caption("Author: Jonah Alva")

st.markdown("### Applicant Information")

# -------------------------------
# Input widgets
# -------------------------------

col1, col2 = st.columns(2)

with col1:
    reason_label = st.selectbox(
        "Reason for Loan",
        list(REASON_OPTIONS.keys()),
    )
    fico_label = st.selectbox(
        "FICO Score Group",
        list(FICO_GROUP_OPTIONS.keys()),
    )
    employment_label = st.selectbox(
        "Employment Status",
        list(EMPLOYMENT_STATUS_OPTIONS.keys()),
    )
    employment_sector = st.text_input(
        "Employment Sector",
        value="consumer_discretionary",
        help="Industry or sector for the applicant's employer.",
    )

with col2:
    lender = st.selectbox(
        "Preferred Lender",
        ["A", "B", "C"],
        format_func=lambda x: f"Lender {x}",
    )
    fico_score = st.number_input(
        "FICO Score",
        min_value=300.0,
        max_value=850.0,
        value=650.0,
        step=1.0,
    )
    requested_amount = st.number_input(
        "Requested Loan Amount ($)",
        min_value=5_000.0,
        max_value=2_500_000.0,
        value=40_000.0,
        step=1_000.0,
    )

st.markdown("### Financial Profile")

col3, col4 = st.columns(2)

with col3:
    monthly_income = st.number_input(
        "Monthly Gross Income ($)",
        min_value=0.0,
        max_value=50_000.0,
        value=5_000.0,
        step=100.0,
    )

with col4:
    housing_payment = st.number_input(
        "Monthly Housing Payment ($)",
        min_value=0.0,
        max_value=50_000.0,
        value=1_500.0,
        step=50.0,
    )

ever_bankrupt = st.selectbox(
    "Ever Bankrupt or Foreclosed?",
    ["No", "Yes"],
)
ever_bankrupt_flag = 1 if ever_bankrupt == "Yes" else 0

# -------------------------------
# Build model input row
# -------------------------------

# Map pretty labels → raw model values
reason_value = REASON_OPTIONS[reason_label]
fico_group_value = FICO_GROUP_OPTIONS[fico_label]
employment_status_value = EMPLOYMENT_STATUS_OPTIONS[employment_label]

input_dict = {
    "Reason": reason_value,
    "Requested_Loan_Amount": requested_amount,
    "FICO_score": fico_score,
    "Fico_Score_group": fico_group_value,
    "Employment_Status": employment_status_value,
    "Employment_Sector": employment_sector,
    "Monthly_Gross_Income": monthly_income,
    "Monthly_Housing_Payment": housing_payment,
    "Ever_Bankrupt_or_Foreclose": ever_bankrupt_flag,
    "Lender": lender,
}

input_df = pd.DataFrame([input_dict])

# One-hot encode exactly like training
input_dummies = pd.get_dummies(input_df, drop_first=True)

# Align columns with training feature matrix
input_aligned = input_dummies.reindex(columns=feature_columns, fill_value=0)

# Scale numerical features
input_scaled = scaler.transform(input_aligned)

# -------------------------------
# Prediction
# -------------------------------

st.markdown("---")
if st.button("🔮 Predict Approval Probability"):
    proba = model.predict_proba(input_scaled)[0, 1]
    threshold = 0.70

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.metric(
            "Predicted Probability of Approval",
            f"{proba:.2%}",
        )

    with col_right:
        decision = "✅ Recommend (Above Cutoff)" if proba >= threshold else "❌ Do Not Recommend"
        st.write(f"**Decision at cutoff {threshold:.2f}:**")
        st.write(decision)

    st.markdown("#### Interpretation")
    if proba >= threshold:
        st.write(
            "This applicant has a relatively **high predicted chance of approval** "
            "given the current model and cutoff. From the platform's perspective, "
            "they may be a good candidate to route to this lender."
        )
    else:
        st.write(
            "This applicant has a **lower predicted chance of approval** under the "
            "current model and cutoff. The platform may want to either decline or "
            "consider alternative lenders/terms."
        )
else:
    st.info("Fill out the form and click **Predict Approval Probability** to see the model's output.")
