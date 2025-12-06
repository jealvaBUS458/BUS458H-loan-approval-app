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
    layout="c
