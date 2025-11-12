import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load Model
# -----------------------------
import pickle
with open('best.pkl', 'rb') as file:
    model = pickle.load(file)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction App")
st.markdown(
    """
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
        }
        .stButton>button {
            background-color: #2e86de;
            color: white;
            border-radius: 10px;
            height: 45px;
            width: 200px;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #1b4f72;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("### 🧠 Enter Customer Details to Predict Churn")

# -----------------------------
# Input Section
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner?", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
    tenure = st.slider("Tenure (Months)", 0, 72, 10)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])

with col2:
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", 10.0, 150.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, 1000.0)

paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

# -----------------------------
# Prepare Input Data
# -----------------------------
if st.button("🔍 Predict Churn"):
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
        'Partner': [1 if partner == 'Yes' else 0],
        'Dependents': [1 if dependents == 'Yes' else 0],
        'tenure': [tenure],
        'PhoneService': [1 if phone_service == 'Yes' else 0],
        'MultipleLines': [1 if multiple_lines == 'Yes' else 0],
        'InternetService': [internet_service],
        'Contract': [contract],
        'PaymentMethod': [payment_method],
        'PaperlessBilling': [1 if paperless_billing == 'Yes' else 0],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # -----------------------------
    # Display Result
    # -----------------------------
    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ Customer is **likely to churn**.\n\n**Churn Probability:** {prob:.2%}")
    else:
        st.success(f"✅ Customer is **not likely to churn**.\n\n**Churn Probability:** {prob:.2%}")

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit and XGBoost")

