import streamlit as st
import pickle
import numpy as np

# -------------------------------
# Load the model safely
# -------------------------------
try:
    with open('best.pkl', 'rb') as file:
        best_model = pickle.load(file)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -------------------------------
# App UI Design
# -------------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>Customer Churn Prediction App</h1>
    <p style='text-align: center; color: gray;'>Predict whether a customer will churn based on key details</p>
    """,
    unsafe_allow_html=True
)

# Input form
with st.form("churn_form"):
    st.subheader("📋 Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
        tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=12)

    with col2:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    submitted = st.form_submit_button("🔍 Predict Churn")

# -------------------------------
# Prediction
# -------------------------------
if submitted:
    gender = 1 if gender == "Male" else 0
    senior_citizen = 1 if senior_citizen == "Yes" else 0
    contract_type = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]

    input_features = np.array([[gender, senior_citizen, tenure, monthly_charges, total_charges, contract_type]])
    prediction = best_model.predict(input_features)

    if prediction[0] == 1:
        st.error("⚠️ The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **likely to stay**.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed by Vishal Jadhav | Data Science Project")
