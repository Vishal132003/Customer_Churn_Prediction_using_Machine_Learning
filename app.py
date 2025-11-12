import streamlit as st
import pickle

# Load the trained model from the file
with open('best.pkl', 'rb') as file:
    best_model = pickle.load(file)

# App title
st.title("Customer Churn Prediction App")

# Input fields
st.header("Enter Customer Details")
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Convert categorical to numeric
gender = 1 if gender == "Male" else 0
senior_citizen = 1 if senior_citizen == "Yes" else 0
contract_type = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]

# Predict button
if st.button("Predict Churn"):
    features = [[gender, senior_citizen, tenure, monthly_charges, total_charges, contract_type]]
    prediction = best_model.predict(features)
    
    if prediction[0] == 1:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is likely to stay.")

# Footer
st.markdown("---")
st.caption("Developed by Vishal Jadhav | Data Science Project")
