import streamlit as st
import pickle
import numpy as np

# Load trained model
with open('best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# App title
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Prediction App")
st.markdown("### Predict whether a customer will churn or stay!")

# Sidebar input section
st.sidebar.header("🧾 Enter Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 150.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 2500.0)

# Convert categorical to numeric (example encoding)
gender = 1 if gender == "Male" else 0
senior = 1 if senior == "Yes" else 0
partner = 1 if partner == "Yes" else 0
dependents = 1 if dependents == "Yes" else 0

# Input array
features = np.array([[gender, senior, partner, dependents, tenure, monthly_charges, total_charges]])

# Predict
if st.sidebar.button("🔍 Predict"):
    prediction = best_model.predict(features)
    result = "Customer is likely to Churn ❌" if prediction[0] == 1 else "Customer is likely to Stay ✅"
    
    st.success(result)

# Footer
st.markdown("---")
st.markdown("👨‍💻 *Developed by Vishal Jadhav*")
