import streamlit as st
import pandas as pd
import joblib

# Load model
try:
    with open("best.pkl", "rb") as file:
        best_model = joblib.load(file)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# App title
st.title("📊 Customer Churn Prediction")

st.markdown("Enter customer details below to predict if they will churn.")

# Input fields (only key ones for simplicity)
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", 0, 100, 12)
monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 1000.0, 70.0)
total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Create DataFrame with required columns
columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

# Fill other required columns with default values
data = {
    'gender': gender,
    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
    'Partner': "No",
    'Dependents': "No",
    'tenure': tenure,
    'PhoneService': "Yes",
    'MultipleLines': "No",
    'InternetService': "Fiber optic",
    'OnlineSecurity': "No",
    'OnlineBackup': "No",
    'DeviceProtection': "No",
    'TechSupport': "No",
    'StreamingTV': "No",
    'StreamingMovies': "No",
    'Contract': contract,
    'PaperlessBilling': "Yes",
    'PaymentMethod': "Electronic check",
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

input_df = pd.DataFrame([data], columns=columns)

# Predict button
if st.button("🔮 Predict Churn"):
    try:
        prediction = best_model.predict(input_df)[0]
        prob = best_model.predict_proba(input_df)[0][1]

        st.subheader("📈 Prediction Result:")
        if prediction == 1:
            st.error(f"Customer is likely to CHURN (Probability: {prob*100:.2f}%)")
        else:
            st.success(f"Customer will STAY (Probability: {(1-prob)*100:.2f}%)")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
