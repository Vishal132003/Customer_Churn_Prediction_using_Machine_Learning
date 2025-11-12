import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
try:
    with open("best.pkl", "rb") as file:
        best_model = joblib.load(file)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Streamlit UI
st.title("📊 Customer Churn Prediction")
st.markdown("Provide the details below to check if a customer is likely to churn.")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 1000.0, 70.0)
total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

# Prepare input DataFrame
input_data = pd.DataFrame([{
    'gender': gender,
    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}])

# Prediction button
if st.button("🔮 Predict Churn"):
    try:
        prediction = best_model.predict(input_data)[0]
        prob = best_model.predict_proba(input_data)[0][1]

        st.subheader("📈 Prediction Result:")
        if prediction == 1:
            st.error(f"Customer is likely to CHURN ❌ (Probability: {prob*100:.2f}%)")
        else:
            st.success(f"Customer will STAY ✅ (Probability: {(1-prob)*100:.2f}%)")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
