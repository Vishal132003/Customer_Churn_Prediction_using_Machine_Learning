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

st.title("📊 Customer Churn Prediction")

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

# Build DataFrame
data = pd.DataFrame([{
    'gender': gender,
    'SeniorCitizen': senior_citizen,
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

# Manual encoding (must match training preprocessing)
def preprocess(df):
    df = df.copy()

    # Convert Yes/No to 1/0
    yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in yes_no_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # SeniorCitizen to numeric
    df['SeniorCitizen'] = df['SeniorCitizen'].map({'Yes': 1, 'No': 0})

    # Encode gender
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    # Encode categorical features using simple mappings
    mapping = {
        'MultipleLines': {'No phone service': 0, 'No': 1, 'Yes': 2},
        'InternetService': {'No': 0, 'DSL': 1, 'Fiber optic': 2},
        'OnlineSecurity': {'No internet service': 0, 'No': 1, 'Yes': 2},
        'OnlineBackup': {'No internet service': 0, 'No': 1, 'Yes': 2},
        'DeviceProtection': {'No internet service': 0, 'No': 1, 'Yes': 2},
        'TechSupport': {'No internet service': 0, 'No': 1, 'Yes': 2},
        'StreamingTV': {'No internet service': 0, 'No': 1, 'Yes': 2},
        'StreamingMovies': {'No internet service': 0, 'No': 1, 'Yes': 2},
        'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
        'PaymentMethod': {
            'Electronic check': 0,
            'Mailed check': 1,
            'Bank transfer (automatic)': 2,
            'Credit card (automatic)': 3
        }
    }

    for col, map_dict in mapping.items():
        df[col] = df[col].map(map_dict)

    return df

processed_data = preprocess(data)

# Predict
if st.button("🔮 Predict Churn"):
    try:
        prediction = best_model.predict(processed_data)[0]
        prob = best_model.predict_proba(processed_data)[0][1]

        st.subheader("📈 Prediction Result:")
        if prediction == 1:
            st.error(f"Customer is likely to CHURN ❌ (Probability: {prob*100:.2f}%)")
        else:
            st.success(f"Customer will STAY ✅ (Probability: {(1-prob)*100:.2f}%)")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
