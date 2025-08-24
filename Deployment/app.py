import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import json
import datetime

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # DS_PROJ_1 root
MODEL_PATH = os.path.join(BASE_DIR, "Models", "best_xgboost_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "Models", "preprocessors", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "Models", "preprocessors", "label_encoders.pkl")
JSON_DB_PATH = r"D:\Projects\DS_Proj_1\Deployment\JSON_Database\all_predictions.json"

# Create folder if not exists
os.makedirs(os.path.dirname(JSON_DB_PATH), exist_ok=True)

# --- Load model, scaler, encoders ---
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODER_PATH)   # dict of LabelEncoders

# --- Streamlit UI ---
st.title("Customer Churn Predictor")
st.write("Fill in the customer details below:")

with st.form(key='churn_form'):
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", ['0', '1'])
    Partner = st.selectbox("Partner", ["No", "Yes"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes","No phone service"])
    InternetService = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes","No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes","No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes","No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes","No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes","No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes","No internet service"])
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=5000.0, value=1500.0)

    submit_button = st.form_submit_button(label='Predict Churn')

# --- Prediction ---
if submit_button:
    # Prepare input dataframe
    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])

    categorical_cols = ["gender", "SeniorCitizen", "Partner", "Dependents",
                        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    # Encode categorical features
    encoded_features = []
    for col in categorical_cols:
        if col in encoders:
            encoded_col = encoders[col].transform(input_df[col])
            encoded_features.append(encoded_col.reshape(-1, 1))
        else:
            raise ValueError(f"Encoder for column '{col}' not found!")
    encoded = np.hstack(encoded_features)

    # Scale numerical features
    scaled = scaler.transform(input_df[numerical_cols])

    # Combine features
    final_input = np.hstack([encoded, scaled])

    # Predict
    pred = model.predict(final_input)[0]
    prob = model.predict_proba(final_input)[0][1]

    # Show predictions
    st.success(f"Churn Prediction: {'Yes' if pred==1 else 'No'}")
    st.info(f"Churn Probability: {prob:.4f}")

    # Prepare result entry
    entry = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "PredictedChurn": 'Yes' if pred==1 else 'No',
        "ChurnProbability": float(prob),
        "timestamp": str(datetime.datetime.now())
    }

    # Append entry to single JSON database
    if os.path.exists(JSON_DB_PATH):
        with open(JSON_DB_PATH, "r+") as f:
            data = json.load(f)
            data.append(entry)
            f.seek(0)
            json.dump(data, f, indent=4)
    else:
        with open(JSON_DB_PATH, "w") as f:
            json.dump([entry], f, indent=4)

    st.success(f"Prediction saved to local JSON database!")
