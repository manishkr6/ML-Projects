import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("fraud_detection_pipeline.pkl")

# Page Config
st.set_page_config(page_title="Fraud Detection App", page_icon="🚨", layout="centered")

# Sidebar
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown(
        """
        This app predicts whether a financial transaction is **fraudulent**  
        using a pre-trained machine learning model.  
        ---
        🔹 Enter transaction details  
        🔹 Click **Predict**  
        🔹 Get fraud risk result instantly 🚀
        """
    )

# Main Title
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>💳 Fraud Detection Prediction</h1>", unsafe_allow_html=True)
st.write("Please enter the transaction details below and click **Predict** to check for fraud.")

st.divider()

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"])
    amount = st.number_input("💰 Amount", min_value=0.0, value=1000.0, step=100.0)
    oldbalanceOrg = st.number_input("🏦 Old Balance (Sender)", min_value=0.0, value=10000.0, step=100.0)

with col2:
    newbalanceOrig = st.number_input("🏦 New Balance (Sender)", min_value=0.0, value=9000.0, step=100.0)
    oldbalanceDest = st.number_input("👤 Old Balance (Receiver)", min_value=0.0, value=0.0, step=100.0)
    newbalanceDest = st.number_input("👤 New Balance (Receiver)", min_value=0.0, value=0.0, step=100.0)

# Prediction Button
if st.button("🔍 Predict", use_container_width=True):
    # Prepare Data
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    # Predict
    prediction = model.predict(input_data)[0]

    # If model supports probability
    try:
        prob = model.predict_proba(input_data)[0][1]
        confidence = round(prob * 100, 2)
    except:
        confidence = None

    # Result Card
    if prediction == 1:
        st.error("🚨 **High Risk! This transaction may be fraudulent.**")
        if confidence is not None:
            st.write(f"**Fraud Probability:** {confidence}%")
    else:
        st.success("✅ **This transaction looks safe.**")
        if confidence is not None:
            st.write(f"**Fraud Probability:** {confidence}%")

    # Show input summary
    st.markdown("### 📝 Transaction Summary")
    st.table(input_data)
