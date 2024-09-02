import streamlit as st
import pickle
import pandas as pd

# Define columns
categorical_cols = ['Shipping Address']
numerical_cols = ['Transaction Amount', 'Customer Age', 'Account Age Days', 'Transaction Hour', 'Transaction DOW']

# Load the trained model and encoders/scalers
with open('fraud_detection_model2.pkl', 'rb') as file:
    model = pickle.load(file)

with open('label_encoder2.pkl', 'rb') as file:
    le = pickle.load(file)

with open('scaler2.pkl', 'rb') as file:
    sc = pickle.load(file)

# UI Design
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin: auto;
        width: 80%;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        overflow-y: auto;
        max-height: 90vh;
    }
    body {
        background: linear-gradient(135deg, #6d5bba 30%, #8d58bf 90%);
        color: #fff;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 14px 20px;
        margin: 8px 0;
        border: none;
        cursor: pointer;
        border-radius: 25px;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stNumberInput input, .stTextInput input, .stSelectbox select {
        border-radius: 10px;
        padding: 10px;
    }
    .stNumberInput input:focus, .stTextInput input:focus, .stSelectbox select:focus {
        outline: none;
        box-shadow: 0 0 5px rgba(81, 203, 238, 1);
        border: 1px solid rgba(81, 203, 238, 1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)

st.title("🔍 Fraud Detection System")
st.write("Enter the transaction details to check if it is fraudulent:")

# User input fields
transaction_amount = st.number_input('💵 Transaction Amount', min_value=0.0, value=0.0, step=0.01, key='transaction_amount')
shipping_address = st.text_input('📍 Shipping Address', key='shipping_address')
customer_age = st.number_input('🎂 Customer Age', min_value=0, value=0, step=1, key='customer_age')
account_age_days = st.number_input('📅 Account Age (Days)', min_value=0, value=0, step=1, key='account_age_days')
transaction_hour = st.number_input('⏰ Transaction Hour', min_value=0, max_value=23, value=0, step=1, key='transaction_hour')
transaction_dow = st.selectbox('📅 Transaction Day of Week', [0, 1, 2, 3, 4, 5, 6], key='transaction_dow')

# Preprocessing user input
input_data = pd.DataFrame({
    'Transaction Amount': [transaction_amount],
    'Shipping Address': [shipping_address],
    'Customer Age': [customer_age],
    'Account Age Days': [account_age_days],
    'Transaction Hour': [transaction_hour],
    'Transaction DOW': [transaction_dow]
})

# Handle unseen labels for Shipping Address
shipping_address_labels = le.classes_.tolist()
if shipping_address in shipping_address_labels:
    shipping_address_transformed = le.transform([shipping_address])[0]
else:
    # Use a fallback value or handle unseen labels
    shipping_address_transformed = le.transform([shipping_address_labels[0]])[0]

input_data['Shipping Address'] = [shipping_address_transformed]

# Scale numerical columns
input_data[numerical_cols] = sc.transform(input_data[numerical_cols])

# Prediction
if st.button('🔍 Predict'):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("🚨 This transaction is likely fraudulent!")
    else:
        st.success("✅ This transaction appears to be legitimate.")

st.markdown('</div>', unsafe_allow_html=True)