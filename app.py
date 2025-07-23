import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="wide"
)

# Title
st.title("Salary Prediction App 💼")
st.write("This app predicts salary based on user input features.")

# Load models
model = joblib.load('best_gradient_boosting_model.pkl')
gender_encoder = joblib.load('encoders/gender_encoder.pkl')
education_encoder = joblib.load('encoders/education_encoder.pkl')
job_title_columns = joblib.load('encoders/job_title_columns.pkl')
feature_names = joblib.load('encoders/feature_names.pkl')

st.markdown("---")

# Input section
st.subheader("📝 Enter Employee Details")

# Get options from encoders
gender_options = gender_encoder.classes_.tolist()
education_options = education_encoder.classes_.tolist()
job_options = job_title_columns.copy()

# Create input fields in columns for better layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", options=gender_options)
    education = st.selectbox("Education Level", options=education_options)
    age = st.slider("Age", min_value=18, max_value=65, value=30)

with col2:
    job = st.selectbox("Job Title", options=job_options)
    experience = st.slider("Years of Experience", min_value=0, max_value=40, value=5)

# Predict button (full width at bottom)
st.markdown("---")
if st.button("Predict Salary", type="primary", use_container_width=True):
    # Encode categorical inputs
    gender_encoded = gender_encoder.transform([gender])[0]
    education_encoded = education_encoder.transform([education])[0]
    
    # Create input data
    input_data = {}
    input_data['Gender'] = gender_encoded
    input_data['Age'] = age
    input_data['Education Level'] = education_encoded
    input_data['Years of Experience'] = experience
    
    # Initialize all job title columns to 0
    for job_col in job_title_columns:
        input_data[job_col] = 0
    
    # Set the selected job to 1
    if job in job_title_columns:
        input_data[job] = 1
    
    # Create DataFrame and ensure column order matches model expectations
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names]
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Display result
    st.success(f"💰 **Estimated Salary: Rs. {prediction:,.2f}**")