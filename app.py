import streamlit as st
import joblib
import pandas as pd
import os
import numpy as np

from metrics import weighted_recall_precision, weighted_scorer

st.set_page_config(
    page_title="Job/Internship Scam Detector",
    page_icon="🔎",
    layout="centered"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "scam_prediction_model.pkl")

def load_model():
    """Loads the model from the specified path."""
    try:
        with st.spinner("Loading model..."):
            model = joblib.load(MODEL_PATH)
        st.success("Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please check the path.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")
        return None

# Load the model
model = load_model()

if model is None:
    st.stop()

# App UI
st.title("Job/Internship Scam Detector 🔎")
st.markdown("##### Predict whether a job or internship might be a scam.")
st.markdown("---")

with st.form("scam_prediction_form"):
    st.write("Enter the job/internship details below:")
    
    # Text areas for better input experience
    Title = st.text_input("Job/Internship Title", help="e.g., 'Data Scientist Intern'", key="title")
    Description = st.text_area("Job/Internship Description", height=150, help="Paste the full job description here.", key="description")
    Location = st.text_input("Job/Internship Location", help="e.g., 'San Francisco, CA'", key="location")
    Requirements = st.text_area("Job/Internship Requirements", height=100, help="List the key requirements for the role.", key="requirements")
    
    submitted = st.form_submit_button("Predict")

if submitted:
    if not all([Title, Description, Location]):
        st.error("Please fill in the Title, Description, and Location fields.")
    else:
        input_data = [f"{Title} {Description} {Location} {Requirements}"]
        
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        with st.spinner("Analyzing..."):
            import time
            time.sleep(1)
            
            if prediction == 1:
                st.error("### This job is likely a scam! ❌")
                st.markdown(f"**Confidence:** **:red[{prediction_proba[1] * 100:.1f}%]**")
                st.markdown("---")
                st.info("⚠️ This prediction is based on the provided details and a machine learning model. Always exercise caution and do your own research.")
            else:
                st.success("### This job seems genuine! ✅")
                st.markdown(f"**Confidence:** **:green[{prediction_proba[0] * 100:.1f}%]**")
                st.markdown("---")
                st.info("✅ This prediction is a helpful guide but not a guarantee. We recommend verifying the company and listing details independently.")