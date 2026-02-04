import streamlit as st
import pandas as pd
import joblib
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load the AI Assets
try:
    rf_model = joblib.load('rf_model.pkl')
    # You MUST use the scaler from your notebook or the numbers will be wrong!
    scaler = joblib.load('scaler.pkl') 
except:
    st.error("Missing model files! Run the 'joblib.dump' lines in your notebook first.")

# 2. Load the RAG Librarian
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Ensure './chroma_db' matches the folder name created in your notebook
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 3. UI Setup
st.set_page_config(page_title="Credit Risk AI", page_icon="🏦")
st.title("🏦 Smart Loan Officer")

# Inputs (Match these to your most important features)
col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Annual Income ($)", min_value=0, value=50000)
    emp_length = st.number_input("Years of Employment", min_value=0, value=5)
with col2:
    loan_amt = st.number_input("Loan Amount ($)", min_value=0, value=10000)
    age = st.number_input("Age", min_value=18, value=25)

dti = loan_amt / income if income > 0 else 0

if st.button("Run Credit Audit"):
    # ⚠️ CRITICAL STEP: Construct the full feature list
    # Your model expects the EXACT same columns as X_train.columns in your notebook.
    # We create a dictionary with ALL columns and fill missing ones with 0 or averages.
    
    input_data = pd.DataFrame({
        'person_age': [age],
        'person_income': [income],
        'person_emp_length': [emp_length],
        'loan_amnt': [loan_amt],
        'dti_ratio': [dti],
        # Add any other columns your model needs here (e.g., encoded categories)
    })
    
    # 1. Scale the data (Important!)
    input_scaled = scaler.transform(input_data)
    
    # 2. Predict
    pred = rf_model.predict(input_scaled)[0]
    
    if pred == 1:
        st.error("### Decision: REJECTED")
        # RAG Explanation
        query = f"Risk factor for applicant with DTI {dti:.2f} and income {income}"
        docs = vectorstore.similarity_search(query, k=1)
        st.warning(f"**AI Justification:** {docs[0].page_content}")
    else:
        st.success("### Decision: APPROVED")
        st.info("Applicant meets low-risk criteria.")