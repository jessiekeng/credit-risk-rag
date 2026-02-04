import streamlit as st
import pandas as pd
import joblib
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. LOAD ASSETS ---
try:
    rf_model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
except:
    st.error("Model files not found. Ensure .pkl files are in the same folder as app.py")

# Load RAG
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# --- 2. UI ---
st.set_page_config(page_title="Credit Risk AI", page_icon="🏦")
st.title("🏦 Smart Loan Officer")

# Feature List (Directly from your X_train)
all_columns = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
               'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
               'dti_ratio', 'person_home_ownership_OTHER', 'person_home_ownership_OWN', 
               'person_home_ownership_RENT', 'loan_intent_EDUCATION', 
               'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 
               'loan_intent_PERSONAL', 'loan_intent_VENTURE', 'loan_grade_B', 
               'loan_grade_C', 'loan_grade_D', 'loan_grade_E', 'loan_grade_F', 
               'loan_grade_G', 'cb_person_default_on_file_Y']

# Sidebar Inputs
st.sidebar.header("Applicant Profile")
age = st.sidebar.number_input("Age", 18, 100, 25)
income = st.sidebar.number_input("Annual Income ($)", 0, 1000000, 50000)
loan_amt = st.sidebar.number_input("Loan Amount ($)", 0, 500000, 10000)
emp_length = st.sidebar.slider("Years Employed", 0, 40, 5)
int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 25.0, 11.0)

# Logic for derived features
dti = loan_amt / income if income > 0 else 0
loan_percent = loan_amt / income if income > 0 else 0

if st.button("Run Credit Audit"):
    # 3. ALIGNMENT: Create a blank template of all 23 columns
    input_df = pd.DataFrame(0, index=[0], columns=all_columns)
    
    # Fill numeric values
    input_df['person_age'] = age
    input_df['person_income'] = income
    input_df['person_emp_length'] = emp_length
    input_df['loan_amnt'] = loan_amt
    input_df['loan_int_rate'] = int_rate
    input_df['loan_percent_income'] = loan_percent
    input_df['cb_person_cred_hist_length'] = 2 # Default value
    input_df['dti_ratio'] = dti
    
    # 4. PREDICT
    # Scale first using the exact same scaler from training
    input_scaled = scaler.transform(input_df)
    pred = rf_model.predict(input_scaled)[0]
    
    # 5. RESULTS
    if pred == 1:
        st.error(f"### Decision: REJECTED (DTI: {dti:.2%})")
        # RAG Search
        query = f"Reason for rejection with DTI {dti:.2f} and income {income}"
        docs = vectorstore.similarity_search(query, k=1)
        st.warning(f"**Policy Explanation:** {docs[0].page_content}")
    else:
        st.success(f"### Decision: APPROVED (DTI: {dti:.2%})")
        st.info("The applicant meets the safe risk profile based on current bank policy.")