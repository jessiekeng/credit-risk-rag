# 🏦 AI-Powered Credit Risk Decision Engine

An end-to-end Credit Risk assessment system that combines a **Random Forest Classifier** for risk prediction with **Retrieval-Augmented Generation (RAG)** to provide transparent, policy-backed explanations for loan decisions.



## 🌟 Overview
Most credit scoring models are "black boxes"—they give a decision but don't explain why. This project solves that by cross-referencing model predictions against a **Vector Database** containing internal bank credit policies.

### Key Features:
* **Predictive Modeling:** A Random Forest model trained on historical lending data to predict the probability of default.
* **Explainable AI (RAG):** Uses `all-MiniLM-L6-v2` embeddings and `ChromaDB` to retrieve specific policy rules (e.g., DTI limits) to justify rejections.
* **Interactive Dashboard:** A Streamlit web application for real-time credit auditing.
* **Robust Preprocessing:** Handles extreme outliers (like 144-year-old applicants) and engineered features like the **Debt-to-Income (DTI) Ratio**.

---

## 🏗️ The Architecture



1.  **Data Layer:** Preprocessed Credit Risk dataset with 23+ features.
2.  **Logic Layer:** Scikit-Learn Random Forest model + Scaler pipeline.
3.  **Knowledge Layer:** LangChain-based Vector Store containing `credit_policy.txt`.
4.  **UI Layer:** Streamlit interface for user input and decision visualization.

---

## 🛠️ Tech Stack
* **Language:** Python 3.12
* **ML:** Scikit-Learn, Pandas, NumPy, Joblib
* **LLM/RAG:** LangChain, HuggingFace (sentence-transformers), ChromaDB
* **Deployment:** Streamlit

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/credit-risk-rag.git](https://github.com/your-username/credit-risk-rag.git)
   cd credit-risk-rag

#### Create & activate the virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

#### Install dependencies

```bash
pip install -r requirements.txt
```

#### Run the application

```bash
streamlit run app.py
```
📊 Model Performance
Best Model: Random Forest Classifier (Optimized via K-Fold Cross-Validation).

Key Metric: Successfully identifies high-risk borrowers with a focus on DTI Ratio and Income thresholds.

Outlier Handling: Automated removal of illogical data (Age > 100, Employment Length > 60).

📝 Usage Example
When an applicant is rejected, the system doesn't just say "No." It retrieves the exact policy clause:

AI Justification: "Policy Violation: Applicant's Debt-to-Income ratio (45%) exceeds the internal 40% threshold for unsecured personal loans."

👤 Author
Zixing Keng Aspiring AI Developer & Data Scientist