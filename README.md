# 📦 Customer Churn Prediction

## 🧠 Project Overview

This project focuses on predicting customer churn, which is when customers stop using a company’s services. By analyzing customer data, it builds a machine learning model to identify at-risk customers. The project includes:

- A dataset (`customer_churn_data.csv`)
- A Jupyter notebook (`notebook.ipynb`) for analysis and modeling
- A Streamlit web app (`app.py`) for user-friendly predictions

---

## ❓ What is Customer Churn?

Customer churn refers to the phenomenon where customers stop doing business with a company. Predicting churn is crucial as it allows businesses to take proactive measures to retain customers.

This project builds a **machine learning model** to predict churn using customer data and provides a user interface via Streamlit for predictions.

---

## 📊 Data Description

The dataset used is `customer_churn_data.csv` containing information on **1,000 customers**. Each row represents a customer with the following attributes:

| Column Name       | Description                                                  |
|-------------------|--------------------------------------------------------------|
| CustomerID        | Unique identifier for each customer                         |
| Age               | Age of the customer                                          |
| Gender            | Gender of the customer (Male/Female)                         |
| Tenure            | Number of months with the company                            |
| MonthlyCharges    | Amount charged monthly                                       |
| ContractType      | Type of contract (Month-to-Month, One-Year, Two-Year)        |
| InternetService   | Type of internet service (Fiber Optic, DSL, None)            |
| TotalCharges      | Total amount charged to the customer                         |
| TechSupport       | Whether the customer has tech support (Yes/No)               |
| Churn             | Whether the customer has churned (Yes/No) - **Target**       |

---

## 📒 Analysis (notebook.ipynb)

The Jupyter Notebook contains a step-by-step analysis, including:

### 🔍 Data Exploration:

- Loaded the dataset using `pandas`
- Checked for missing values (e.g., handled "InternetService" nulls by filling empty strings)
- Visualized insights using `matplotlib` and `seaborn`:

  - 📈 Pie chart of churn distribution (showing class imbalance)
  - 📊 Bar charts comparing monthly charges by contract type
  - 📉 Histograms of tenure and monthly charges

📌 Findings:
- Customers who churn have:
  - Higher monthly charges (avg: **₹75.96** vs **₹62.55**)
  - Shorter tenure (avg: **17.48** vs **30.26** months)

---

### 🔧 Feature Engineering:

- Selected features: `Age`, `Gender`, `Tenure`, `MonthlyCharges`
- Encoded categorical variables:
  - `Gender`: Female = 1, Male = 0
  - `Churn`: Yes = 1, No = 0

---

### 🧹 Data Preprocessing:

- Split dataset into training (80%) and testing (20%) sets
- Standardized features using `StandardScaler`
- Saved the scaler as `scaler.pkl`

---

### 🤖 Model Building:

Trained and tuned multiple models using `GridSearchCV`:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

---

### 📊 Model Evaluation:

| Model               | Accuracy |
|---------------------|----------|
| Support Vector Machine (SVM) | 87.5%    |
| Random Forest       | 87.5%    |
| Logistic Regression | 87.0%    |
| KNN                 | 86.5%    |
| Decision Tree       | 82.5%    |

✅ Final Model: **SVM with linear kernel and C=0.01**, chosen for simplicity and performance.

---

### 💾 Model Saving:

- Saved trained model as `model.pkl`
- Saved feature scaler as `scaler.pkl`

---

## 🌐 Application (Streamlit)

The Streamlit web app (`app.py`) allows interactive predictions.

### 👤 User Inputs:

- **Age**: 10–100 years
- **Gender**: Male or Female
- **Tenure**: 0–130 months
- **Monthly Charge**: ₹30–₹150

### ▶️ On "Predict!" button click:

- Encodes gender input (Female=1, Male=0)
- Scales inputs using `scaler.pkl`
- Predicts using `model.pkl` (SVM)
- Displays result: `Churn` or `Not Churn`

---

## ⚙️ How to Use

### ✅ Web App

To use the web app:

```bash
pip install streamlit
streamlit run app.py
```

📌 Make sure `model.pkl` and `scaler.pkl` are in the **same directory** as `app.py`.

---

### ✅ Notebook

To explore analysis:

1. Open `notebook.ipynb` using Jupyter Notebook or JupyterLab
2. Ensure these libraries are installed:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

---

## 📁 Files in Repository

| File Name                | Description                                      |
|--------------------------|--------------------------------------------------|
| `customer_churn_data.csv`| Raw dataset                                      |
| `notebook.ipynb`         | Jupyter notebook for analysis and modeling       |
| `app.py`                 | Streamlit app for churn prediction               |
| `model.pkl`              | Trained SVM model                                |
| `scaler.pkl`             | Scaler used to preprocess user inputs            |

---

## ✅ Installation and Setup

Install all dependencies using:

```bash
pip install pandas numpy matplotlib scikit-learn joblib streamlit
```

Then run either:

- `notebook.ipynb` for analysis
- `streamlit run app.py` for prediction

---

## 📈 Conclusion

This project delivers a complete pipeline for **churn prediction** using machine learning. It includes:

- Exploratory data analysis
- Model training and evaluation
- A web app for real-time prediction

### 🔍 Key Takeaways:

- SVM model achieved **87.5% accuracy**
- Churners typically have higher monthly charges and shorter tenure
- Web app allows non-technical users to interact with the model

---

## 🚀 Future Improvements

- Use more features like `ContractType` or `TechSupport`
- Address class imbalance using SMOTE or reweighting
- Explore advanced models like XGBoost, LightGBM, or neural networks

---

This end-to-end churn prediction tool can help businesses identify and retain at-risk customers effectively.
