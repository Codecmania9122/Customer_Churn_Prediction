# 🧠 Customer Churn Prediction

This repository contains the implementation of a **Customer Churn Prediction System** built using the **Telco Customer Churn dataset by IBM**.  
The project applies **machine learning** and **deep learning** techniques to identify customers likely to discontinue services, helping businesses design proactive retention strategies.

---

## 🚀 Overview

The goal of this project is to build an **end-to-end churn prediction pipeline** — from data preprocessing and feature engineering to model training, evaluation, versioning, and deployment via Streamlit.  

The project explores both **supervised and unsupervised learning strategies** to uncover customer churn patterns and improve predictive accuracy.  
It integrates **DVC for data versioning**, **MLflow (via DagsHub)** for experiment tracking, and a **CI/CD pipeline** for automated testing and deployment — ensuring a fully reproducible, production-ready ML workflow.

---

## 💡 Key Highlights

✅ Conducted extensive **Exploratory Data Analysis (EDA)** on IBM’s Telco Customer Churn dataset to uncover behavioral and demographic churn indicators.  
✅ Applied **data preprocessing, feature extraction, and encoding techniques** to prepare the dataset for modeling.  
✅ Engineered new features to capture meaningful patterns in customer activity and contract attributes.  
✅ Experimented with a variety of **machine learning algorithms** including:
- Logistic Regression (LR)  
- Random Forest (RF)  
- Support Vector Machine (SVM)  
- Decision Tree (DT)  
- Naïve Bayes (NB)  
- Gradient Boosting (GB)  
- XGBoost  
- CatBoost  
- K-Means Clustering *(unsupervised)*  

✅ Built and evaluated **deep learning architectures**, including:
- Perceptron  
- Multi-Layer Perceptron (MLP)  
- Deep Neural Network (DNN)  

✅ Implemented a **DVC-based pipeline** to version data, preprocessing steps, and model artifacts for consistent experiment tracking.  
✅ Integrated a **CI/CD pipeline (GitHub Actions)** to automate linting, testing, and deployment of the Streamlit app.  
✅ Leveraged **MLflow experiment tracking on DagsHub** to log metrics, hyperparameters, and model artifacts for full reproducibility.  
✅ Tuned hyperparameters extensively and selected the **best-performing model** based on accuracy, precision, recall, F1-score, and AUC-ROC metrics.  
✅ Deployed the final model through a **Streamlit web interface** for real-time churn prediction.

---

## 🧰 Tech Stack

| Category | Tools / Frameworks |
|-----------|--------------------|
| **Language** | Python |
| **Libraries** | NumPy, Pandas, Scikit-learn, XGBoost, CatBoost, TensorFlow / Keras |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Pipeline & Tracking** | DVC, MLflow (DagsHub) |
| **Deployment & CI/CD** | Streamlit, GitHub Actions |
| **Version Control** | Git, GitHub |

---

## ⚙️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/Codecmania9122/Customer_Churn_Prediction.git

# Navigate to the folder
cd Customer_Churn_Prediction

# Install dependencies
pip install -r requirements.txt

# Reproduce pipeline (optional if using DVC)
dvc repro

# Run the Streamlit app
streamlit run app.py
