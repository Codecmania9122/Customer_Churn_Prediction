import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px

# --- JSON DB Path ---
JSON_DB_PATH = r"D:\Projects\DS_Proj_1\Deployment\JSON_Database\all_predictions.json"

# --- Streamlit App ---
st.set_page_config(page_title="Churn Monitoring Dashboard", layout="wide")
st.title("Customer Churn Live Dashboard")

# --- Load data function ---
@st.cache_data
def load_data():
    if os.path.exists(JSON_DB_PATH):
        with open(JSON_DB_PATH, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df
    else:
        return pd.DataFrame()  # empty if file doesn't exist

# --- Load JSON ---
df = load_data()

# --- Check if data exists ---
if df.empty:
    st.warning("No predictions available yet.")
else:
    # --- Overall Churn Statistics ---
    st.subheader("Churn Overview")
    total_preds = len(df)
    churned = df[df['PredictedChurn'] == 'Yes'].shape[0]
    not_churned = df[df['PredictedChurn'] == 'No'].shape[0]
    churn_rate = churned / total_preds * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", total_preds)
    col2.metric("Total Churned", churned)
    col3.metric("Churn Rate (%)", f"{churn_rate:.2f}%")

    # --- Churn by Contract Type ---
    st.subheader("Churn by Contract Type")
    contract_counts = df.groupby(['Contract', 'PredictedChurn']).size().reset_index(name='count')
    fig_contract = px.bar(contract_counts, x='Contract', y='count', color='PredictedChurn',
                          barmode='group', text='count', title="Churn by Contract Type")
    st.plotly_chart(fig_contract, use_container_width=True)

    # --- Churn Probability Distribution ---
    st.subheader("Churn Probability Distribution")
    fig_prob = px.histogram(df, x='ChurnProbability', nbins=20, color='PredictedChurn',
                            title="Churn Probability Distribution")
    st.plotly_chart(fig_prob, use_container_width=True)

    # --- Latest Predictions Table ---
    st.subheader("Latest Predictions")
    st.dataframe(df.sort_values(by='timestamp', ascending=False).head(10))

# --- Refresh Button ---
if st.button("Refresh Dashboard"):
    st.experimental_rerun()
