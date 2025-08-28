import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go

# --- JSON DB Path ---
JSON_DB_PATH = r"D:\Projects\DS_Proj_1\Deployment\JSON_Database\all_predictions.json"

# --- Streamlit App Config ---
st.set_page_config(page_title="Churn Monitoring Dashboard", layout="wide")
st.title("Customer Churn Live Dashboard")

# --- Load Data Function ---
@st.cache_data
def load_data():
    if os.path.exists(JSON_DB_PATH):
        with open(JSON_DB_PATH, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        return pd.DataFrame()

# --- Load Data ---
df = load_data()

# --- If no data ---
if df.empty:
    st.warning("No predictions available yet. Run predictions first.")
    st.stop()

# ================================
# Sidebar Filters
# ================================
st.sidebar.header("Filters")
selected_contract = st.sidebar.multiselect("Contract Type", df["Contract"].unique())
selected_payment = st.sidebar.multiselect("Payment Method", df["PaymentMethod"].unique())
selected_internet = st.sidebar.multiselect("Internet Service", df["InternetService"].unique())

# Apply filters dynamically
filtered_df = df.copy()
if selected_contract:
    filtered_df = filtered_df[filtered_df["Contract"].isin(selected_contract)]
if selected_payment:
    filtered_df = filtered_df[filtered_df["PaymentMethod"].isin(selected_payment)]
if selected_internet:
    filtered_df = filtered_df[filtered_df["InternetService"].isin(selected_internet)]

# ================================
# KPI Section
# ================================
st.subheader("Churn Overview")

total_preds = len(filtered_df)
churned = filtered_df[filtered_df['PredictedChurn'] == 'Yes'].shape[0]
not_churned = filtered_df[filtered_df['PredictedChurn'] == 'No'].shape[0]
churn_rate = churned / total_preds * 100 if total_preds > 0 else 0

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", total_preds)
col2.metric("Churned Customers", churned, delta=f"{churn_rate:.2f}% Churn Rate")
col3.metric("Retained Customers", not_churned)

# ================================
# Charts
# ================================

# Churn by Contract Type
st.subheader("Churn by Contract Type")
contract_counts = filtered_df.groupby(['Contract', 'PredictedChurn']).size().reset_index(name='count')
fig_contract = px.bar(
    contract_counts,
    x='Contract',
    y='count',
    color='PredictedChurn',
    barmode='group',
    text='count',
    title="Churn by Contract Type"
)
st.plotly_chart(fig_contract, use_container_width=True)

# Churn by Payment Method
st.subheader("Churn by Payment Method")
payment_counts = filtered_df.groupby(['PaymentMethod', 'PredictedChurn']).size().reset_index(name='count')
fig_payment = px.bar(
    payment_counts,
    x='PaymentMethod',
    y='count',
    color='PredictedChurn',
    barmode='group',
    text='count',
    title="Churn by Payment Method"
)
st.plotly_chart(fig_payment, use_container_width=True)

# Churn Probability Distribution
st.subheader("Churn Probability Distribution")
fig_prob = px.histogram(
    filtered_df,
    x='ChurnProbability',
    nbins=20,
    color='PredictedChurn',
    title="Churn Probability Distribution",
    marginal="box"
)
st.plotly_chart(fig_prob, use_container_width=True)

# Churn Rate Pie Chart
st.subheader("Churn Ratio")
fig_pie = go.Figure(data=[go.Pie(
    labels=["Churned", "Not Churned"],
    values=[churned, not_churned],
    hole=0.5,
    marker=dict(colors=["red", "green"])
)])
fig_pie.update_layout(title_text="Churn vs Retained Customers", showlegend=True)
st.plotly_chart(fig_pie, use_container_width=True)

# ================================
# Latest Predictions Table
# ================================
with st.expander("Show Latest Predictions"):
    st.dataframe(filtered_df.sort_values(by='timestamp', ascending=False).head(10))

# ================================
# Refresh Button
# ================================
if st.button("Refresh Dashboard"):
    st.cache_data.clear()
    st.rerun()
