import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from recommend_baseline import recommend_popular
from recommend_new import recommend_content_based
from ab_test import assign_groups, simulate_clicks
from evaluation import evaluate_ctr
import mlflow


# Load data
df = pd.read_csv("data/interactions.csv")
df = assign_groups(df)

# Sidebar: Select User
user_ids = df['user_id'].unique()
selected_user = st.sidebar.selectbox("Select a user ID", user_ids)

# Determine group assignment
group = df[df['user_id'] == selected_user]['group'].values[0]

# Generate recommendation
if group == 'A':
    recs = recommend_popular(df)
else:
    user_history = df[df['user_id'] == selected_user]['item_id'].tolist()
    recs = recommend_content_based(user_history)

# Display
st.title("A/B Testing Recommender UI")
st.markdown(f"**User ID:** {selected_user}")
st.markdown(f"**Group:** {'Baseline' if group == 'A' else 'Content-Based'}")
st.markdown("### Recommended Items:")
st.write(recs)

# Simulate Clicks and CTR
df = simulate_clicks(df)
ctr = evaluate_ctr(df)

# Plot
st.markdown("### CTR Comparison")
fig, ax = plt.subplots()
ctr.plot(kind="bar", color=["skyblue", "salmon"], ax=ax)
plt.ylabel("Click-Through Rate")
plt.title("CTR by Group")
plt.grid(axis="y", linestyle="--", alpha=0.7)
st.pyplot(fig)

# MLflow logging
mlflow.set_experiment("AB_Test_Recommender")

with mlflow.start_run():
    mlflow.log_param("user_id", selected_user)
    mlflow.log_param("group", group)
    mlflow.log_metric("ctr_A", ctr['A'])
    mlflow.log_metric("ctr_B", ctr['B'])
    mlflow.log_metric("lift", (ctr['B'] - ctr['A']) / ctr['A'] * 100)