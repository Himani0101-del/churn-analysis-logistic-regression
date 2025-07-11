import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load prediction results with churn probabilities
df = pd.read_csv("model_output.csv")  # Make sure it includes 'Churn_Probability' and all cleaned features

# 🎛 Sidebar Filters
st.sidebar.header("🔧 Filter Options")
threshold = st.sidebar.slider("Churn Risk Threshold", 0.0, 1.0, 0.5, 0.05)
income_type = st.sidebar.selectbox("Income Type", sorted(df['Type_Income'].dropna().unique()))
education = st.sidebar.selectbox("Education Level", sorted(df['EDUCATION'].dropna().unique()))
housing = st.sidebar.selectbox("Housing Type", sorted(df['Housing_type'].dropna().unique()))

# 🏷️ Dashboard Title
st.title("📊 Churn Prediction Dashboard")

# 🔍 Latest Model Metrics (From your classification report)
st.subheader("🚀 Model Performance")
st.metric("Accuracy", "64.36%")
st.metric("Recall (Churn)", "68%")
st.metric("F1 Score (Churn)", "0.66")

# 📉 Confusion Matrix Visualization
st.subheader("📉 Confusion Matrix")
conf_matrix = [[167, 109], [87, 187]]  # Updated values
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# 📌 Feature Importance (Optional — you can update with real model coefficients later)
features = ['Age', 'Years_Employed', 'Annual_income', 'Family_Members']
importances = [0.21, 0.17, 0.19, 0.14]  # Placeholder values
st.subheader("📌 Feature Importance")
fig2, ax2 = plt.subplots()
ax2.barh(features, importances, color='skyblue')
ax2.set_xlabel("Importance Score")
st.pyplot(fig2)

# 🧠 Customer-Level Predictions Table
st.subheader("🧠 High-Risk Customers")
filtered_df = df[
    (df['Churn_Probability'] >= threshold) &
    (df['Type_Income'] == income_type) &
    (df['EDUCATION'] == education) &
    (df['Housing_type'] == housing)
]
st.dataframe(filtered_df[[
    'GENDER', 'Age', 'Years_Employed', 'Annual_income', 'Family_Members',
    'Type_Income', 'EDUCATION', 'Housing_type', 'Churn_Probability'
]].sort_values(by='Churn_Probability', ascending=False).style.highlight_max(axis=0))

# Footer
st.markdown("---")
st.markdown("Created by Himani • Updated with Logistic Regression + SMOTE Results")
