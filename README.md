# churn-analysis-logistic-regression
🧠 Credit Risk Classification & Churn Dashboard
This project tackles the challenge of evaluating credit card applications through a machine learning lens. By analyzing customer data and predicting application outcomes, the goal is to help financial institutions minimize risk while maintaining fairness and transparency.
🔍 Overview
Credit approval decisions can deeply impact both customer experience and a company’s financial exposure. Approving a high-risk applicant may result in future losses, while unnecessarily rejecting low-risk ones slows business growth. This project balances those priorities by identifying risky profiles using predictive modeling and wrapping it all in a dynamic, user-friendly dashboard.
📂 Dataset
The dataset was sourced from Kaggle(**https://www.kaggle.com/datasets/rohitudageri/credit-card-details**) and includes anonymized credit card applicant information—spanning income, education, housing, marital status, employment duration, and application outcomes.
⚙️ Workflow Highlights
- Data preprocessing: Missing value imputation, outlier handling, feature engineering (Age, Years_Employed)
- Class balancing: Used SMOTE to handle imbalanced labels
- Modeling: Applied  Logistic Regression (with RFE) 
- Interpretability: Evaluated metrics like recall, precision, confusion matrix
- Visualization: Deployed a Streamlit dashboard for stakeholder engagement and exploration
📊 Key Insights
While a Gradient Boosting model showed high overall accuracy, it underperformed in identifying rejected applications—meaning risky approvals could slip through. In contrast, the logistic model (enhanced with SMOTE and RFE) had better recall for rejections, making it the preferred choice despite lower headline accuracy.
This illustrates how raw performance metrics alone don’t determine a model’s business value—recall for high-risk outcomes is critical, especially in domains like credit lending.
