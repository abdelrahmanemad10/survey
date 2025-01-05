import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency, spearmanr, kruskal
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('./Survey.csv')

# Print original column names to verify
print("Original Column Names:")
print(df.columns)

# Rename columns to simpler names
df.rename(columns={
    'In which country is your laboratory or organization based ?': 'Country',
    '  What is the type of organization your laboratory operates under?  ': 'Lab_Type',
    'How many years of experience do you have in your field?': 'Experience',
    'What is your role in the organization? ': 'Role',
    'How familiar are you with AI technologies in laboratory operations? ': 'AI_Familiarity',
    'To what extent is AI currently used in your laboratory operations? ': 'AI_Usage',
    'To what extent do the following barriers affect AI implementation in your laboratory? [Financial constraints]': 'Financial_Constraints',
    'To what extent do the following barriers affect AI implementation in your laboratory? [Ethical concerns (e.g., data privacy, transparency)]': 'Ethical_Concerns',
    'To what extent do the following barriers affect AI implementation in your laboratory? [Staff resistance to change]': 'Staff_Resistance',
    'To what extent do the following barriers affect AI implementation in your laboratory? [Lack of training and education]': 'Lack_Training',
    'To what extent do the following barriers affect AI implementation in your laboratory? [Regulatory compliance issues]': 'Regulatory_Compliance',
    'If you could prioritize one strategy, which would it be?': 'Prioritized_Strategy'
}, inplace=True)

# Print updated column names to verify
print("\nUpdated Column Names:")
print(df.columns)

# Select relevant columns
columns = [
    'Country',
    'Lab_Type',
    'Experience',
    'Role',
    'AI_Familiarity',
    'AI_Usage',
    'Financial_Constraints',
    'Ethical_Concerns',
    'Staff_Resistance',
    'Lack_Training',
    'Regulatory_Compliance',
    'Prioritized_Strategy'
]

df = df[columns]

# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Remove outliers using Z-scores
z_scores = np.abs((df - df.mean()) / df.std())
df = df[(z_scores < 3).all(axis=1)]

# Split the data into features and target
X = df.drop('Prioritized_Strategy', axis=1)
y = df['Prioritized_Strategy']

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models with OneVsRestClassifier
rf_model = OneVsRestClassifier(RandomForestClassifier(random_state=42))
rf_model.fit(X_train, y_train)

gb_model = OneVsRestClassifier(GradientBoostingClassifier(random_state=42))
gb_model.fit(X_train, y_train)

svm_model = OneVsRestClassifier(SVC(probability=True, random_state=42))
svm_model.fit(X_train, y_train)

log_reg_model = OneVsRestClassifier(LogisticRegression(random_state=42, max_iter=1000))
log_reg_model.fit(X_train, y_train)

xgb_model = OneVsRestClassifier(XGBClassifier(random_state=42))
xgb_model.fit(X_train, y_train)

# Evaluate the models
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
log_reg_pred = log_reg_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
gb_accuracy = accuracy_score(y_test, gb_pred)
svm_accuracy = accuracy_score(y_test, svm_pred)
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
xgb_accuracy = accuracy_score(y_test, xgb_pred)

# Define plot_roc_curve for multiclass
def plot_roc_curve(model, X_test, y_test, model_name, n_classes, label_encoders):
    y_prob = model.predict_proba(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc, model_name

# Determine number of classes
n_classes = len(label_encoders["Prioritized_Strategy"].classes_)

# Collect ROC curve data
models = [
    (rf_model, "Random Forest"),
    (gb_model, "Gradient Boosting"),
    (svm_model, "Support Vector Machine"),
    (log_reg_model, "Logistic Regression"),
    (xgb_model, "XGBoost")
]

roc_data = [plot_roc_curve(model, X_test, y_test, name, n_classes, label_encoders) for model, name in models]

# Feature importance for RandomForestClassifier
if hasattr(rf_model.estimator, 'feature_importances_'):
    feature_importance = rf_model.estimator.feature_importances_
else:
    # Handle OneVsRestClassifier with multiple models
    importances = []
    for estimator in rf_model.estimator:
        if hasattr(estimator, 'feature_importances_'):
            importances.append(estimator.feature_importances_)
    feature_importance = np.mean(importances, axis=0)  # Average importance from each classifier

features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Correlation matrix
correlation_matrix = df.corr()

# Streamlit UI
st.title("Survey Data Analysis Dashboard")

# Sidebar for filtering
st.sidebar.header("Filter Options")
selected_country = st.sidebar.selectbox('Country', label_encoders['Country'].inverse_transform(df['Country'].unique()))
selected_lab_type = st.sidebar.selectbox('Type of Lab', label_encoders['Lab_Type'].inverse_transform(df['Lab_Type'].unique()))
selected_role = st.sidebar.selectbox('Role in the Organization', label_encoders['Role'].inverse_transform(df['Role'].unique()))
selected_financial_constraints = st.sidebar.selectbox('Financial Constraints', label_encoders['Financial_Constraints'].inverse_transform(df['Financial_Constraints'].unique()))
selected_ethical_concerns = st.sidebar.selectbox('Ethical Concerns', label_encoders['Ethical_Concerns'].inverse_transform(df['Ethical_Concerns'].unique()))
selected_staff_resistance = st.sidebar.selectbox('Staff Resistance', label_encoders['Staff_Resistance'].inverse_transform(df['Staff_Resistance'].unique()))
selected_lack_of_training = st.sidebar.selectbox('Lack of Training', label_encoders['Lack_Training'].inverse_transform(df['Lack_Training'].unique()))
selected_regulatory_compliance = st.sidebar.selectbox('Regulatory Compliance', label_encoders['Regulatory_Compliance'].inverse_transform(df['Regulatory_Compliance'].unique()))

# Filter data based on selections
filtered_df = df[ 
    (df['Country'] == label_encoders['Country'].transform([selected_country])[0]) &
    (df['Lab_Type'] == label_encoders['Lab_Type'].transform([selected_lab_type])[0]) &
    (df['Role'] == label_encoders['Role'].transform([selected_role])[0]) &
    (df['Financial_Constraints'] == label_encoders['Financial_Constraints'].transform([selected_financial_constraints])[0]) &
    (df['Ethical_Concerns'] == label_encoders['Ethical_Concerns'].transform([selected_ethical_concerns])[0]) &
    (df['Staff_Resistance'] == label_encoders['Staff_Resistance'].transform([selected_staff_resistance])[0]) &
    (df['Lack_Training'] == label_encoders['Lack_Training'].transform([selected_lack_of_training])[0]) &
    (df['Regulatory_Compliance'] == label_encoders['Regulatory_Compliance'].transform([selected_regulatory_compliance])[0])
]

# Decode the filtered data for display
for column in filtered_df.columns:
    filtered_df.loc[:, column] = label_encoders[column].inverse_transform(filtered_df[column])

# Display filtered data
st.subheader("Filtered Data")
st.write(filtered_df)

# Visualizations
st.subheader("Visualizations")

# ROC Curves
for fpr, tpr, roc_auc, model_name in roc_data:
    fig = go.Figure()
    for i in range(n_classes):
        fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i], mode='lines', name=f'Class {i} (AUC = {roc_auc[i]:.2f})'))
    fig.update_layout(
        title=f'ROC Curve for {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template="plotly_dark"
    )
    st.plotly_chart(fig)

# Feature importance plot
st.subheader("Feature Importance (Random Forest)")
fig = px.bar(importance_df, x='Feature', y='Importance', title="Feature Importance", template="plotly_dark")
st.plotly_chart(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
st.pyplot()

