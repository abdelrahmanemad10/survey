import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
# Load the data
df = pd.read_csv('./Survey.csv')

# Select relevant columns
columns = [
    'In which country is your laboratory or organization based ?',
    '  What is the type of organization your laboratory operates under?  ',
    'How many years of experience do you have in your field?',
    'What is your role in the organization? ',
    'How familiar are you with AI technologies in laboratory operations? ',
    'To what extent is AI currently used in your laboratory operations? ',
    'To what extent do the following barriers affect AI implementation in your laboratory? [Financial constraints]',
    'To what extent do the following barriers affect AI implementation in your laboratory? [Ethical concerns (e.g., data privacy, transparency)]',
    'To what extent do the following barriers affect AI implementation in your laboratory? [Staff resistance to change]',
    'To what extent do the following barriers affect AI implementation in your laboratory? [Lack of training and education]',
    'To what extent do the following barriers affect AI implementation in your laboratory? [Regulatory compliance issues]',
    'If you could prioritize one strategy, which would it be?'
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

# Split the data into features and target
X = df.drop('If you could prioritize one strategy, which would it be?', axis=1)
y = df['If you could prioritize one strategy, which would it be?']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Train a Gradient Boosting classifier
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Train a Support Vector Machine classifier
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# Train a Logistic Regression classifier
log_reg_model = LogisticRegression(random_state=42)
log_reg_model.fit(X_train, y_train)

# Evaluate the models
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
log_reg_pred = log_reg_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
gb_accuracy = accuracy_score(y_test, gb_pred)
svm_accuracy = accuracy_score(y_test, svm_pred)
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)

# Feature importance
feature_importance = rf_model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Correlation matrix
correlation_matrix = df.corr()

# Streamlit UI
st.title("Survey Data Analysis Dashboard")

# Sidebar for filtering
st.sidebar.header("Filter Options")
selected_country = st.sidebar.selectbox('Country', label_encoders['In which country is your laboratory or organization based ?'].inverse_transform(df['In which country is your laboratory or organization based ?'].unique()))
selected_lab_type = st.sidebar.selectbox('Type of Lab', label_encoders['  What is the type of organization your laboratory operates under?  '].inverse_transform(df['  What is the type of organization your laboratory operates under?  '].unique()))
selected_role = st.sidebar.selectbox('Role in the Organization', label_encoders['What is your role in the organization? '].inverse_transform(df['What is your role in the organization? '].unique()))
selected_financial_constraints = st.sidebar.selectbox('Financial Constraints', label_encoders['To what extent do the following barriers affect AI implementation in your laboratory? [Financial constraints]'].inverse_transform(df['To what extent do the following barriers affect AI implementation in your laboratory? [Financial constraints]'].unique()))
selected_ethical_concerns = st.sidebar.selectbox('Ethical Concerns', label_encoders['To what extent do the following barriers affect AI implementation in your laboratory? [Ethical concerns (e.g., data privacy, transparency)]'].inverse_transform(df['To what extent do the following barriers affect AI implementation in your laboratory? [Ethical concerns (e.g., data privacy, transparency)]'].unique()))
selected_staff_resistance = st.sidebar.selectbox('Staff Resistance', label_encoders['To what extent do the following barriers affect AI implementation in your laboratory? [Staff resistance to change]'].inverse_transform(df['To what extent do the following barriers affect AI implementation in your laboratory? [Staff resistance to change]'].unique()))
selected_lack_of_training = st.sidebar.selectbox('Lack of Training', label_encoders['To what extent do the following barriers affect AI implementation in your laboratory? [Lack of training and education]'].inverse_transform(df['To what extent do the following barriers affect AI implementation in your laboratory? [Lack of training and education]'].unique()))
selected_regulatory_compliance = st.sidebar.selectbox('Regulatory Compliance', label_encoders['To what extent do the following barriers affect AI implementation in your laboratory? [Regulatory compliance issues]'].inverse_transform(df['To what extent do the following barriers affect AI implementation in your laboratory? [Regulatory compliance issues]'].unique()))

# Filter data based on selections
filtered_df = df[
    (df['In which country is your laboratory or organization based ?'] == label_encoders['In which country is your laboratory or organization based ?'].transform([selected_country])[0]) &
    (df['  What is the type of organization your laboratory operates under?  '] == label_encoders['  What is the type of organization your laboratory operates under?  '].transform([selected_lab_type])[0]) &
    (df['What is your role in the organization? '] == label_encoders['What is your role in the organization? '].transform([selected_role])[0]) &
    (df['To what extent do the following barriers affect AI implementation in your laboratory? [Financial constraints]'] == label_encoders['To what extent do the following barriers affect AI implementation in your laboratory? [Financial constraints]'].transform([selected_financial_constraints])[0]) &
    (df['To what extent do the following barriers affect AI implementation in your laboratory? [Ethical concerns (e.g., data privacy, transparency)]'] == label_encoders['To what extent do the following barriers affect AI implementation in your laboratory? [Ethical concerns (e.g., data privacy, transparency)]'].transform([selected_ethical_concerns])[0]) &
    (df['To what extent do the following barriers affect AI implementation in your laboratory? [Staff resistance to change]'] == label_encoders['To what extent do the following barriers affect AI implementation in your laboratory? [Staff resistance to change]'].transform([selected_staff_resistance])[0]) &
    (df['To what extent do the following barriers affect AI implementation in your laboratory? [Lack of training and education]'] == label_encoders['To what extent do the following barriers affect AI implementation in your laboratory? [Lack of training and education]'].transform([selected_lack_of_training])[0]) &
    (df['To what extent do the following barriers affect AI implementation in your laboratory? [Regulatory compliance issues]'] == label_encoders['To what extent do the following barriers affect AI implementation in your laboratory? [Regulatory compliance issues]'].transform([selected_regulatory_compliance])[0])
]

# Decode the filtered data for display
for column in filtered_df.columns:
    filtered_df.loc[:, column] = label_encoders[column].inverse_transform(filtered_df[column])

# Display filtered data
st.subheader("Filtered Data")
st.write(filtered_df)

# Visualizations
st.subheader("Visualizations")

# Experience Distribution
fig_experience = px.histogram(df, x='How many years of experience do you have in your field?', title='Years of Experience Distribution')
st.plotly_chart(fig_experience)

# Familiarity with AI
fig_familiarity = px.histogram(df, x='How familiar are you with AI technologies in laboratory operations? ', title='Familiarity with AI Technologies')
st.plotly_chart(fig_familiarity)

# AI Usage
fig_ai_usage = px.histogram(df, x='To what extent is AI currently used in your laboratory operations? ', title='Current AI Usage in Laboratory Operations')
st.plotly_chart(fig_ai_usage)

# Barriers to AI Implementation
fig_barriers = px.histogram(df, x='To what extent do the following barriers affect AI implementation in your laboratory? [Financial constraints]', title='Financial Constraints')
st.plotly_chart(fig_barriers)

fig_barriers_ethics = px.histogram(df, x='To what extent do the following barriers affect AI implementation in your laboratory? [Ethical concerns (e.g., data privacy, transparency)]', title='Ethical Concerns')
st.plotly_chart(fig_barriers_ethics)

fig_barriers_staff = px.histogram(df, x='To what extent do the following barriers affect AI implementation in your laboratory? [Staff resistance to change]', title='Staff Resistance to Change')
st.plotly_chart(fig_barriers_staff)

fig_barriers_training = px.histogram(df, x='To what extent do the following barriers affect AI implementation in your laboratory? [Lack of training and education]', title='Lack of Training and Education')
st.plotly_chart(fig_barriers_training)

fig_barriers_regulatory = px.histogram(df, x='To what extent do the following barriers affect AI implementation in your laboratory? [Regulatory compliance issues]', title='Regulatory Compliance Issues')
st.plotly_chart(fig_barriers_regulatory)

# Recommended Strategy
fig_strategy = px.histogram(df, x='If you could prioritize one strategy, which would it be?', title='Recommended Strategy')
st.plotly_chart(fig_strategy)

# Country Distribution Pie Chart
st.subheader("Country Distribution")
country_counts = df['In which country is your laboratory or organization based ?'].value_counts()
country_names = label_encoders['In which country is your laboratory or organization based ?'].inverse_transform(country_counts.index)
fig_country = px.pie(values=country_counts, names=country_names, title='Distribution of Responses by Country')
st.plotly_chart(fig_country)

# Feature Importance
st.subheader("Feature Importance")
fig_importance = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance')
st.plotly_chart(fig_importance)

# Correlation Matrix
st.subheader("Correlation Matrix")
fig_corr, ax = plt.subplots()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig_corr)

# Advanced Statistical Analysis
st.subheader("Advanced Statistical Analysis")

# Hypothesis Testing: Chi-Squared Test
st.write("### Chi-Squared Test for Independence")
contingency_table = pd.crosstab(df['How familiar are you with AI technologies in laboratory operations? '], df['If you could prioritize one strategy, which would it be?'])
chi2, p, dof, ex = chi2_contingency(contingency_table)
st.write(f"Chi-Squared Test Statistic: {chi2}")
st.write(f"P-Value: {p}")

# Correlation Analysis: Spearman Correlation
st.write("### Spearman Correlation")
spearman_corr, _ = spearmanr(df['How familiar are you with AI technologies in laboratory operations? '], df['To what extent is AI currently used in your laboratory operations? '])
st.write(f"Spearman Correlation Coefficient: {spearman_corr}")

# Regression Analysis: Logistic Regression
st.write("### Logistic Regression Analysis")
log_reg_accuracy = log_reg_model.score(X_test, y_test)
st.write(f"Logistic Regression Accuracy: {log_reg_accuracy}")

# Model Comparison
st.subheader("Model Comparison")
model_comparison = pd.DataFrame({
    'Model': ['Random Forest', 'Gradient Boosting', 'Support Vector Machine', 'Logistic Regression'],
    'Accuracy': [rf_accuracy, gb_accuracy, svm_accuracy, log_reg_accuracy]
})
fig_model_comparison = px.bar(model_comparison, x='Model', y='Accuracy',title='Model Comparison')
st.plotly_chart(fig_model_comparison)

# Summary
st.subheader("Summary")
st.write("This dashboard provides an overview of the survey data, including distributions of experience, familiarity with AI, current AI usage, barriers to AI implementation, and recommended strategies. Use the filter options in the sidebar to explore the data for specific lab types and roles within the organization.")
st.write("The feature importance chart shows the impact of different features on the recommended strategy, and the correlation matrix provides insights into the relationships between different variables.")
st.write("Advanced statistical analysis includes hypothesis testing, correlation analysis, and regression analysis. Model comparison shows the performance of different machine learning models.")