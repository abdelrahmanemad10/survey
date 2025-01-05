import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr, kruskal
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st

# Load and preprocess data
@st.cache
def load_data():
    data = pd.read_csv('./Survey.csv')
    
    # Handle missing values
    data.fillna(data.median(numeric_only=True), inplace=True)
    data.fillna(data.mode().iloc[0], inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in data.select_dtypes(include='object').columns:
        data[col] = le.fit_transform(data[col])
    
    # Remove outliers using Z-scores
    z_scores = (data - data.mean()) / data.std()
    outliers = (np.abs(z_scores) > 3).any(axis=1)
    data = data[~outliers]
    
    # Normalize numeric variables
    scaler = StandardScaler()
    num_cols = data.select_dtypes(include=np.number).columns
    data[num_cols] = scaler.fit_transform(data[num_cols])
    
    return data

data = load_data()

# Feature Engineering
# Correlation Matrix
correlation_matrix = data.corr()

# VIF Calculation
def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_data

vif = calculate_vif(data)

# Statistical Tests
# Kruskal-Wallis Test
group_col = 'In which country is your laboratory or organization based ?'
value_col = 'How familiar are you with AI technologies in laboratory operations? '
groups = [data[value_col][data[group_col] == grp] for grp in data[group_col].unique()]
stat, p = kruskal(*groups)

# Machine Learning
# Split data
target = 'If you could prioritize one strategy, which would it be?'
X = data.drop(columns=target)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# XGBoost Classifier
xgb = XGBClassifier()
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='roc_auc', cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred_xgb = best_model.predict(X_test)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])

# Streamlit app
st.title("Survey Analysis Dashboard")

# Display raw data
st.write("Raw Data Preview:")
st.dataframe(data.head())

# Correlation Matrix
st.subheader("Correlation Matrix")
fig_corr, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig_corr)

# VIF
st.subheader("Variance Inflation Factors (VIF)")
st.write(vif)

# Kruskal-Wallis Test
st.subheader("Kruskal-Wallis Test")
st.write(f"Statistic: {stat}, P-Value: {p}")

# Model Performance
st.subheader("Model Performance")
st.write("### Logistic Regression")
st.text(classification_report(y_test, y_pred_lr))

st.write("### XGBoost Best Model")
st.text(classification_report(y_test, y_pred_xgb))

# ROC Curve
st.subheader("ROC Curve")
fig_roc, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, label="XGBoost (AUC = {:.2f})".format(roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])))
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
ax.set_title("ROC Curve")
st.pyplot(fig_roc)

st.write("Analysis Completed!")
