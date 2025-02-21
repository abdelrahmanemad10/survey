import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  
from imblearn.over_sampling import SMOTE  
from sklearn.preprocessing import StandardScaler  
import streamlit as st  
import time  
import graphviz  
from sklearn.tree import export_graphviz  
import textwrap  

# Load the dataset  
try:  
    df = pd.read_csv('Survey.csv')  
except FileNotFoundError:  
    st.error("The data file was not found. Please check the path.")  
except pd.errors.EmptyDataError:  
    st.error("The data file is empty. Please provide a valid CSV file.")  

# Prepare features and target variable  
X = df.drop('target_label', axis=1)  # Replace 'target_label' with the actual class label  
y = df['target_label']  # Class labels  

# Handle class imbalance using SMOTE  
smote = SMOTE(random_state=42)  
X_resampled, y_resampled = smote.fit_resample(X, y)  

# Split the data into training and testing sets with stratification  
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)  

# Feature Scaling  
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  

# Train Random Forest with GridSearchCV for hyperparameter tuning  
rf_param_grid = {  
    'n_estimators': [100, 200, 300],  
    'max_depth': [None, 10, 20, 30],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4],  
    'max_features': ['auto', 'sqrt'],  
    'bootstrap': [True, False]  
}  

rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),   
                               param_grid=rf_param_grid,   
                               scoring='accuracy',   
                               n_jobs=-1,  
                               cv=3)  

rf_grid_search.fit(X_train_scaled, y_train)  
best_rf = rf_grid_search.best_estimator_  

# Train XGBoost Classifier  
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')  
xgb_model.fit(X_train_scaled, y_train)  

# Evaluate Models  
def evaluate_model(model, X_test, y_test):  
    y_pred = model.predict(X_test)  
    accuracy = accuracy_score(y_test, y_pred)  
    report = classification_report(y_test, y_pred)  
    cm = confusion_matrix(y_test, y_pred)  
    return accuracy, report, cm  

# Evaluate Random Forest  
rf_accuracy, rf_report, rf_cm = evaluate_model(best_rf, X_test_scaled, y_test)  

# Evaluate XGBoost  
xgb_accuracy, xgb_report, xgb_cm = evaluate_model(xgb_model, X_test_scaled, y_test)  

# Streamlit UI  
st.title("AI Strategy Recommendation System")  
st.markdown("""  
    Welcome to the **AI Strategy Recommendation System**!  
    This tool helps you identify the best strategy for implementing AI in your laboratory based on your inputs.  
    Fill out the form below to get started.  
""")  

# Sidebar for navigation  
st.sidebar.header("Navigation")  
st.sidebar.markdown("""  
    - **Home**: Overview of the tool.  
    - **Input Parameters**: Provide your inputs.  
    - **Results**: View recommendations and insights.  
""")  

# Input Parameters  
st.sidebar.header("Input Parameters")  

def user_input_features():  
    with st.sidebar.expander("General Information"):  
        country = st.selectbox('Country', df['In which country is your laboratory or organization based ?'].unique())  
        experience = st.selectbox('Years of Experience', df['How many years of experience do you have in your field?'].unique())  
        role = st.selectbox('Role in the Organization', df['What is your role in the organization? '].unique())  
        familiarity = st.selectbox('Familiarity with AI', df['How familiar are you with AI technologies in laboratory operations? '].unique())  
        ai_usage = st.selectbox('Current AI Usage', df['To what extent is AI currently used in your laboratory operations? '].unique())  

    with st.sidebar.expander("Challenges"):  
        financial_constraints = st.selectbox('Financial Constraints', df['To what extent do the following barriers affect AI implementation in your laboratory? [Financial constraints]'].unique())  
        ethical_concerns = st.selectbox('Ethical Concerns', df['To what extent do the following barriers affect AI implementation in your laboratory? [Ethical concerns (e.g., data privacy, transparency)]'].unique())  
        staff_resistance = st.selectbox('Staff Resistance', df['To what extent do the following barriers affect AI implementation in your laboratory? [Staff resistance to change]'].unique())  
        lack_of_training = st.selectbox('Lack of Training', df['To what extent do the following barriers affect AI implementation in your laboratory? [Lack of training and education]'].unique())  
        regulatory_compliance = st.selectbox('Regulatory Compliance', df['To what extent do the following barriers affect AI implementation in your laboratory? [Regulatory compliance issues]'].unique())  

    return {  
        'In which country is your laboratory or organization based ?': country,  
        'How many years of experience do you have in your field?': experience,  
        'What is your role in the organization? ': role,  
        'How familiar are you with AI technologies in laboratory operations? ': familiarity,  
        'To what extent is AI currently used in your laboratory operations? ': ai_usage,  
        'To what extent do the following barriers affect AI implementation in your laboratory? [Financial constraints]': financial_constraints,  
        'To what extent do the following barriers affect AI implementation in your laboratory? [Ethical concerns (e.g., data privacy, transparency)]': ethical_concerns,  
        'To what extent do the following barriers affect AI implementation in your laboratory? [Staff resistance to change]': staff_resistance,  
        'To what extent do the following barriers affect AI implementation in your laboratory? [Lack of training and education]': lack_of_training,  
        'To what extent do the following barriers affect AI implementation in your laboratory? [Regulatory compliance issues]': regulatory_compliance  
    }  

# Get user inputs  
input_data = user_input_features()  

# Display user inputs  
st.subheader("Your Inputs")  
st.json(input_data)  

# Function to recommend strategy  
def recommend_strategy(input_data):  
    input_df = pd.DataFrame([input_data])  
    input_transformed = scaler.transform(input_df)  # Scale input features  
    rf_prediction = best_rf.predict(input_transformed)  
    xgb_prediction = xgb_model.predict(input_transformed)  
    return rf_prediction[0], xgb_prediction[0]  

# Display the user inputs  
if st.button("Get Recommendation"):  
    with st.spinner("Generating recommendation..."):  
        time.sleep(2)  # Simulate processing time  
        rf_recommendation, xgb_recommendation = recommend_strategy(input_data)  
        st.success(f"Recommended Strategy (Random Forest): {rf_recommendation}")  
        st.success(f"Recommended Strategy (XGBoost): {xgb_recommendation}")  

# Confusion Matrices Visualization  
st.subheader("Model Performance")  

# Random Forest Confusion Matrix  
st.write("### Random Forest Confusion Matrix")  
fig1, ax1 = plt.subplots()  
sns.heatmap(rf_cm, annot=True, fmt='d', ax=ax1, cmap='Blues')  
ax1.set_title('Random Forest Confusion Matrix')  
ax1.set_xlabel('Predicted')  
ax1.set_ylabel('True')  
st.pyplot(fig1)  

# XGBoost Confusion Matrix  
st.write("### XGBoost Confusion Matrix")  
fig2, ax2 = plt.subplots()  
sns.heatmap(xgb_cm, annot=True, fmt='d', ax=ax2, cmap='Greens')  
ax2.set_title('XGBoost Confusion Matrix')  
ax2.set_xlabel('Predicted')  
ax2.set_ylabel('True')  
st.pyplot(fig2)  

# Display accuracy and classification reports  
st.subheader("Model Performance Metrics")  
st.write("### Random Forest Performance")  
st.write(f"Accuracy: {rf_accuracy:.2f}")  
st.text(rf_report)  

st.write("### XGBoost Performance")  
st.write(f"Accuracy: {xgb_accuracy:.2f}")  
st.text(xgb_report)
