import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
import joblib

# Constants
DATA_PATH = r"C:\Users\s30201011223559\Desktop\ml work\survey work\Survey.csv"
MODEL_PATH = './rf_model.pkl'
ENCODERS_PATH = './label_encoders.pkl'

# Load data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)  # Use read_csv for CSV files
        st.success("Data loaded successfully.")
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at {DATA_PATH}. Please ensure it exists.")
        return None
    except pd.errors.EmptyDataError:
        st.error("The CSV file is empty. Please provide a valid file.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# Preprocess data
def preprocess_data(df, label_encoders=None):
    selected_columns = [
        'In which country is your laboratory or organization based ?',
        'How many years of experience do you have in your field?',
        'What is your role in the organization? ',
        'How familiar are you with AI technologies in laboratory operations? ',
        'To what extent is AI currently used in your laboratory operations? ',
        'Which strategies have been implemented in your laboratory to overcome AI challenges? (Select all that apply) ',
        'If you could prioritize one strategy, which would it be?'
    ]

    # Exclude the 'Email Address' column if it exists
    if 'Email Address' in df.columns:
        df = df.drop(columns=['Email Address'])

    df = df[selected_columns].dropna()

    # Encode categorical variables
    if label_encoders is None:
        label_encoders = {}
        for column in df.columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    else:
        for column in df.columns:
            if column not in ['Email Address', 'Timestamp']:
                df[column] = label_encoders[column].transform(df[column])

    return df, label_encoders

# Train model
def train_model(df):
    X = df.drop('If you could prioritize one strategy, which would it be?', axis=1)
    y = df['If you could prioritize one strategy, which would it be?']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    st.write("### Model Performance")
    st.text(classification_report(y_test, y_pred))

    # Save the model and encoders
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoders, ENCODERS_PATH)

    return model

# Load model and encoders
def load_model_and_encoders():
    try:
        model = joblib.load(MODEL_PATH)
        label_encoders = joblib.load(ENCODERS_PATH)
        return model, label_encoders
    except FileNotFoundError:
        st.warning("Model or encoders not found. Retraining...")
        return None, None

# Recommend strategy
def recommend_strategy(input_data, model, label_encoders):
    input_df = pd.DataFrame([input_data])
    for column in input_df.columns:
        if column not in label_encoders:
            le = LabelEncoder()
            input_df[column] = le.fit_transform(input_df[column])
            label_encoders[column] = le
        else:
            if column not in ['Email Address', 'Timestamp']:
                input_df[column] = label_encoders[column].transform(input_df[column])
                
    prediction = model.predict(input_df)
    return label_encoders['If you could prioritize one strategy, which would it be?'].inverse_transform(prediction)[0]

# Feature importance visualization
def feature_importance_visualization(model, X):
    importance = permutation_importance(model, X, model.predict, random_state=42)
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance.importances_mean
    }).sort_values(by='Importance', ascending=False)

    st.bar_chart(feature_importances.set_index('Feature'))

# Streamlit UI
st.title("AI Strategy Recommendation System")
st.sidebar.header("Input Parameters")

# Load and preprocess data
data = load_data()
if data is not None:
    model, label_encoders = load_model_and_encoders()

    if model is None or label_encoders is None:
        data, label_encoders = preprocess_data(data)
        model = train_model(data)

    st.sidebar.write("### Enter Your Inputs")
    input_data = {}
    for column in data.columns[:-1]:
        if column not in ['Email Address', 'Timestamp']:
            unique_values = label_encoders[column].classes_
            input_data[column] = st.sidebar.selectbox(column, unique_values)

    st.subheader("User Input")
    st.write(input_data)

    recommended_strategy = recommend_strategy(input_data, model, label_encoders)
    st.subheader("Recommended Strategy")
    st.success(recommended_strategy)

    # Display feature importance
    st.write("### Feature Importance")
    X = data.drop('If you could prioritize one strategy, which would it be?', axis=1)
    feature_importance_visualization(model, X)
