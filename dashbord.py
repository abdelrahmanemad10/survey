import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Preprocess data
def preprocess_data(df):
    # Select relevant columns
    columns = [
        'In which country is your laboratory or organization based ?',
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
    df.dropna(inplace=True)
    return df

# Encode categorical variables
def encode_data(df):
    label_encoders = {}
    for column in df.columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders

# Decode function for visualization
def decode_column(encoded_column, encoder):
    return encoder.inverse_transform(encoded_column)

# Train model
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Feature importance plot
def plot_feature_importance(importance_df):
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance')
    st.plotly_chart(fig)

# Correlation matrix plot
def plot_correlation_matrix(correlation_matrix):
    fig_corr, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig_corr)

# Visualizations
def plot_histograms(filtered_df):
    st.subheader("Visualizations")

    # Country Distribution
    country_counts = filtered_df['Decoded Country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Count']
    fig_country = px.bar(
        country_counts,
        x='Country',
        y='Count',
        title='Number of Respondents by Country',
        labels={'Country': 'Country', 'Count': 'Number of Respondents'},
        text='Count'
    )
    fig_country.update_traces(textposition='outside')
    st.plotly_chart(fig_country)

    # Experience Distribution
    fig_experience = px.histogram(filtered_df, x='How many years of experience do you have in your field?', title='Years of Experience Distribution')
    st.plotly_chart(fig_experience)

    # Familiarity with AI
    fig_familiarity = px.histogram(filtered_df, x='How familiar are you with AI technologies in laboratory operations? ', title='Familiarity with AI Technologies')
    st.plotly_chart(fig_familiarity)

    # AI Usage
    fig_ai_usage = px.histogram(filtered_df, x='To what extent is AI currently used in your laboratory operations? ', title='Current AI Usage in Laboratory Operations')
    st.plotly_chart(fig_ai_usage)

    # Barriers to AI Implementation
    barriers = [
        'Financial constraints', 'Ethical concerns (e.g., data privacy, transparency)', 
        'Staff resistance to change', 'Lack of training and education', 'Regulatory compliance issues'
    ]
    for barrier in barriers:
        fig_barrier = px.histogram(filtered_df, x=f'To what extent do the following barriers affect AI implementation in your laboratory? [{barrier}]', title=barrier)
        st.plotly_chart(fig_barrier)

    # Recommended Strategy
    fig_strategy = px.histogram(filtered_df, x='If you could prioritize one strategy, which would it be?', title='Recommended Strategy')
    st.plotly_chart(fig_strategy)

# Main function for the Streamlit app
def main():
    # Load and preprocess data
    df = load_data('./Survey.csv')
    df = preprocess_data(df)
    df, label_encoders = encode_data(df)

    # Split data for model training
    X = df.drop('If you could prioritize one strategy, which would it be?', axis=1)
    y = df['If you could prioritize one strategy, which would it be?']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Model Evaluation
    y_pred = model.predict(X_test)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    st.subheader("Model Evaluation")
    st.text(classification_rep)

    # Feature importance
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)
    plot_feature_importance(importance_df)

    # Correlation matrix
    correlation_matrix = df.corr()
    plot_correlation_matrix(correlation_matrix)

    # Decode country names for visualization
    df['Decoded Country'] = decode_column(df['In which country is your laboratory or organization based ?'], label_encoders['In which country is your laboratory or organization based ?'])

    # Streamlit UI
    st.title("Survey Data Analysis Dashboard")

    # Sidebar for filtering
    st.sidebar.header("Filter Options")
    selected_country = st.sidebar.selectbox('Country', df['Decoded Country'].unique())
    selected_role = st.sidebar.selectbox(
        'Role in the Organization',
        pd.Series(decode_column(df['What is your role in the organization? '], label_encoders['What is your role in the organization? '])).unique()
    )

    # Filter data based on selections
    filtered_df = df[(df['Decoded Country'] == selected_country) & 
                     (df['What is your role in the organization? '] == label_encoders['What is your role in the organization? '].transform([selected_role])[0])]

    # Display filtered data
    st.subheader("Filtered Data")
    st.write(filtered_df)

    # Generate visualizations for filtered data
    plot_histograms(filtered_df)

    # Summary
    st.subheader("Summary")
    st.write("This dashboard provides an overview of the survey data, including distributions of experience, familiarity with AI, current AI usage, barriers to AI implementation, and recommended strategies. Use the filter options in the sidebar to explore the data for specific countries and roles within the organization.")
    st.write("The feature importance chart shows the impact of different features on the recommended strategy, and the correlation matrix provides insights into the relationships between different variables.")

if __name__ == "__main__":
    main()
