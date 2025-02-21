import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import time
import matplotlib.pyplot as plt
import textwrap

# Set Streamlit page configuration (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(page_title="AI Strategy Recommendation", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main { background-color: #f4f4f9; }
    .sidebar .sidebar-content { background-color: #e8e8ef; }
    .stButton>button { border-radius: 5px; background-color: #4CAF50; color: white; }
    .stProgress > div > div > div { background-color: #4CAF50; }
    .stExpander > div { background-color: #ffffff; border-radius: 5px; padding: 10px; }
    .stMarkdown h1 { color: #4CAF50; }
    .stMarkdown h2 { color: #2E86C1; }
    .stMarkdown h3 { color: #D35400; }
    </style>
    """, unsafe_allow_html=True
)

# Load the data
df = pd.read_csv('./Survey.csv')

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

# Separate features and target
X = df.drop('If you could prioritize one strategy, which would it be?', axis=1)
y = df['If you could prioritize one strategy, which would it be?']

# Encode categorical variables
categorical_cols = [
    'In which country is your laboratory or organization based ?',
    'How many years of experience do you have in your field?',
    'What is your role in the organization? ',
    'How familiar are you with AI technologies in laboratory operations? ',
    'To what extent is AI currently used in your laboratory operations? ',
    'To what extent do the following barriers affect AI implementation in your laboratory? [Financial constraints]',
    'To what extent do the following barriers affect AI implementation in your laboratory? [Ethical concerns (e.g., data privacy, transparency)]',
    'To what extent do the following barriers affect AI implementation in your laboratory? [Staff resistance to change]',
    'To what extent do the following barriers affect AI implementation in your laboratory? [Lack of training and education]',
    'To what extent do the following barriers affect AI implementation in your laboratory? [Regulatory compliance issues]'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='error'), categorical_cols)
    ]
)

# Preprocess the data
X_processed = preprocessor.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_processed, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Hyperparameter tuning for RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

# Evaluate the model
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

import streamlit as st
from sklearn.tree import export_graphviz
import graphviz

st.subheader("Random Forest Visualization")

# Allow user to select a tree index to display
tree_index = st.slider("Select a Tree Index", 0, len(best_rf.estimators_) - 1, 0)

# Get the selected tree
selected_tree = best_rf.estimators_[tree_index]

# Export tree to Graphviz format
dot_data = export_graphviz(
    selected_tree, 
    feature_names=preprocessor.get_feature_names_out(), 
    class_names=best_rf.classes_, 
    filled=True, 
    rounded=True, 
    special_characters=True
)

# Display the tree
st.graphviz_chart(dot_data)

st.markdown(f"Showing Tree {tree_index} of {len(best_rf.estimators_)} in the Random Forest.")

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
        country = st.selectbox(
            'Country',
            df['In which country is your laboratory or organization based ?'].unique()
        )
        experience = st.selectbox(
            'Years of Experience',
            df['How many years of experience do you have in your field?'].unique()
        )
        role = st.selectbox(
            'Role in the Organization',
            df['What is your role in the organization? '].unique()
        )
        familiarity = st.selectbox(
            'Familiarity with AI',
            df['How familiar are you with AI technologies in laboratory operations? '].unique()
        )
        ai_usage = st.selectbox(
            'Current AI Usage',
            df['To what extent is AI currently used in your laboratory operations? '].unique()
        )

    with st.sidebar.expander("Challenges"):
        financial_constraints = st.selectbox(
            'Financial Constraints',
            df['To what extent do the following barriers affect AI implementation in your laboratory? [Financial constraints]'].unique()
        )
        ethical_concerns = st.selectbox(
            'Ethical Concerns',
            df['To what extent do the following barriers affect AI implementation in your laboratory? [Ethical concerns (e.g., data privacy, transparency)]'].unique()
        )
        staff_resistance = st.selectbox(
            'Staff Resistance',
            df['To what extent do the following barriers affect AI implementation in your laboratory? [Staff resistance to change]'].unique()
        )
        lack_of_training = st.selectbox(
            'Lack of Training',
            df['To what extent do the following barriers affect AI implementation in your laboratory? [Lack of training and education]'].unique()
        )
        regulatory_compliance = st.selectbox(
            'Regulatory Compliance',
            df['To what extent do the following barriers affect AI implementation in your laboratory? [Regulatory compliance issues]'].unique()
        )

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

# Feature Importance Analysis
importances = best_rf.feature_importances_
feature_names = preprocessor.get_feature_names_out()

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Wrap feature names for better readability
feature_importance_df['Feature'] = feature_importance_df['Feature'].apply(
    lambda x: '\n'.join(textwrap.wrap(x, width=30))  # Adjust width as needed
)

# Plot feature importances
st.subheader("Feature Importance")
fig, ax = plt.subplots(figsize=(10, len(feature_importance_df) * 0.5))  # Adjust height dynamically
ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
ax.set_xlabel('Importance')
ax.set_title('Feature Importance')
plt.xticks(fontsize=8)  # Adjust font size
plt.yticks(fontsize=8)  # Adjust font size
st.pyplot(fig)

# Function to recommend strategy
def recommend_strategy(input_data):
    input_df = pd.DataFrame([input_data])
    input_transformed = preprocessor.transform(input_df)
    prediction = best_rf.predict(input_transformed)
    return prediction[0]

# Display the user inputs
if st.button("Get Recommendation"):
    with st.spinner("Generating recommendation..."):
        time.sleep(2)  # Simulate processing time
        recommended_strategy = recommend_strategy(input_data)
        st.success(f"Recommended Strategy: {recommended_strategy}")
