import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, classification_report  
from imblearn.over_sampling import SMOTE  
import streamlit as st  
import matplotlib.pyplot as plt  
import time  
from sklearn.tree import export_graphviz  
import graphviz  
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

# Define and train the RandomForest classifier  
rf = RandomForestClassifier(random_state=42, class_weight='balanced')  

# Hyperparameter tuning with RandomizedSearchCV  
param_dist = {  
    'n_estimators': [100, 200, 300],  
    'max_depth': [None, 10, 20, 30],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4],  
    'max_features': ['auto', 'sqrt'],  
    'bootstrap': [True, False]  
}  

random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=100, cv=3, scoring='accuracy', n_jobs=-1)  
random_search.fit(X_train, y_train)  

# Get the best model  
best_rf = random_search.best_estimator_  

# Evaluate the model  
y_pred = best_rf.predict(X_test)  
accuracy = accuracy_score(y_test, y_pred)  
report = classification_report(y_test, y_pred, target_names=np.unique(y))  

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
    input_transformed = preprocessor.transform(input_df)  # Add your preprocessing steps if needed  
    prediction = best_rf.predict(input_transformed)  
    return prediction[0]  

# Display the user inputs  
if st.button("Get Recommendation"):  
    with st.spinner("Generating recommendation..."):  
        time.sleep(2)  # Simulate processing time  
        recommended_strategy = recommend_strategy(input_data)  
        st.success(f"Recommended Strategy: {recommended_strategy}")  

# Random Forest Visualization  
st.subheader("Random Forest Visualization")  

# Allow user to select a tree index to display  
tree_index = st.slider("Select a Tree Index", 0, len(best_rf.estimators_) - 1, 0)  

# Get the selected tree  
selected_tree = best_rf.estimators_[tree_index]  

# Export tree to Graphviz format  
dot_data = export_graphviz(  
    selected_tree,   
    feature_names=input_df.columns,  # Ensure this corresponds to your preprocessed features  
    class_names=best_rf.classes_,   
    filled=True,   
    rounded=True,   
    special_characters=True,  
    node_ids=True  
)  

# Create Graphviz Source  
graph = graphviz.Source(dot_data, format="png", engine="dot")  

# Display the tree  
st.graphviz_chart(graph.source)  

st.markdown(f"Showing Tree {tree_index} of {len(best_rf.estimators_)} in the Random Forest.")  

# Feature Importance Analysis  
importances = np.mean([tree.feature_importances_ for tree in best_rf.estimators_], axis=0)  

# Create a DataFrame for visualization  
feature_importance_df = pd.DataFrame({  
    'Feature': input_df.columns,  # Ensure this corresponds to your preprocessed features  
    'Importance': importances  
}).sort_values(by='Importance', ascending=False)  

# Wrap feature names for better readability  
feature_importance_df['Feature'] = feature_importance_df['Feature'].apply(  
    lambda x: '\n'.join(textwrap.wrap(x, width=30))  # Adjust wrap width as needed  
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

# Display accuracy and classification report  
st.subheader("Model Performance")  
st.write(f"Model Accuracy: {accuracy:.2f}")  
st.text(report)
