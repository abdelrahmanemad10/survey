import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import streamlit as st

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

# Encode categorical variables
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

X = df.drop('If you could prioritize one strategy, which would it be?', axis=1)
y = df['If you could prioritize one strategy, which would it be?']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

def recommend_strategy(input_data):
    input_df = pd.DataFrame([input_data])
    for column in input_df.columns:
        if input_df[column][0] not in label_encoders[column].classes_:
            st.error(f"Value '{input_df[column][0]}' for '{column}' is not recognized. Please select a valid value.")
            return None
        input_df[column] = label_encoders[column].transform(input_df[column])
    prediction = model.predict(input_df)
    return label_encoders['If you could prioritize one strategy, which would it be?'].inverse_transform(prediction)[0]

# New function to display user inputs
def display_user_inputs(user_inputs):
    st.markdown("## 📝 **Summary of Your Inputs**")

    # General Information Section
    st.markdown("""
        <h3 style="color: #4CAF50;"> General Information</h3>
        <ul>
            <li><strong>Country:</strong> <span style="color: #D35400;">{}</span></li>
            <li><strong>Years of Experience:</strong> <span style="color: #D35400;">{}</span></li>
            <li><strong>Role:</strong> <span style="color: #D35400;">{}</span></li>
            <li><strong>Familiarity with AI:</strong> <span style="color: #D35400;">{}</span></li>
            <li><strong>AI Usage in Operations:</strong> <span style="color: #D35400;">{}</span></li>
        </ul>
    """.format(
        user_inputs['In which country is your laboratory or organization based ?'],
        user_inputs['How many years of experience do you have in your field?'],
        user_inputs['What is your role in the organization? '],
        user_inputs['How familiar are you with AI technologies in laboratory operations? '],
        user_inputs['To what extent is AI currently used in your laboratory operations? ']
    ), unsafe_allow_html=True)

    # Barriers Section styled similarly
    st.markdown("""
        <h3 style="color: #4CAF50;"> Barriers to AI Implementation</h3>
        <ul>
            <li><strong> Financial Constraints:</strong> <span style="color: #D35400;">{}</span></li>
            <li><strong> Ethical Concerns:</strong> <span style="color: #D35400;">{}</span></li>
            <li><strong> Staff Resistance:</strong> <span style="color: #D35400;">{}</span></li>
            <li><strong> Lack of Training:</strong> <span style="color: #D35400;">{}</span></li>
            <li><strong> Regulatory Compliance:</strong> <span style="color: #D35400;">{}</span></li>
        </ul>
    """.format(
        user_inputs['To what extent do the following barriers affect AI implementation in your laboratory? [Financial constraints]'],
        user_inputs['To what extent do the following barriers affect AI implementation in your laboratory? [Ethical concerns (e.g., data privacy, transparency)]'],
        user_inputs['To what extent do the following barriers affect AI implementation in your laboratory? [Staff resistance to change]'],
        user_inputs['To what extent do the following barriers affect AI implementation in your laboratory? [Lack of training and education]'],
        user_inputs['To what extent do the following barriers affect AI implementation in your laboratory? [Regulatory compliance issues]']
    ), unsafe_allow_html=True)
 
   

# Streamlit UI
st.set_page_config(page_title="AI Strategy Recommendation", layout="wide")

st.title("AI Strategy Recommendation System")
st.markdown(
    """
    <style>
    .main { background-color: #f4f4f9; }
    .sidebar .sidebar-content { background-color: #e8e8ef; }
    .stButton>button { border-radius: 5px; background-color: #4CAF50; color: white; }
    </style>
    """, unsafe_allow_html=True
)

st.sidebar.header("Input Parameters")

def user_input_features():
    tabs = st.sidebar.tabs(["General Information", "Challenges"])
    with tabs[0]:
        country = st.selectbox(
            'Country',
            label_encoders['In which country is your laboratory or organization based ?'].inverse_transform(df['In which country is your laboratory or organization based ?'].unique())
        )
        experience = st.selectbox(
            'Years of Experience',
            label_encoders['How many years of experience do you have in your field?'].inverse_transform(df['How many years of experience do you have in your field?'].unique())
        )
        role = st.selectbox(
            'Role in the Organization',
            label_encoders['What is your role in the organization? '].inverse_transform(df['What is your role in the organization? '].unique())
        )
        familiarity = st.selectbox(
            'Familiarity with AI',
            label_encoders['How familiar are you with AI technologies in laboratory operations? '].inverse_transform(df['How familiar are you with AI technologies in laboratory operations? '].unique())
        )
        ai_usage = st.selectbox(
            'Current AI Usage',
            label_encoders['To what extent is AI currently used in your laboratory operations? '].inverse_transform(df['To what extent is AI currently used in your laboratory operations? '].unique())
        )
    with tabs[1]:
        financial_constraints = st.selectbox(
            'Financial Constraints',
            label_encoders['To what extent do the following barriers affect AI implementation in your laboratory? [Financial constraints]'].inverse_transform(df['To what extent do the following barriers affect AI implementation in your laboratory? [Financial constraints]'].unique())
        )
        ethical_concerns = st.selectbox(
            'Ethical Concerns',
            label_encoders['To what extent do the following barriers affect AI implementation in your laboratory? [Ethical concerns (e.g., data privacy, transparency)]'].inverse_transform(df['To what extent do the following barriers affect AI implementation in your laboratory? [Ethical concerns (e.g., data privacy, transparency)]'].unique())
        )
        staff_resistance = st.selectbox(
            'Staff Resistance',
            label_encoders['To what extent do the following barriers affect AI implementation in your laboratory? [Staff resistance to change]'].inverse_transform(df['To what extent do the following barriers affect AI implementation in your laboratory? [Staff resistance to change]'].unique())
        )
        lack_of_training = st.selectbox(
            'Lack of Training',
            label_encoders['To what extent do the following barriers affect AI implementation in your laboratory? [Lack of training and education]'].inverse_transform(df['To what extent do the following barriers affect AI implementation in your laboratory? [Lack of training and education]'].unique())
        )
        regulatory_compliance = st.selectbox(
            'Regulatory Compliance',
            label_encoders['To what extent do the following barriers affect AI implementation in your laboratory? [Regulatory compliance issues]'].inverse_transform(df['To what extent do the following barriers affect AI implementation in your laboratory? [Regulatory compliance issues]'].unique())
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

input_data = user_input_features()
st.subheader("User Input Parameters")
st.json(input_data)

# Display the user inputs
display_user_inputs(input_data)

if st.button("Get Recommendation"):
    recommended_strategy = recommend_strategy(input_data)
    if recommended_strategy:
        st.success(f"Recommended Strategy: {recommended_strategy}")
