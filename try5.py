import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr, f_oneway
import numpy as np
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import tempfile
import os
import plotly.express as px

# Load the dataset
file_path = "Survey.csv"
df = pd.read_csv(file_path)

# Columns to exclude
exclude_columns = ["Username", "Email Address", "Timestamp"]

# Preprocess the data
# Attempt to parse the Timestamp column with a specified format, if possible
try:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %I:%M:%S %p %Z", errors="coerce")
except:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")  # Fallback to individual parsing

# Drop the excluded columns
df.drop(columns=exclude_columns, inplace=True, errors='ignore')

categorical_columns = [col for col in df.columns if df[col].dtype == "object"]

# Streamlit app
st.title("Survey Data Analysis")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio(
    "Select an option:",
    ["Overview", "Visualizations", "Advanced Statistical Analysis"]
)

# Initialize a list to store plot images
plot_images = []

if options == "Overview":
    st.header("Dataset Overview")
    st.write(df.head())
    st.write("Shape of the dataset:", df.shape)
    st.write("Summary statistics:")
    st.write(df.describe(include="all"))

elif options == "Visualizations":
    st.header("Visualizations")

    # Column selection for visualization
    col1, col2 = st.columns(2)
    with col1:
        selected_column1 = st.selectbox("Select the first categorical column for visualization:", categorical_columns)
    with col2:
        selected_column2 = st.selectbox("Select the second categorical column for visualization:", categorical_columns)

    if selected_column1 and selected_column2:
        st.subheader(f"Relationship between {selected_column1} and {selected_column2}")
        fig, ax = plt.subplots()
        sns.countplot(x=selected_column1, hue=selected_column2, data=df, ax=ax)
        ax.set_title(f"Relationship between {selected_column1} and {selected_column2}")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
        
        # Save the plot as an image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name)
            plot_images.append(tmpfile.name)

    # Pie Chart for Countries
    st.subheader("Country Distribution")
    country_column = "In which country is your laboratory or organization based ?"
    if country_column in df.columns:
        country_counts = df[country_column].value_counts()
        fig_country = px.pie(values=country_counts, names=country_counts.index, title='Distribution of Responses by Country')
        st.plotly_chart(fig_country)
        
        # Save the plot as an image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig_country.write_image(tmpfile.name)
            plot_images.append(tmpfile.name)

    # Histogram for Strategies and Challenges
    st.subheader("Histogram for Strategies and Challenges")
    strategy_column = "Which strategies have been implemented in your laboratory to overcome AI challenges? (Select all that apply) "
    challenge_column = "To what extent do the following barriers affect AI implementation in your laboratory? [Financial constraints]"
    if strategy_column in df.columns and challenge_column in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=strategy_column, hue=challenge_column, multiple="stack", ax=ax)
        ax.set_title(f"Histogram of {strategy_column} and {challenge_column}")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
        
        # Save the plot as an image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name)
            plot_images.append(tmpfile.name)

elif options == "Advanced Statistical Analysis":
    st.header("Advanced Statistical Analysis")

    # Chi-Squared Test
    st.subheader("Chi-Squared Test for Independence")
    col1, col2 = st.columns(2)
    with col1:
        challenge = st.selectbox("Select the first categorical column:", categorical_columns)
    with col2:
        strategy = st.selectbox("Select the second categorical column:", categorical_columns)

    if st.button("Run Chi-Squared Test"):
        contingency_table = pd.crosstab(df[challenge], df[strategy])
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        st.write(f"Chi-Squared Statistic: {chi2}")
        st.write(f"p-value: {p}")
        st.write(f"Degrees of Freedom: {dof}")
        if p < 0.05:
            st.write("There is a significant association between the selected columns.")
        else:
            st.write("No significant association found.")

        # Heatmap of the contingency table
        st.subheader("Heatmap of Contingency Table")
        fig, ax = plt.subplots()
        sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
        
        # Save the plot as an image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name)
            plot_images.append(tmpfile.name)

        # Cramér's V
        st.subheader("Cramér's V for Association Strength")
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
        st.write(f"Cramér's V: {cramers_v}")
        if cramers_v > 0.25:
            st.write("Strong association.")
        elif cramers_v > 0.15:
            st.write("Moderate association.")
        else:
            st.write("Weak association.")

    # ANOVA (Analysis of Variance)
    st.subheader("ANOVA (Analysis of Variance)")
    col1, col2 = st.columns(2)
    with col1:
        group_col = st.selectbox("Select the categorical column for grouping:", categorical_columns)
    with col2:
        value_col = st.selectbox("Select the numerical column for analysis:", df.select_dtypes(include=np.number).columns)

    if st.button("Run ANOVA"):
        groups = [df[value_col][df[group_col] == group] for group in df[group_col].unique()]
        f_stat, p_val = f_oneway(*groups)
        st.write(f"F-Statistic: {f_stat}")
        st.write(f"p-value: {p_val}")
        if p_val < 0.05:
            st.write("There is a significant difference between the groups.")
        else:
            st.write("No significant difference between the groups.")

    # Correlation Analysis for Categorical Data
    st.subheader("Correlation Analysis for Categorical Data")
    col1, col2 = st.columns(2)
    with col1:
        col_x = st.selectbox("Select the first column for correlation:", categorical_columns)
    with col2:
        col_y = st.selectbox("Select the second column for correlation:", categorical_columns)

    if st.button("Run Correlation Analysis"):
        le = LabelEncoder()
        df_encoded = df.copy()
        df_encoded[col_x] = le.fit_transform(df_encoded[col_x].astype(str))
        df_encoded[col_y] = le.fit_transform(df_encoded[col_y].astype(str))
        corr, p = spearmanr(df_encoded[col_x], df_encoded[col_y])
        st.write(f"Spearman Correlation: {corr}")
        st.write(f"p-value: {p}")
        if p < 0.05:
            st.write("Significant correlation found.")
        else:
            st.write("No significant correlation found.")

        # Scatter plot for correlation
        st.subheader("Scatter Plot for Correlation")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df_encoded[col_x], y=df_encoded[col_y], ax=ax)
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.set_title(f"Scatter Plot of {col_x} vs {col_y}")
        st.pyplot(fig)
        
        # Save the plot as an image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name)
            plot_images.append(tmpfile.name)

# Function to create a PDF report
def create_pdf_report(plot_images, output_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.cell(200, 10, txt="Survey Data Analysis Report", ln=True, align="C")
    
    # Add overview section
    pdf.ln(10)
    pdf.cell(200, 10, txt="Overview", ln=True, align="L")
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=f"Shape of the dataset: {df.shape}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt="Summary statistics:")
    pdf.ln(5)
    summary_stats = df.describe(include="all").to_string()
    pdf.multi_cell(0, 10, txt=summary_stats.encode('latin-1', 'replace').decode('latin-1'))
    
    # Add visualizations section
    pdf.add_page()
    pdf.cell(200, 10, txt="Visualizations", ln=True, align="L")
    for img_path in plot_images:
        pdf.ln(10)
        pdf.image(img_path, x=10, w=180)
    
    # Save the PDF
    pdf.output(output_path)

# Export cleaned data and analysis results to PDF
st.sidebar.header("Export")
if st.sidebar.button("Export Cleaned Data and Analysis"):
    cleaned_file_path = "cleaned_survey_data.csv"
    df.to_csv(cleaned_file_path, index=False)
    
    # Create PDF report
    pdf_output_path = "survey_analysis_report.pdf"
    create_pdf_report(plot_images, pdf_output_path)
    
    st.sidebar.success(f"Cleaned data saved as {cleaned_file_path}")
    st.sidebar.success(f"Analysis report saved as {pdf_output_path}")