import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr, f_oneway, chi2
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import tempfile
import os
import plotly.express as px
from sklearn.cluster import KMeans
from prince import MCA
from factor_analyzer import FactorAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.utils import resample

# Load the dataset
file_path = "Survey.csv"
df = pd.read_csv(file_path)

# Columns to exclude
exclude_columns = ["Username", "Email Address", "Timestamp"]

# Preprocess the data
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
        chi2_stat, p, dof, ex = chi2_contingency(contingency_table)
        st.write(f"Chi-Squared Statistic: {chi2_stat}")
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
        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
        st.write(f"Cramér's V: {cramers_v}")
        if cramers_v > 0.25:
            st.write("Strong association.")
        elif cramers_v > 0.15:
            st.write("Moderate association.")
        else:
            st.write("Weak association.")

    # Likelihood Ratio Test
    st.subheader("Likelihood Ratio Test")
    st.write("Compares the goodness of fit of two nested models.")
    if st.button("Run Likelihood Ratio Test"):
        # Example: Compare two contingency tables
        contingency_table1 = pd.crosstab(df[categorical_columns[0]], df[categorical_columns[1]])
        contingency_table2 = pd.crosstab(df[categorical_columns[0]], df[categorical_columns[2]])
        
        # Calculate likelihood ratio test statistic
        chi2_stat1, _, _, _ = chi2_contingency(contingency_table1)
        chi2_stat2, _, _, _ = chi2_contingency(contingency_table2)
        lr_stat = -2 * (chi2_stat1 - chi2_stat2)
        p_value = chi2.sf(lr_stat, df=1)  # Degrees of freedom = 1 for nested models
        st.write(f"Likelihood Ratio Statistic: {lr_stat}")
        st.write(f"p-value: {p_value}")
        if p_value < 0.05:
            st.write("The models are significantly different.")
        else:
            st.write("No significant difference between the models.")

    # Permutation Tests
    st.subheader("Permutation Tests")
    st.write("Assesses the significance of relationships by randomly shuffling data.")
    
    # Column selection for permutation test
    col1, col2 = st.columns(2)
    with col1:
        perm_col1 = st.selectbox("Select the first column for permutation test:", categorical_columns)
    with col2:
        perm_col2 = st.selectbox("Select the second column for permutation test:", categorical_columns)

    if st.button("Run Permutation Test"):
        if perm_col1 and perm_col2:
            # Encode categorical data
            le = LabelEncoder()
            df_encoded = df.copy()
            df_encoded[perm_col1] = le.fit_transform(df_encoded[perm_col1].astype(str))
            df_encoded[perm_col2] = le.fit_transform(df_encoded[perm_col2].astype(str))

            # Calculate observed correlation
            observed_corr, _ = spearmanr(df_encoded[perm_col1], df_encoded[perm_col2])

            # Permutation test
            n_permutations = 1000
            perm_corrs = []
            for _ in range(n_permutations):
                perm_col2_shuffled = resample(df_encoded[perm_col2], replace=False, random_state=42)
                perm_corr, _ = spearmanr(df_encoded[perm_col1], perm_col2_shuffled)
                perm_corrs.append(perm_corr)

            # Calculate p-value
            p_value = (np.sum(np.abs(perm_corrs) >= np.abs(observed_corr)) + 1) / (n_permutations + 1)
            st.write(f"Observed Correlation: {observed_corr}")
            st.write(f"p-value: {p_value}")
            if p_value < 0.05:
                st.write("The correlation is statistically significant.")
            else:
                st.write("No significant correlation found.")
        else:
            st.write("Please select both columns before running the test.")

    # Latent Class Analysis (LCA)
st.subheader("Latent Class Analysis (LCA)")
st.write("LCA identifies hidden subgroups within categorical data.")
if st.button("Run Latent Class Analysis"):
    # Encode categorical data for LCA
    df_encoded = df.copy()
    for col in categorical_columns:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

    # Perform LCA
    n_components = 3  # Number of latent classes
    lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
    lda.fit(df_encoded)
    st.write("LCA Components:")
    st.write(lda.components_)

    # Calculate BIC
    st.subheader("Bayesian Information Criterion (BIC)")
    log_likelihood = lda.score(df_encoded)
    bic = -2 * log_likelihood + n_components * np.log(df_encoded.shape[0])
    st.write(f"BIC: {bic}")

    # Scatterplots
    st.subheader("Scatterplots for Latent Classes")
    reduced_features = df_encoded.iloc[:, :2].values  # Use first two features for simplicity
    lda_transformed = lda.transform(df_encoded)
    latent_classes = np.argmax(lda_transformed, axis=1)
    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c=latent_classes, cmap="viridis", alpha=0.7)
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)
    ax.set_title("Scatterplot of Latent Classes")
    st.pyplot(fig)

    # Heatmaps
    st.subheader("Heatmaps for Latent Classes")
    class_means = pd.DataFrame(lda.components_).T
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(class_means, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Heatmap of Latent Class Means")
    st.pyplot(fig)

    # Profile Plots
    st.subheader("Profile Plots")
    class_means = pd.DataFrame(lda.components_).T
    fig, ax = plt.subplots(figsize=(10, 6))
    for class_idx in range(n_components):
        ax.plot(class_means.index, class_means[class_idx], label=f"Class {class_idx + 1}")
    ax.set_title("Profile Plot of Latent Classes")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Mean Value")
    ax.legend()
    st.pyplot(fig)


    # Structural Equation Modeling (SEM) for Categorical Data
    st.subheader("Structural Equation Modeling (SEM)")
    st.write("SEM models relationships between observed and latent categorical variables.")
    if st.button("Run SEM"):
        # Encode categorical data for SEM
        df_encoded = df.copy()
        for col in categorical_columns:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

        # Perform SEM using FactorAnalyzer
        fa = FactorAnalyzer(n_factors=3, rotation='varimax')  # Adjust n_factors as needed
        fa.fit(df_encoded)
        st.write("Factor Loadings:")
        st.write(fa.loadings_)

    # Cluster Analysis
    st.subheader("Cluster Analysis")
    st.write("Cluster Analysis groups similar cases or variables, especially useful for categorical data.")
    if st.button("Run Cluster Analysis"):
        # Encode categorical data for clustering
        df_encoded = df.copy()
        for col in categorical_columns:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

        # Perform KMeans clustering
        n_clusters = 3  # Number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_encoded['Cluster'] = kmeans.fit_predict(df_encoded)
        st.write("Cluster Assignments:")
        st.write(df_encoded['Cluster'].value_counts())

    # Multiple Correspondence Analysis (MCA)
    st.subheader("Multiple Correspondence Analysis (MCA)")
    st.write("MCA explores relationships between multiple categorical variables.")
    if st.button("Run MCA"):
        # Perform MCA
        mca = MCA(n_components=2)  # Adjust n_components as needed
        mca.fit(df[categorical_columns])
        st.write("MCA Eigenvalues:")
        st.write(mca.eigenvalues_)
        st.write("MCA Row Coordinates:")
        st.write(mca.row_coordinates(df[categorical_columns]))

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
