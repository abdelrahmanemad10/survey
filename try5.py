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
from pingouin import cronbach_alpha
import base64
from datetime import datetime
from io import BytesIO

# Load the dataset
@st.cache
def load_data(file_path):
    return pd.read_csv("Survey.csv")

# Preprocess the data
def preprocess_data(df, exclude_columns):
    try:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %I:%M:%S %p %Z", errors="coerce")
    except:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")  # Fallback to individual parsing

    df.drop(columns=exclude_columns, inplace=True, errors='ignore')
    categorical_columns = [col for col in df.columns if df[col].dtype == "object"]
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    return df, categorical_columns, numerical_columns

# Function to calculate Cronbach's Alpha
def calculate_cronbach_alpha(data):
    """Calculate Cronbach's Alpha for a DataFrame."""
    k = data.shape[1]
    if k < 2:
        return None  # Cronbach's Alpha requires at least 2 columns
    item_variances = data.var(axis=0, ddof=1)
    total_variance = data.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return None  # Avoid division by zero
    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    return alpha

# Function to calculate AVE (Average Variance Extracted)
def calculate_ave(data):
    """Calculate AVE for a DataFrame."""
    if data.shape[1] < 2:
        return None  # AVE requires at least 2 columns
    fa = FactorAnalyzer(n_factors=1, rotation=None)
    fa.fit(data)
    loadings = fa.loadings_
    ave = np.mean(loadings**2)
    return ave

# Function to perform sensitivity analysis
def sensitivity_analysis(df, selected_columns):
    """Analyze the impact of excluding each column on overall metrics."""
    overall_alpha = calculate_cronbach_alpha(df[selected_columns])
    overall_ave = calculate_ave(df[selected_columns])

    results = []
    for col in selected_columns:
        excluded_columns = [c for c in selected_columns if c != col]
        alpha_excluded = calculate_cronbach_alpha(df[excluded_columns])
        ave_excluded = calculate_ave(df[excluded_columns])
        alpha_impact = abs(overall_alpha - alpha_excluded) if overall_alpha and alpha_excluded else None
        ave_impact = abs(overall_ave - ave_excluded) if overall_ave and ave_excluded else None
        results.append({
            "Column": col,
            "Alpha Impact": alpha_impact,
            "AVE Impact": ave_impact
        })

    return pd.DataFrame(results)

# Function to encode categorical data
def encode_data(df):
    """Encode categorical data using LabelEncoder."""
    encoded_df = df.copy()
    for col in encoded_df.columns:
        if encoded_df[col].dtype == "object":
            encoded_df[col] = LabelEncoder().fit_transform(encoded_df[col].astype(str))
    return encoded_df

# Function to export results to Excel
def export_to_excel(results):
    """Export results to an Excel file."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        results.to_excel(writer, index=False, sheet_name="Results")
    output.seek(0)
    return output

# Streamlit UI for Reliability and Validity Analysis
def reliability_and_validity_analysis(df):
    st.title("Reliability and Validity Analysis")

    # Encode categorical data
    encoded_df = encode_data(df)

    # Initialize session state for column selection
    if "selected_columns" not in st.session_state:
        st.session_state.selected_columns = list(encoded_df.columns)

    # Display and allow column selection
    st.sidebar.header("Select Columns for Analysis")
    selected_columns = st.sidebar.multiselect(
        "Choose columns:",
        options=list(encoded_df.columns),
        default=st.session_state.selected_columns
    )
    st.session_state.selected_columns = selected_columns

    # Initialize results DataFrame
    results = pd.DataFrame(columns=["Column", "Cronbach's Alpha", "AVE"])

    # Calculate Cronbach's Alpha and AVE for selected columns
    for column in selected_columns:
        alpha = calculate_cronbach_alpha(encoded_df[[column]])
        ave = calculate_ave(encoded_df[[column]])
        alpha = "Not applicable" if alpha is None else alpha
        ave = "Not applicable" if ave is None else ave
        new_row = pd.DataFrame({"Column": [column], "Cronbach's Alpha": [alpha], "AVE": [ave]})
        results = pd.concat([results, new_row], ignore_index=True)

    # Calculate overall Cronbach's Alpha and AVE for selected columns
    if len(selected_columns) > 1:
        overall_alpha = calculate_cronbach_alpha(encoded_df[selected_columns])
        overall_ave = calculate_ave(encoded_df[selected_columns])
    else:
        overall_alpha, overall_ave = None, None

    overall_alpha = "Not applicable" if overall_alpha is None else overall_alpha
    overall_ave = "Not applicable" if overall_ave is None else overall_ave
    new_row = pd.DataFrame({"Column": ["Overall"], "Cronbach's Alpha": [overall_alpha], "AVE": [overall_ave]})
    results = pd.concat([results, new_row], ignore_index=True)

    # Display results
    st.subheader("Reliability and Validity Results")
    st.write(results)

    # Perform sensitivity analysis if more than one column is selected
    if len(selected_columns) > 1:
        st.subheader("Column Sensitivity Analysis")
        sensitivity_results = sensitivity_analysis(encoded_df, selected_columns)
        st.write(sensitivity_results)

    # Export results to Excel
    st.subheader("Export Results")
    if st.button("Export to Excel"):
        excel_file = export_to_excel(results)
        b64 = base64.b64encode(excel_file.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="reliability_validity_results.xlsx">Download Excel File</a>'
        st.markdown(href, unsafe_allow_html=True)

# Streamlit app
def main():
    st.title("Survey Data Analysis")

    # Load and preprocess data
    file_path = "Survey.csv"
    exclude_columns = ["Username", "Email Address", "Timestamp"]
    df = load_data(file_path)
    df, categorical_columns, numerical_columns = preprocess_data(df, exclude_columns)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio(
        "Select an option:",
        ["Overview", "Data Validation", "Visualizations", "Advanced Statistical Analysis", "Reliability and Validity Analysis"]
    )

    # Initialize a list to store plot images
    plot_images = []

    if options == "Overview":
        show_overview(df)
    elif options == "Data Validation":
        show_data_validation(df, categorical_columns, numerical_columns)
    elif options == "Visualizations":
        show_visualizations(df, categorical_columns, plot_images)
    elif options == "Advanced Statistical Analysis":
        show_advanced_statistical_analysis(df, categorical_columns, plot_images)
    elif options == "Reliability and Validity Analysis":
        reliability_and_validity_analysis(df)

    # Export cleaned data and analysis results to PDF
    st.sidebar.header("Export")
    if st.sidebar.button("Export Cleaned Data and Analysis"):
        export_data_and_analysis(df, plot_images)

# Function to display dataset overview
def show_overview(df):
    st.header("Dataset Overview")
    st.write(df.head())
    st.write("Shape of the dataset:", df.shape)
    st.write("Summary statistics:")
    st.write(df.describe(include="all"))

# Function to display data validation
def show_data_validation(df, categorical_columns, numerical_columns):
    st.header("Data Validation")
    st.write("This section checks for common data quality issues and provides user controls for validation.")

    # User controls for validation
    st.sidebar.header("Validation Controls")
    check_missing_values = st.sidebar.checkbox("Check for Missing Values", value=True)
    check_duplicates = st.sidebar.checkbox("Check for Duplicate Rows", value=True)
    check_data_types = st.sidebar.checkbox("Check Data Types", value=True)
    check_outliers = st.sidebar.checkbox("Check for Outliers", value=True)
    check_unique_values = st.sidebar.checkbox("Check Unique Values in Categorical Columns", value=True)
    check_value_ranges = st.sidebar.checkbox("Check Value Ranges for Numerical Columns", value=True)
    check_invalid_categories = st.sidebar.checkbox("Check for Invalid Categories in Categorical Columns", value=True)

    if check_missing_values:
        check_missing_values_func(df)
    if check_duplicates:
        check_duplicates_func(df)
    if check_data_types:
        check_data_types_func(df)
    if check_outliers:
        check_outliers_func(df, numerical_columns)
    if check_unique_values:
        check_unique_values_func(df, categorical_columns)
    if check_value_ranges:
        check_value_ranges_func(df, numerical_columns)
    if check_invalid_categories:
        check_invalid_categories_func(df, categorical_columns)

# Function to check for missing values
def check_missing_values_func(df):
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        st.success("No missing values found in the dataset.")
    else:
        st.warning("Missing values found in the following columns:")
        st.write(missing_values[missing_values > 0])
        st.write("Consider handling missing values before proceeding with analysis.")

# Function to check for duplicates
def check_duplicates_func(df):
    st.subheader("Duplicate Rows")
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows == 0:
        st.success("No duplicate rows found in the dataset.")
    else:
        st.warning(f"Found {duplicate_rows} duplicate rows.")
        st.write("Consider removing duplicates before proceeding with analysis.")

# Function to check for inconsistent data types
def check_data_types_func(df):
    st.subheader("Data Types")
    data_types = df.dtypes
    st.write("Data types of each column:")
    st.write(data_types)
    inconsistent_columns = [col for col in df.columns if df[col].dtype == "object" and df[col].str.contains("[^a-zA-Z0-9 ]").any()]
    if inconsistent_columns:
        st.warning("The following columns contain non-alphanumeric characters:")
        st.write(inconsistent_columns)
    else:
        st.success("All columns have consistent data types.")

# Function to check for outliers in numerical columns
def check_outliers_func(df, numerical_columns):
    st.subheader("Outliers in Numerical Columns")
    if len(numerical_columns) > 0:
        for col in numerical_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if outliers.shape[0] > 0:
                st.warning(f"Outliers found in column '{col}':")
                st.write(outliers)
            else:
                st.success(f"No outliers found in column '{col}'.")
    else:
        st.info("No numerical columns found for outlier detection.")

# Function to check for unique values in categorical columns
def check_unique_values_func(df, categorical_columns):
    st.subheader("Unique Values in Categorical Columns")
    for col in categorical_columns:
        unique_values = df[col].nunique()
        st.write(f"Column '{col}' has {unique_values} unique values.")
        if unique_values > 20:
            st.warning(f"Column '{col}' has a high number of unique values ({unique_values}). Consider grouping or binning.")

# Function to check value ranges for numerical columns
def check_value_ranges_func(df, numerical_columns):
    st.subheader("Value Ranges for Numerical Columns")
    if len(numerical_columns) > 0:
        for col in numerical_columns:
            min_value = df[col].min()
            max_value = df[col].max()
            st.write(f"Column '{col}' has a range of [{min_value}, {max_value}].")
            if min_value < 0 or max_value > 100:  # Example range check
                st.warning(f"Column '{col}' has values outside the expected range.")
    else:
        st.info("No numerical columns found for value range checks.")

# Function to check for invalid categories in categorical columns
def check_invalid_categories_func(df, categorical_columns):
    st.subheader("Invalid Categories in Categorical Columns")
    for col in categorical_columns:
        valid_categories = st.text_input(f"Enter valid categories for '{col}' (comma-separated):")
        if valid_categories:
            valid_categories = [cat.strip() for cat in valid_categories.split(",")]
            invalid_categories = df[~df[col].isin(valid_categories)][col].unique()
            if len(invalid_categories) > 0:
                st.warning(f"Invalid categories found in column '{col}':")
                st.write(invalid_categories)
            else:
                st.success(f"No invalid categories found in column '{col}'.")

# Function to display visualizations
def show_visualizations(df, categorical_columns, plot_images):
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

# Function to display advanced statistical analysis
def show_advanced_statistical_analysis(df, categorical_columns, plot_images):
    st.header("Advanced Statistical Analysis")

    # Reference Table for Statistical Tests and Metrics
    st.subheader("Reference Table for Statistical Tests and Metrics")
    reference_data = {
        "Test/Metric": [
            "Cronbach's Alpha",
            "Average Variance Extracted (AVE)",
            "Cramér's V",
            "p-value",
            "Likelihood Ratio Test (p-value)",
            "Permutation Test (p-value)"
        ],
        "Threshold/Interpretation": [
            "> 0.7: Good internal consistency",
            "> 0.5: Acceptable convergent validity",
            "0.0-0.15: Weak association, 0.15-0.25: Moderate association, > 0.25: Strong association",
            "< 0.05: Statistically significant",
            "< 0.05: Models are significantly different",
            "< 0.05: Correlation is statistically significant"
        ]
    }
    reference_df = pd.DataFrame(reference_data)
    st.table(reference_df)

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
    
    # Add a title page
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Survey Data Analysis Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(20)
    pdf.cell(200, 10, txt="Prepared by: [Dr. Mohamed Helmy]", ln=True, align="C")
    
    # Add a table of contents
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Table of Contents", ln=True, align="L")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, txt="1. Overview", ln=True, align="L")
    pdf.cell(200, 10, txt="2. Data Validation", ln=True, align="L")
    pdf.cell(200, 10, txt="3. Visualizations", ln=True, align="L")
    pdf.cell(200, 10, txt="4. Advanced Statistical Analysis", ln=True, align="L")
    
    # Add overview section
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="1. Overview", ln=True, align="L")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    
    # Add dataset shape
    pdf.multi_cell(0, 10, txt=f"Shape of the dataset: {df.shape}")
    pdf.ln(5)
    
    # Add summary statistics
    pdf.multi_cell(0, 10, txt="Summary statistics:")
    pdf.ln(5)
    
    # Format summary statistics as a table
    summary_stats = df.describe(include="all").T  # Transpose for better readability
    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 10, txt="Column", border=1)
    pdf.cell(30, 10, txt="Count", border=1)
    pdf.cell(30, 10, txt="Mean", border=1)
    pdf.cell(30, 10, txt="Std Dev", border=1)
    pdf.cell(30, 10, txt="Min", border=1)
    pdf.cell(30, 10, txt="25%", border=1)
    pdf.cell(30, 10, txt="50%", border=1)
    pdf.cell(30, 10, txt="75%", border=1)
    pdf.cell(30, 10, txt="Max", border=1)
    pdf.ln()
    
    pdf.set_font("Arial", "", 12)
    for index, row in summary_stats.iterrows():
        # Replace or remove unsupported Unicode characters in the column name
        index_cleaned = index.encode("latin-1", errors="replace").decode("latin-1")
        
        pdf.cell(40, 10, txt=index_cleaned, border=1)
        
        # Handle 'count'
        count_value = row['count'] if 'count' in row and pd.notna(row['count']) else "N/A"
        pdf.cell(30, 10, txt=str(count_value), border=1)
        
        # Handle 'mean'
        mean_value = row['mean'] if 'mean' in row and pd.notna(row['mean']) else "N/A"
        pdf.cell(30, 10, txt=str(mean_value), border=1)
        
        # Handle 'std'
        std_value = row['std'] if 'std' in row and pd.notna(row['std']) else "N/A"
        pdf.cell(30, 10, txt=str(std_value), border=1)
        
        # Handle 'min'
        min_value = row['min'] if 'min' in row and pd.notna(row['min']) else "N/A"
        pdf.cell(30, 10, txt=str(min_value), border=1)
        
        # Handle '25%'
        p25_value = row['25%'] if '25%' in row and pd.notna(row['25%']) else "N/A"
        pdf.cell(30, 10, txt=str(p25_value), border=1)
        
        # Handle '50%'
        p50_value = row['50%'] if '50%' in row and pd.notna(row['50%']) else "N/A"
        pdf.cell(30, 10, txt=str(p50_value), border=1)
        
        # Handle '75%'
        p75_value = row['75%'] if '75%' in row and pd.notna(row['75%']) else "N/A"
        pdf.cell(30, 10, txt=str(p75_value), border=1)
        
        # Handle 'max'
        max_value = row['max'] if 'max' in row and pd.notna(row['max']) else "N/A"
        pdf.cell(30, 10, txt=str(max_value), border=1)
        
        pdf.ln()
    
    # Add visualizations section
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="3. Visualizations", ln=True, align="L")
    for img_path in plot_images:
        pdf.ln(10)
        pdf.image(img_path, x=10, w=180)
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Figure: {os.path.basename(img_path)}", ln=True, align="L")
    
    # Save the PDF
    pdf.output(output_path)

# Function to create a download link for the PDF
def create_download_link(pdf_path, filename):
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download PDF Report</a>'
    return href

# Function to export cleaned data and analysis results
def export_data_and_analysis(df, plot_images):
    cleaned_file_path = "cleaned_survey_data.csv"
    df.to_csv(cleaned_file_path, index=False)
    
    # Create PDF report
    pdf_output_path = "survey_analysis_report.pdf"
    create_pdf_report(plot_images, pdf_output_path)
    
    # Provide download link for the PDF
    st.sidebar.success(f"Cleaned data saved as {cleaned_file_path}")
    st.sidebar.markdown(create_download_link(pdf_output_path, "survey_analysis_report.pdf"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
