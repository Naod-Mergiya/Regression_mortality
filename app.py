import os
import numpy as np
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import urllib.parse
from sqlalchemy import create_engine
from utils.preprocessing import preprocess_data, filter_weeks
from utils.regression_analysis import run_regression
from utils.visualization import plot_coefficients, plot_individual_variable
import warnings
import time  # For timing

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# Set page configuration
st.set_page_config(page_title="Mortality Regression Dashboard", layout="wide")

# Title and description
st.title("Mortality Regression Analysis Dashboard")
st.write("""
This dashboard analyzes a preprocessed mortality dataset using linear regression. The data is loaded from Azure SQL using service principal authentication.
Note: The dataset has a small sample size, is predominantly categorical, and has high dimensionality, which may affect model robustness.
""")

@st.cache_data
def read_doc_mortality():
    """
    Reads the 'doc_mortality.csv' file from the same directory 
    as the current notebook and returns it as a pandas DataFrame.
    """
    # Get the directory of the current notebook
    current_dir = os.getcwd()
    
    # Build the full path to the CSV file
    file_path = os.path.join(current_dir, "doc_mortality.csv")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read and return the CSV as a DataFrame
    df = pd.read_csv(file_path)
    return df

df = read_doc_mortality()

# Apply the drop for docs_received == 0 (as per latest update) - use .copy() to avoid SettingWithCopyWarning
if 'docs_received' in df.columns:
    original_rows = len(df)
    df = df[df['docs_received'] != 0].copy()
    st.info(f"Dropped {original_rows - len(df)} rows where docs_received == 0. New shape: {df.shape}")

# Compute first_week_mortality using .loc to avoid SettingWithCopyWarning
df.loc[:, "first_week_mortality"] = df["doc_dead_1st_week"] / df["docs_received"]
y_col = "first_week_mortality"
x_cols = ["hatchery", "hatcher", "setter", "driver_id", 
          "vehicle_number", "source_of_eggs", "customer_type"]
df = preprocess_data(df, y_col, x_cols)  # Optional if data is already preprocessed

# Debug: Check data
st.write("### Data Overview")
st.write(f"Independent variables: {x_cols}")
st.write(f"NaNs in y: {df[y_col].isna().sum()}")
st.write(f"Infs in y: {np.isinf(df[y_col]).sum()}")
st.write(f"NaNs in X columns:\n{df[x_cols].isna().sum()}")
st.write(f"Data shape: {df.shape}")
if df.empty:
    st.error("DataFrame is empty. Check the source or load_data() function.")
    st.stop()

# Filter weeks using data_preprocessing.py function
df_last_week, df_last_8_weeks = filter_weeks(df)

# Cached regression runner with timing and progress
@st.cache_data
def run_cached_regression(subset, x_cols, y_col, name):
    with st.spinner(f"Fitting {name} model..."):
        start_time = time.time()
        model, _, _ = run_regression(subset, x_cols, y_col, name)
        elapsed = time.time() - start_time
        st.success(f"{name} model fitted in {elapsed:.2f}s")
        return model

# Run regression and collect model summaries
st.write("### Regression Results")
models = {}
summary_data = {}
for subset, name in [(df, "Full_Dataset"), (df_last_week, "Last_Week"), (df_last_8_weeks, "Last_8_Weeks")]:
    if not subset.empty:
        st.write(f"#### {name}")
        model = run_cached_regression(subset, x_cols, y_col, name)
        models[name] = model
        if model is not None:
            # Extract summary statistics
            coefs = model.params[1:]  # Exclude intercept
            pvals = model.pvalues[1:]
            # Create DataFrame for coefficients
            coef_df = pd.DataFrame({
                'Variable': coefs.index,
                'Coefficient': coefs.values,
                'P-Value': pvals.values,
                'Significance': pvals < 0.05
            })
            coef_df = coef_df[coef_df['P-Value'] < 0.05]  # Focus on p < 0.05
            
            # Add global model statistics
            global_stats = pd.DataFrame({
                'Variable': ['R-squared', 'F-value', 'F-p-value'],
                'Value': [model.rsquared, model.fvalue, model.f_pvalue],
                'P-Value': [None, None, None],
                'Significance': [None, None, None]
            })
            
            # Combine coefficient and global stats
            summary_df = pd.concat([coef_df, global_stats]).reset_index(drop=True)
            summary_data[name] = summary_df
            summary_df = summary_df.fillna({'Coefficient': 0.0, 'Value': 0.0})  # Safe fill for formatting
            st.table(summary_df.style.format({'Coefficient': '{:.4f}', 'P-Value': '{:.4f}', 'Value': '{:.4f}'}))  # Display table
            fig = plot_coefficients(model, None, name)  # Visualization
            st.pyplot(fig)
        else:
            st.write(f"No valid model generated for {name}. Check data or regression.")
    else:
        st.write(f"No data available for {name}")

# Additional visualization: R-squared comparison
if summary_data:
    r_squared_data = {name: models[name].rsquared for name, model in models.items() if model is not None}
    if r_squared_data:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(r_squared_data.keys(), r_squared_data.values(), color='skyblue')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('R-squared')
        ax.set_title('R-squared Comparison Across Datasets')
        for i, v in enumerate(r_squared_data.values()):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
        st.pyplot(fig)

# Individual variable visualization
st.write("### Hatchery Distribution")
if not df_last_week.empty and not df_last_8_weeks.empty:
    fig = plot_individual_variable(df_last_week, df_last_8_weeks, 'hatchery', y_col)
    st.pyplot(fig)
else:
    st.write("Insufficient data for hatchery distribution plots.")