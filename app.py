import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocessing import load_data, preprocess_data, filter_weeks
from utils.regression_analysis import run_regression
from utils.visualization import plot_coefficients, plot_individual_variable

# Set page configuration
st.set_page_config(page_title="Mortality Regression Dashboard", layout="wide")

# Title and description
st.title("Mortality Regression Analysis Dashboard")
st.write("""
This dashboard analyzes a preprocessed mortality dataset using linear regression. The data is loaded and transformed 
using functions from data_preprocessing.py. Note: The dataset has a small sample size, is predominantly categorical, 
and has high dimensionality, which may affect model robustness.
""")

# Load and preprocess data using data_preprocessing.py functions
df = load_data()  # Update default path in load_data if needed (e.g., "d:\Regression\processed_data.csv")
y_col = "first_week_mortality"
x_cols = ["hatchery", "hatcher", "setter", "DriverName", 
          "VehicleNumber", "source_of_eggs", "CustomerType"]
df = preprocess_data(df, y_col, x_cols)  # Optional if data is already preprocessed

# Debug: Check data
st.write("### Data Overview")
st.write(f"Independent variables: {x_cols}")
st.write(f"NaNs in y: {df[y_col].isna().sum()}")
st.write(f"NaNs in X columns:\n{df[x_cols].isna().sum()}")
st.write(f"Data shape: {df.shape}")
if df.empty:
    st.error("DataFrame is empty. Check the source or load_data() function.")
    st.stop()

# Filter weeks using data_preprocessing.py function
df_last_week, df_last_8_weeks = filter_weeks(df)

# Run regression and collect model summaries
st.write("### Regression Results")
models = {}
summary_data = {}
for subset, name in [(df, "Full_Dataset"), (df_last_week, "Last_Week"), (df_last_8_weeks, "Last_8_Weeks")]:
    if not subset.empty:
        st.write(f"#### {name}")
        model, _, _ = run_regression(subset, x_cols, y_col, name)
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
            coef_df = coef_df[coef_df['P-Value'] < 0.1]  # Focus on p < 0.1
            
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