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
  # Update path as needed (e.g., "d:\Regression\processed_data.csv")
df = load_data()
y_col = "first_week_mortality"
x_cols = ["hatchery", "hatcher", "setter", "DriverName", 
          "VehicleNumber", "source_of_eggs", "CustomerType"]
df = preprocess_data(df, y_col, x_cols)  # Optional if data is already preprocessed

st.write("### Data Overview")
st.write(f"Independent variables: {x_cols}")
st.write(f"NaNs in y: {df[y_col].isna().sum()}")
st.write(f"NaNs in X columns:\n{df[x_cols].isna().sum()}")
st.write(f"Data shape: {df.shape}")

# Filter weeks using data_preprocessing.py function
df_last_week, df_last_8_weeks = filter_weeks(df)

# Run regression and display results
st.write("### Regression Results")
models = {}
for subset, name in [(df, "Full_Dataset"), (df_last_week, "Last_Week"), (df_last_8_weeks, "Last_8_Weeks")]:
    if not subset.empty:
        st.write(f"#### {name}")
        model, _, _ = run_regression(subset, x_cols, y_col, name)
        models[name] = model
        if model is not None:
            fig = plt.figure(figsize=(10, 8))
            plot_coefficients(model, None, name)
            st.pyplot(fig)
    else:
        st.write(f"No data available for {name}")

# Individual variable visualization
st.write("### Hatchery Distribution")
if not df_last_week.empty and not df_last_8_weeks.empty:
    fig = plt.figure(figsize=(12, 6))
    plot_individual_variable(df_last_week, df_last_8_weeks, 'hatchery', y_col)
    st.pyplot(fig)
else:
    st.write("Insufficient data for hatchery distribution plots.")