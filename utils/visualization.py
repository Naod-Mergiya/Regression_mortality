import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_coefficients(model, feature_names, subset_name):
    """
    Plot significant coefficients from the regression model.
    
    Args:
        model: Fitted statsmodels OLS model
        feature_names: List of feature names from preprocessor (optional)
        subset_name (str): Name of the dataset subset (for title)
    """
    if model is None:
        print(f"No model available for {subset_name}, skipping visualization")
        return
    
    # Extract coefficients and p-values
    coefs = model.params[1:]  # Skip const
    pvals = model.pvalues[1:]
    
    # Filter significant (p<0.05) and sort by abs(coef)
    sig_mask = pvals < 0.05
    sig_coefs = coefs[sig_mask]
    sig_names = pd.Series(model.params.index[1:][sig_mask])  # convert Index → Series
    sig_abs = np.abs(sig_coefs)
    
    if sig_names.empty:
        print(f"No significant coefficients (p<0.05) found for {subset_name}")
        return
    
    sorted_idx = np.argsort(sig_abs)[::-1][:15]  # Top 15
    
    # Use iloc safely
    top_names = sig_names.iloc[sorted_idx]
    top_abs   = sig_abs.iloc[sorted_idx]
    top_coefs = sig_coefs.iloc[sorted_idx]
    
    # Plot horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['red' if c > 0 else 'blue' for c in top_coefs]
    bars = ax.barh(top_names, top_abs, color=colors, alpha=0.7)
    ax.set_xlabel('Absolute Impact on First-Week Mortality (|Coefficient|)')
    ax.set_title(f'Top Drivers of First-Week Mortality ({subset_name})')
    ax.invert_yaxis()  # Largest on top
    
    # Add value labels
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.0001,
                bar.get_y() + bar.get_height()/2,
                f'{top_abs.iloc[i]:.4f}', 
                va='center')
    
    plt.tight_layout()
    plt.show()

def plot_individual_variable(df_last_week, df_last_8_weeks, variable, y_col):
    """
    Plot box plots of y_col by variable categories for last week and last 8 weeks.
    
    Args:
        df_last_week (pd.DataFrame): Data for last week
        df_last_8_weeks (pd.DataFrame): Data for last 8 weeks
        variable (str): Independent variable to plot
        y_col (str): Target column name
    """
    if variable not in df_last_week.columns or variable not in df_last_8_weeks.columns:
        print(f"Variable {variable} not found in the dataset")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    
    # Last week box plot
    if not df_last_week.empty:
        data_last_week = [df_last_week[df_last_week[variable] == cat][y_col].values 
                          for cat in df_last_week[variable].unique()]
        labels_last_week = df_last_week[variable].unique()
        ax1.boxplot(data_last_week, labels=labels_last_week, vert=False)
        ax1.set_title('Last Week')
        ax1.set_xlabel('First-Week Mortality')
        ax1.set_ylabel(variable)
    else:
        ax1.text(0.5, 0.5, 'No data for Last Week', ha='center', va='center')
        ax1.set_title('Last Week')
        ax1.set_xlabel('First-Week Mortality')
        ax1.set_ylabel(variable)
    
    # Last 8 weeks box plot
    if not df_last_8_weeks.empty:
        data_last_8_weeks = [df_last_8_weeks[df_last_8_weeks[variable] == cat][y_col].values 
                             for cat in df_last_8_weeks[variable].unique()]
        labels_last_8_weeks = df_last_8_weeks[variable].unique()
        ax2.boxplot(data_last_8_weeks, labels=labels_last_8_weeks, vert=False)
        ax2.set_title('Last 8 Weeks')
        ax2.set_xlabel('First-Week Mortality')
    else:
        ax2.text(0.5, 0.5, 'No data for Last 8 Weeks', ha='center', va='center')
        ax2.set_title('Last 8 Weeks')
        ax2.set_xlabel('First-Week Mortality')
    
    plt.suptitle(f'Distribution of First-Week Mortality by {variable}')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
