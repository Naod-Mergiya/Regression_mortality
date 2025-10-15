import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

def run_regression(df_subset, x_cols, y_col, subset_name):
    """
    Run regression on the given dataset and return model results.
    
    Args:
        df_subset (pd.DataFrame): Subset of data to analyze
        x_cols (list): List of feature column names
        y_col (str): Target column name
        subset_name (str): Name of the dataset subset (for printing)
    
    Returns:
        tuple: (model, X_transformed_df, feature_names) or (None, None, None) if invalid
    """
    if df_subset.empty:
        print(f"No data available for {subset_name}")
        return None, None, None
    
    # Separate features (X) and target (y)
    X = df_subset[x_cols]
    y = df_subset[y_col]
    
    if len(y) < 2:
        print(f"Insufficient data for regression in {subset_name}")
        return None, None, None
    
    # Create a preprocessor to handle categorical variables using OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), x_cols)
        ])
    
    # Transform X
    X_transformed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    
    # Force dense conversion to fix sparse matrix flattening in pd.DataFrame
    if hasattr(X_transformed, 'toarray'):
        X_transformed = X_transformed.toarray()
    elif isinstance(X_transformed, np.ndarray) and X_transformed.ndim == 1:
        raise ValueError(f"X_transformed is unexpectedly 1D: {X_transformed.shape}. Check x_cols for non-categorical columns.")
    
    # Debug: Verify shapes match (remove this line after confirming it works)
    print(f"Debug: X_transformed shape: {X_transformed.shape}, expected columns: {len(feature_names)}")
    
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
    
    # Add constant for intercept
    X_transformed_df = sm.add_constant(X_transformed_df)
    
    # Fit the model using statsmodels
    model = sm.OLS(y, X_transformed_df).fit()
    
    # Print the summary
    print(f"\n{subset_name} Regression Summary:")
    print(model.summary())
    
    # Print sorted parameters by absolute value
    print(f"\n{subset_name} Parameters sorted by absolute value:")
    print(model.params.abs().sort_values(ascending=False))
    
    return model, X_transformed_df, feature_names