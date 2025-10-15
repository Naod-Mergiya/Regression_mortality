# data_preprocessing.py
import pandas as pd

def preprocess_data(df, y_col, x_cols):
    """
    Preprocess the dataset: parse dates, add week column, handle NaNs.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        y_col (str): Target column name
        x_cols (list): List of feature column names
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    # Parse dates
    df['Dispatch Date'] = pd.to_datetime(df['Dispatch Date'], format='%d-%b-%y')
    df['Week'] = df['Dispatch Date'].dt.isocalendar().week
    if 'Week' not in x_cols:
        x_cols.append('Week')
    
    # Handle NaNs in y: Drop rows
    df = df.dropna(subset=[y_col])
    
    # Handle NaNs in X: Fill with 'Unknown'
    df[x_cols] = df[x_cols].fillna('Unknown')
    
    return df

def filter_weeks(df):
    """
    Filter data for last week and last 8 weeks based on Dispatch Date.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame with Week_Start column
    
    Returns:
        tuple: (df_last_week, df_last_8_weeks)
    """
    # Add Week_Start column
    df['Weekday'] = df['Dispatch Date'].dt.weekday  # 0=Monday to 6=Sunday
    df['Week_Start'] = df['Dispatch Date'] - pd.to_timedelta(df['Weekday'], unit='d')
    
    # Find max_date and its week start
    max_date = df['Dispatch Date'].max()
    max_week_start = df.loc[df['Dispatch Date'] == max_date, 'Week_Start'].iloc[0]
    
    # Last week: data from the ISO week of max_date
    df_last_week = df[df['Week_Start'] == max_week_start]
    
    # Last 8 weeks: data from the 8 ISO weeks up to and including the last week
    start_8_weeks_ago = max_week_start - pd.to_timedelta(49, unit='d')  # 7 weeks back
    df_last_8_weeks = df[df['Week_Start'] >= start_8_weeks_ago]
    
    return df_last_week, df_last_8_weeks
