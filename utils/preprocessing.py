
import pandas as pd

def load_data(csv_path: str = r"regression test.csv") -> pd.DataFrame:
    """
    Load the raw CSV file and return a DataFrame.
    
    Args:
        csv_path (str): Full path to the CSV file. Defaults to specified path.
    
    Returns:
        pd.DataFrame: Loaded data.
    """
    df = pd.read_csv(csv_path)
    print(f"Data loaded from {csv_path}")
    print("First 5 rows:")
    print(df.head())
    return df

def preprocess_data(df, y_col, x_cols):
    """
    Preprocess the dataset: parse dates, add week column, handle NaNs.
    """
    df['Dispatch Date'] = pd.to_datetime(df['Dispatch Date'], format='%d-%b-%y')
    df['Week'] = df['Dispatch Date'].dt.isocalendar().week
    if 'Week' not in x_cols:
        x_cols.append('Week')
    
    # Drop rows where the target is missing
    df = df.dropna(subset=[y_col])
    
    # Fill missing predictors with 'Unknown'
    df[x_cols] = df[x_cols].fillna('Unknown')
    
    return df

def filter_weeks(df):
    """
    Return DataFrames for the most recent week and the last 8 weeks.
    """
    df['Weekday'] = df['Dispatch Date'].dt.weekday
    df['Week_Start'] = df['Dispatch Date'] - pd.to_timedelta(df['Weekday'], unit='d')
    
    max_date = df['Dispatch Date'].max()
    max_week_start = df.loc[df['Dispatch Date'] == max_date, 'Week_Start'].iloc[0]
    
    df_last_week = df[df['Week_Start'] == max_week_start]
    start_8_weeks_ago = max_week_start - pd.to_timedelta(49, unit='d')
    df_last_8_weeks = df[df['Week_Start'] >= start_8_weeks_ago]
    
    return df_last_week, df_last_8_weeks
