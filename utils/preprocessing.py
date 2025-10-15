
import os
import pandas as pd
from dotenv import load_dotenv
import urllib.parse
import pyodbc
from sqlalchemy import create_engine, text

# Load data from Azure SQL Database using service principal authentication
def load_data():
    load_dotenv()
    server = os.getenv("SQL_SERVER")
    database = os.getenv("SQL_DATABASE")
    client_id = os.getenv("AZURE_CLIENT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")
    tenant_id = os.getenv("AZURE_TENANT_ID")
    
    if not all([server, database, client_id, client_secret, tenant_id]):
        raise ValueError("Missing required environment variables for Azure SQL connection.")
    
    # URL-encode the password
    password_encoded = urllib.parse.quote_plus(client_secret)
    
    # Build the connection string for Azure SQL with AAD service principal
    connection_string = (
        f"mssql+pyodbc://{urllib.parse.quote_plus(client_id)}:{password_encoded}@{server}/{database}?"
        f"driver=ODBC+Driver+17+for+SQL+Server&"
        f"authentication=ActiveDirectoryServicePrincipal&"
        f"tenant_id={tenant_id}&"
        f"TrustServerCertificate=yes"
    )
    
    engine = create_engine(connection_string)
    
# Replace with your actual query/table name
    query = """SELECT 
            hatchery,
            hatcher,
            setter,
            driver_id,
            vehicle_number,
            source_of_eggs,
            customer_type,
            doc_dead_1st_week,
            docs_received,
            dispatch_date
        FROM [hatchconnect].[SurveysData]"""  
    df = pd.read_sql(query, engine)
    print(f"✅ Data loaded successfully from {database}")
    print(f"📊 Rows: {len(df)}, Columns: {len(df.columns)}")
    print(df.head())
    df.to_csv('doc_mortality.csv')
    
    return df
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

def preprocess_data(df, y_col, x_cols):
    """
    Preprocess the dataset: parse dates, add week column, handle NaNs.
    """
    df['dispatch_date'] = pd.to_datetime(df['dispatch_date'], errors='coerce')
    df['Week'] = df['dispatch_date'].dt.isocalendar().week
    if 'Week' not in x_cols:
        x_cols.append('Week')
    
    # Drop rows where the target is missing
    df = df.dropna(subset=[y_col])
    
    # Fill missing predictors: 'Unknown' for object/categorical, 0 for numeric
    for col in x_cols:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].fillna('Unknown')
            df[col] = df[col].astype('category')  # Convert to category dtype
        else:
            df[col] = df[col].fillna(0)
    
    # Optional: Print value counts for debugging
    # for col in x_cols:
    #     print(f"{col} value counts after fill:")
    #     print(df[col].value_counts())
    
    return df

def filter_weeks(df):
    """
    Return DataFrames for the most recent week and the last 8 weeks.
    """
    df['Weekday'] = df['dispatch_date'].dt.weekday
    df['Week_Start'] = df['dispatch_date'] - pd.to_timedelta(df['Weekday'], unit='d')
    
    max_date = df['dispatch_date'].max()
    max_week_start = df.loc[df['dispatch_date'] == max_date, 'Week_Start'].iloc[0]
    
    df_last_week = df[df['Week_Start'] == max_week_start]
    start_8_weeks_ago = max_week_start - pd.to_timedelta(49, unit='d')
    df_last_8_weeks = df[df['Week_Start'] >= start_8_weeks_ago]
    
    return df_last_week, df_last_8_weeks
