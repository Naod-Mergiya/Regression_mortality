
import os
import pandas as pd
from dotenv import load_dotenv
import pyodbc
from sqlalchemy import create_engine, text

def load_data() -> pd.DataFrame:
    """
    Load selected columns from Azure SQL Database using Service Principal authentication.

    Returns:
        pd.DataFrame: DataFrame with selected columns.
    """
    # Environment variables
    server = os.getenv("SQL_SERVER")
    database = os.getenv("SQL_DATABASE")
    client_id = os.getenv("AZURE_CLIENT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")
    tenant_id = os.getenv("AZURE_TENANT_ID")

    # Build connection string for SQLAlchemy
    conn_str = (
        f"mssql+pyodbc://{client_id}:{client_secret}@{server}:1433/{database}"
        f"?driver=ODBC+Driver+17+for+SQL+Server"
        f"&authentication=ActiveDirectoryServicePrincipal"
        f"&tenant_id={tenant_id}"
        f"&Encrypt=yes"
        f"&TrustServerCertificate=no"
    )

    # SQL query — only the needed columns
    query = text("""
        SELECT 
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
        FROM [hatchconnect].[SurveysData]
    """)

    # Create SQLAlchemy engine
    engine = create_engine(conn_str)

    # Execute query
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    print(f"✅ Data loaded successfully from {database}")
    print(f"📊 Rows: {len(df)}, Columns: {len(df.columns)}")
    print(df.head())

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
