"""
Shared helper functions for the beginner-friendly ML scripts.

The training scripts save complete scikit-learn pipelines, so the prediction
scripts can load one file and use it without repeating preprocessing steps.
"""

from pathlib import Path

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv


# Project folders are resolved from this file, so scripts work from any terminal.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_DATA_PATH = PROJECT_ROOT / "reference aiman" / "simple_dash_customers.csv"
CUSTOMERS_TABLE_NAME = "customers"


def ensure_models_dir():
    """Create the models folder if it does not exist yet."""
    MODELS_DIR.mkdir(exist_ok=True)


def load_csv_data(csv_path=None):
    """Load customer data from a CSV file into a pandas DataFrame."""
    data_path = Path(csv_path) if csv_path else DEFAULT_DATA_PATH

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file: {data_path}")

    return pd.read_csv(data_path)


def load_latest_customer_data(csv_path=None):
    """
    Load customer data from the app's current data source.

    The dashboard stores customers in Supabase. For local learning or offline
    work, this function falls back to the sample CSV if credentials are missing
    or the database cannot be reached.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    supabase_url = _read_setting("SUPABASE_URL")
    supabase_key = _read_setting("SUPABASE_KEY")

    if supabase_url and supabase_key and "your-" not in supabase_url.lower():
        try:
            return load_supabase_table(supabase_url, supabase_key, CUSTOMERS_TABLE_NAME)
        except Exception as error:
            print(f"Could not load Supabase data, using CSV instead: {error}")

    return load_csv_data(csv_path)


def load_supabase_table(supabase_url, supabase_key, table_name):
    """Read all rows from a Supabase table through the REST API."""
    url = f"{supabase_url.rstrip('/')}/rest/v1/{table_name}"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
    }
    params = {"select": "*"}

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()

    return pd.DataFrame(response.json())


def save_model(model, model_filename):
    """Save a trained model pipeline inside the models folder."""
    ensure_models_dir()
    model_path = MODELS_DIR / model_filename
    joblib.dump(model, model_path)
    return model_path


def load_model(model_filename):
    """Load a trained model pipeline from the models folder."""
    model_path = MODELS_DIR / model_filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find {model_path}. Train the model first."
        )

    return joblib.load(model_path)


def clean_dataframe(df):
    """Return a copy of the data with blank cells handled consistently."""
    cleaned_df = df.copy()
    cleaned_df = cleaned_df.replace("", pd.NA)
    return cleaned_df


def make_prediction_dataframe(input_data):
    """Convert one dictionary or many dictionaries into a DataFrame."""
    if isinstance(input_data, dict):
        return pd.DataFrame([input_data])

    return pd.DataFrame(input_data)


def _read_setting(name):
    """Read a setting from the environment and strip extra spaces."""
    import os

    value = os.getenv(name)
    return str(value).strip() if value else None
