import os
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client

# --------------------------------------------------
# Load Supabase credentials from .env file
# --------------------------------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# --------------------------------------------------
# Check whether credentials exist
# --------------------------------------------------
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase credentials are missing. Please check your .env file.")
    st.stop()

# --------------------------------------------------
# Create Supabase client connection
# --------------------------------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------------------------------------
# Streamlit page
# --------------------------------------------------
st.title("Simple Dashboard App")
st.success("Supabase credentials loaded successfully.")
st.write("Project URL loaded from .env:")
st.code(SUPABASE_URL)