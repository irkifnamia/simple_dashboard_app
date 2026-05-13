import importlib
import os
import sys
from html import escape
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from supabase import create_client

st.set_page_config(
    page_title="Executive Customer Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Load Supabase credentials
# --------------------------------------------------
# load_dotenv() lets you keep SUPABASE_URL and SUPABASE_KEY
# inside a local .env file while developing on your computer.
load_dotenv()


def get_streamlit_secret(name):
    """Try to read a value from Streamlit Cloud secrets."""
    try:
        # Format 1:
        # SUPABASE_URL = "..."
        # SUPABASE_KEY = "..."
        if name in st.secrets:
            return st.secrets[name]

        # Format 2:
        # [supabase]
        # SUPABASE_URL = "..."
        # SUPABASE_KEY = "..."
        if "supabase" in st.secrets and name in st.secrets["supabase"]:
            return st.secrets["supabase"][name]
    except Exception:
        return None

    return None


def get_setting(name):
    """Read settings from Streamlit secrets first, then local .env."""
    value = get_streamlit_secret(name)

    if not value:
        value = os.getenv(name)

    if value:
        return str(value).strip()

    return None


SUPABASE_URL = get_setting("SUPABASE_URL")
SUPABASE_KEY = get_setting("SUPABASE_KEY")


class SupabaseRestResponse:
    """Small response object that matches the .data style used below."""

    def __init__(self, data):
        self.data = data


class SupabaseRestTable:
    """Simple REST table helper for newer Supabase publishable keys."""

    def __init__(self, supabase_url, supabase_key, table_name):
        self.supabase_url = supabase_url.rstrip("/")
        self.supabase_key = supabase_key
        self.table_name = table_name
        self.action = None
        self.payload = None
        self.filters = []
        self.selected_columns = "*"

    def select(self, columns):
        self.action = "select"
        self.selected_columns = columns
        return self

    def insert(self, payload):
        self.action = "insert"
        self.payload = payload
        return self

    def update(self, payload):
        self.action = "update"
        self.payload = payload
        return self

    def delete(self):
        self.action = "delete"
        return self

    def eq(self, column, value):
        safe_value = quote(str(value), safe="")
        self.filters.append(f"{column}=eq.{safe_value}")
        return self

    def execute(self):
        url = f"{self.supabase_url}/rest/v1/{self.table_name}"
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

        if self.action == "select":
            response = requests.get(
                url,
                headers=headers,
                params={"select": self.selected_columns},
                timeout=20,
            )
        elif self.action == "insert":
            response = requests.post(url, headers=headers, json=self.payload, timeout=20)
        elif self.action == "update":
            response = requests.patch(
                f"{url}?{'&'.join(self.filters)}",
                headers=headers,
                json=self.payload,
                timeout=20,
            )
        elif self.action == "delete":
            response = requests.delete(
                f"{url}?{'&'.join(self.filters)}",
                headers=headers,
                timeout=20,
            )
        else:
            raise ValueError("Choose select, insert, update, or delete before execute.")

        if not response.ok:
            raise RuntimeError(response.text)

        return SupabaseRestResponse(response.json() if response.text else [])


class SupabaseRestClient:
    """Small client that supports the table calls used by this dashboard."""

    def __init__(self, supabase_url, supabase_key):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key

    def table(self, table_name):
        return SupabaseRestTable(self.supabase_url, self.supabase_key, table_name)


def create_dashboard_supabase_client(supabase_url, supabase_key):
    """Create a Supabase client that works with old and new key formats."""
    if supabase_key.startswith(("sb_publishable_", "sb_secret_")):
        return SupabaseRestClient(supabase_url, supabase_key)

    return create_client(supabase_url, supabase_key)

# --------------------------------------------------
# Check whether credentials exist
# --------------------------------------------------
if (
    not SUPABASE_URL
    or not SUPABASE_KEY
    or "your-" in SUPABASE_URL.lower()
    or "your-" in SUPABASE_KEY.lower()
):
    st.error(
        "Supabase credentials are missing or still using placeholder values. "
        "Set SUPABASE_URL and SUPABASE_KEY in Streamlit Cloud secrets."
    )
    st.stop()

# --------------------------------------------------
# Create Supabase client connection
# --------------------------------------------------
try:
    supabase = create_dashboard_supabase_client(SUPABASE_URL, SUPABASE_KEY)
except Exception:
    st.error(
        "Supabase connection failed. Check that SUPABASE_URL and SUPABASE_KEY "
        "in Streamlit Cloud secrets are copied exactly from your Supabase project."
    )
    st.stop()

# These fields are used if the customers table is empty.
# If your table already has data, the app will use the existing table columns.
DEFAULT_CUSTOMER_FIELDS = ["customer_code", "customer_name", "email", "phone"]

# These fields are usually managed by the database, so we do not edit them here.
READ_ONLY_FIELDS = ["id", "created_at", "updated_at"]
ML_DIR = Path(__file__).resolve().parent / "ml"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_NOT_FOUND_MESSAGE = "Model not found. Please train the model first."
ML_IDENTIFIER_FIELDS = {
    "id",
    "user_id",
    "customer_id",
    "customer_code",
    "customer_name",
    "full_name",
    "email",
    "phone",
    "address",
}


def apply_custom_styles():
    """Add simple CSS so the app feels like a modern executive dashboard."""
    st.markdown(
        """
        <style>
            :root {
                --primary: #17446b;
                --primary-dark: #0f2f4a;
                --accent: #2563eb;
                --background: #f3f6fb;
                --surface: #ffffff;
                --surface-soft: #f8fafc;
                --chart-surface: #ffffff;
                --border: #cbd5e1;
                --text: #0f172a;
                --muted: #334155;
                --muted-strong: #1e293b;
            }

            .stApp {
                background: var(--background);
                color: var(--text);
            }

            p, label, span, div {
                color: var(--text);
            }

            .stMarkdown, .stText, .stCaptionContainer {
                color: var(--text);
            }

            [data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid var(--border);
            }

            [data-testid="stHeader"] {
                background:
                    radial-gradient(circle at 16px 16px, rgba(212, 175, 55, 0.38) 0 3px, transparent 4px),
                    radial-gradient(circle at 48px 16px, rgba(23, 68, 107, 0.36) 0 3px, transparent 4px),
                    radial-gradient(circle at 16px 48px, rgba(15, 118, 110, 0.30) 0 3px, transparent 4px),
                    radial-gradient(circle at 48px 48px, rgba(212, 175, 55, 0.28) 0 3px, transparent 4px),
                    linear-gradient(135deg, #f8fafc 0%, #edf4fb 42%, #ffffff 100%);
                background-size: 64px 64px, 64px 64px, 64px 64px, 64px 64px, 100% 100%;
                border-bottom: 1px solid #cbd5e1;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            }

            [data-testid="stHeader"]::before {
                background:
                    repeating-linear-gradient(
                        45deg,
                        rgba(23, 68, 107, 0.10) 0 2px,
                        transparent 2px 12px
                    ),
                    repeating-linear-gradient(
                        -45deg,
                        rgba(212, 175, 55, 0.12) 0 2px,
                        transparent 2px 12px
                    );
                content: "";
                inset: 0;
                opacity: 0.9;
                pointer-events: none;
                position: absolute;
            }

            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {
                color: var(--primary-dark);
            }

            [data-testid="stSidebar"] [role="radiogroup"] label {
                border-radius: 10px;
                color: var(--muted-strong);
                font-weight: 600;
                padding: 0.45rem 0.65rem;
                margin-bottom: 0.15rem;
            }

            [data-testid="stSidebar"] small,
            [data-testid="stSidebar"] .stCaptionContainer {
                color: var(--muted-strong);
            }

            [data-testid="stSidebar"] [data-testid="stMultiSelect"] label,
            [data-testid="stSidebar"] [data-testid="stSlider"] label {
                font-size: 0.82rem;
            }

            [data-testid="stSidebar"] [data-testid="stMultiSelect"] span,
            [data-testid="stSidebar"] [data-testid="stSlider"] span {
                font-size: 0.82rem;
            }

            .main .block-container {
                padding-top: 1.4rem;
                padding-bottom: 2rem;
                max-width: 1500px;
            }

            h1, h2, h3 {
                color: var(--text);
                letter-spacing: 0;
            }

            h1 {
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }

            h2 {
                font-size: 1.35rem;
                font-weight: 700;
            }

            h3 {
                font-size: 1.05rem;
                font-weight: 650;
            }

            .page-title {
                font-size: 2.65rem;
                font-weight: 800;
                color: var(--text);
                line-height: 1.1;
                margin-bottom: 0.1rem;
            }

            .page-subtitle {
                color: var(--muted-strong);
                font-size: 0.98rem;
                margin-bottom: 1.1rem;
            }

            .section-label {
                color: var(--primary);
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                margin-top: 0.2rem;
                margin-bottom: 0.25rem;
                text-transform: uppercase;
            }

            .section-title {
                color: var(--text);
                font-size: 1.2rem;
                font-weight: 700;
                margin-bottom: 0.7rem;
            }

            .insight-card {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 14px;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
                padding: 1rem 1.1rem;
                margin-bottom: 1rem;
            }

            .chart-title {
                color: var(--text);
                font-size: 0.95rem;
                font-weight: 700;
                line-height: 1.25;
                margin-bottom: 0.55rem;
            }

            .chart-note {
                color: var(--muted-strong);
                font-size: 0.82rem;
                margin-bottom: 0.45rem;
            }

            div[data-testid="stVegaLiteChart"] {
                background: var(--chart-surface);
                border: 1px solid var(--border);
                border-radius: 14px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
                margin-bottom: 1rem;
                overflow: hidden;
                padding: 1rem 1.1rem 1.15rem;
            }

            div[data-testid="stDataFrame"] {
                background: var(--chart-surface);
                border: 1px solid var(--border);
                border-radius: 14px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
                margin-bottom: 1rem;
                overflow: hidden;
                padding: 0.55rem;
            }

            .pie-chart-wrap {
                align-items: center;
                background: transparent;
                display: flex;
                gap: 0.8rem;
                min-height: 280px;
                padding: 0;
            }

            .pie-chart {
                align-items: center;
                display: flex;
                flex: 0 0 145px;
                height: 145px;
                justify-content: center;
                width: 145px;
            }

            .pie-chart-center {
                align-items: center;
                background: #ffffff;
                border-radius: 50%;
                color: var(--primary-dark);
                display: flex;
                font-size: 1.15rem;
                font-weight: 750;
                height: 88px;
                justify-content: center;
                width: 88px;
            }

            .pie-legend {
                display: flex;
                flex: 1;
                flex-direction: column;
                gap: 0.45rem;
            }

            .pie-legend-row {
                align-items: center;
                display: grid;
                font-size: 0.9rem;
                gap: 0.5rem;
                grid-template-columns: 12px 1fr auto;
            }

            .pie-legend-row span,
            .pie-legend-row strong {
                color: var(--muted-strong);
            }

            .pie-legend-dot {
                border-radius: 999px;
                height: 10px;
                width: 10px;
            }

            .brand-logo {
                align-items: center;
                background:
                    radial-gradient(circle at 70% 28%, #c1121f 0 9px, transparent 10px),
                    linear-gradient(145deg, #05070b 0%, #101827 58%, #1f2937 100%);
                border: 3px solid #d4af37;
                border-radius: 999px;
                box-shadow: 0 12px 28px rgba(15, 23, 42, 0.18);
                color: #ffffff;
                display: flex;
                font-size: 1.35rem;
                font-weight: 800;
                height: 74px;
                justify-content: center;
                letter-spacing: 0.04em;
                margin-bottom: 1rem;
                width: 74px;
            }

            .brand-name {
                color: var(--primary-dark);
                font-size: 1.15rem;
                font-weight: 800;
                line-height: 1.15;
                margin-bottom: 0.25rem;
            }

            .brand-tagline {
                color: var(--muted-strong);
                font-size: 0.82rem;
                font-weight: 600;
                line-height: 1.35;
                margin-bottom: 1.35rem;
            }

            .mini-card {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 14px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
                padding: 0.85rem 1rem;
                min-height: 88px;
            }

            .mini-card-label {
                color: var(--muted-strong);
                font-size: 0.78rem;
                font-weight: 700;
                margin-bottom: 0.35rem;
                text-transform: uppercase;
            }

            .mini-card-value {
                color: var(--primary-dark);
                font-size: 1.35rem;
                font-weight: 750;
                line-height: 1.2;
            }

            [data-testid="stMetric"] {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 14px;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
                padding: 1rem 1.05rem;
            }

            [data-testid="stMetricLabel"] {
                color: var(--muted-strong);
                font-size: 0.82rem;
                font-weight: 650;
            }

            [data-testid="stMetricValue"] {
                color: var(--primary-dark);
                font-size: 1.45rem;
                font-weight: 750;
            }

            div[data-testid="stDataFrame"],
            div[data-testid="stTable"] {
                border-radius: 14px;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
                overflow: hidden;
            }

            div[data-testid="stForm"] {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 14px;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
                padding: 1rem;
            }

            .stButton button,
            .stFormSubmitButton button {
                background: var(--primary) !important;
                border: 1px solid var(--primary) !important;
                border-radius: 10px;
                color: #ffffff !important;
                font-weight: 650;
                min-height: 2.45rem;
            }

            .stButton button *,
            .stFormSubmitButton button * {
                color: #ffffff !important;
            }

            .stButton button:hover,
            .stFormSubmitButton button:hover {
                background: var(--primary-dark) !important;
                border-color: var(--primary-dark) !important;
                color: #ffffff !important;
            }

            .stButton button:hover *,
            .stFormSubmitButton button:hover * {
                color: #ffffff !important;
            }

            .stButton button:focus,
            .stFormSubmitButton button:focus {
                box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.28) !important;
                color: #ffffff !important;
                outline: none !important;
            }

            .stButton button:disabled,
            .stFormSubmitButton button:disabled {
                background: #94a3b8 !important;
                border-color: #94a3b8 !important;
                color: #ffffff !important;
                opacity: 1 !important;
            }

            .stButton button:disabled *,
            .stFormSubmitButton button:disabled * {
                color: #ffffff !important;
            }

            div[data-testid="stAlert"] {
                border-radius: 12px;
            }

            hr {
                margin: 1.25rem 0;
            }

            @media (max-width: 900px) {
                .main .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                }

                .page-title {
                    font-size: 2rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title, subtitle):
    """Show a consistent page heading."""
    st.markdown(f"<div class='page-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def render_section_header(label, title):
    """Show a compact section heading."""
    st.markdown(f"<div class='section-label'>{label}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)


def render_mini_card(label, value):
    """Display a small summary card using beginner-friendly HTML."""
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-card-label">{label}</div>
            <div class="mini-card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_message():
    """Show a success message after Streamlit reruns the app."""
    if "message" in st.session_state:
        st.success(st.session_state.message)
        del st.session_state.message


def fetch_customers():
    """Read all rows from the customers table and return a pandas dataframe."""
    response = supabase.table("customers").select("*").execute()
    return pd.DataFrame(response.data)


def get_editable_fields(customers_df):
    """Choose which customer fields should appear in the forms."""
    if customers_df.empty:
        return DEFAULT_CUSTOMER_FIELDS

    editable_fields = [
        column
        for column in customers_df.columns
        if column not in READ_ONLY_FIELDS
    ]

    # Keep customer_code first because it is the visible customer identifier.
    if "customer_code" in editable_fields:
        editable_fields.remove("customer_code")
        editable_fields.insert(0, "customer_code")

    return editable_fields


def refresh_with_message(message):
    """Save a message, then rerun the app so the dataframe refreshes."""
    st.session_state.message = message
    st.rerun()


def get_numeric_column(customers_df, column_name):
    """Convert a dataframe column to numbers for KPI calculations."""
    if column_name not in customers_df.columns:
        return pd.Series(dtype="float")

    return pd.to_numeric(customers_df[column_name], errors="coerce").fillna(0)


def show_chart_title(title):
    """Display a consistent title above each chart."""
    st.markdown(f"<div class='chart-title'>{title}</div>", unsafe_allow_html=True)


def get_chart_theme():
    """Use one professional color palette across charts."""
    return ["#17446b", "#2563eb", "#0f766e", "#7c3aed", "#b45309", "#be123c"]


def get_chart_config():
    """Keep all built-in charts bright and readable."""
    return {
        "background": "#ffffff",
        "axis": {
            "domainColor": "#cbd5e1",
            "gridColor": "#e2e8f0",
            "labelColor": "#1e293b",
            "labelFontSize": 12,
            "titleColor": "#0f172a",
            "titleFontSize": 13,
        },
        "legend": {
            "labelColor": "#1e293b",
            "labelFontSize": 12,
            "titleColor": "#0f172a",
            "titleFontSize": 13,
        },
        "view": {"stroke": "transparent"},
    }


def show_vega_chart(chart_spec, chart_data):
    """Render a Vega-Lite chart with a consistent enterprise theme."""
    chart_spec["data"] = {"values": chart_data.to_dict("records")}
    chart_spec["config"] = get_chart_config()
    chart_spec["autosize"] = {"type": "fit", "contains": "padding"}
    chart_spec["padding"] = {"left": 16, "right": 18, "top": 14, "bottom": 18}
    st.vega_lite_chart(chart_spec, use_container_width=True)


MALAYSIA_STATE_ALIASES = {
    "johor": "Johor",
    "kedah": "Kedah",
    "kelantan": "Kelantan",
    "melaka": "Melaka",
    "malacca": "Melaka",
    "negeri sembilan": "Negeri Sembilan",
    "n sembilan": "Negeri Sembilan",
    "pahang": "Pahang",
    "penang": "Pulau Pinang",
    "pulau pinang": "Pulau Pinang",
    "perak": "Perak",
    "perlis": "Perlis",
    "sabah": "Sabah",
    "sarawak": "Sarawak",
    "selangor": "Selangor",
    "terengganu": "Terengganu",
    "kuala lumpur": "Kuala Lumpur",
    "wp k lumpur": "Kuala Lumpur",
    "wp kuala lumpur": "Kuala Lumpur",
    "w.p. kuala lumpur": "Kuala Lumpur",
    "wilayah persekutuan kuala lumpur": "Kuala Lumpur",
    "putrajaya": "Putrajaya",
    "wp putrajaya": "Putrajaya",
    "w.p. putrajaya": "Putrajaya",
    "labuan": "Labuan",
    "wp labuan": "Labuan",
    "w.p. labuan": "Labuan",
}


MALAYSIA_STATE_GEOJSON_URL = (
    "https://raw.githubusercontent.com/mptwaktusolat/"
    "jakim.geojson/master/malaysia.state.geojson"
)


def normalize_malaysia_state(value):
    """Return a consistent Malaysia state name for grouping."""
    if pd.isna(value):
        return "Unknown"

    clean_value = str(value).strip()
    if not clean_value:
        return "Unknown"

    return MALAYSIA_STATE_ALIASES.get(clean_value.lower(), clean_value.title())


def interpolate_hex_color(start_hex, end_hex, fraction):
    """Blend two hex colors for the map heat scale."""
    safe_fraction = max(0, min(1, fraction))
    start_rgb = tuple(int(start_hex[index:index + 2], 16) for index in (1, 3, 5))
    end_rgb = tuple(int(end_hex[index:index + 2], 16) for index in (1, 3, 5))
    blended = tuple(
        round(start + (end - start) * safe_fraction)
        for start, end in zip(start_rgb, end_rgb)
    )
    return "#{:02x}{:02x}{:02x}".format(*blended)


def get_state_average_map_data(customers_df, metric_column):
    """Calculate state averages for Malaysia map charts."""
    if "state" not in customers_df.columns or metric_column not in customers_df.columns:
        return pd.DataFrame()

    map_df = customers_df.copy()
    map_df["Malaysia State"] = map_df["state"].apply(normalize_malaysia_state)
    map_df[metric_column] = get_numeric_column(map_df, metric_column)
    map_df = map_df[map_df["Malaysia State"] != "Unknown"]

    if map_df.empty:
        return pd.DataFrame()

    return (
        map_df.groupby("Malaysia State", as_index=False)
        .agg(Average=(metric_column, "mean"), Customers=(metric_column, "size"))
        .sort_values("Average", ascending=False)
    )


@st.cache_data(ttl=3600)
def load_malaysia_state_geojson():
    """Fetch Malaysia state boundaries used by the choropleth maps."""
    response = requests.get(MALAYSIA_STATE_GEOJSON_URL, timeout=20)
    response.raise_for_status()
    return response.json()


def get_geojson_state_name(properties):
    """Find the state name field used in the Malaysia GeoJSON."""
    for key in ["name", "Name", "state_name", "state", "STATE", "shapeName"]:
        if key in properties and properties[key]:
            return normalize_malaysia_state(properties[key])

    return "Unknown"


def get_geometry_rings(geometry):
    """Return all polygon rings from a GeoJSON geometry."""
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])

    if geometry_type == "Polygon":
        return coordinates

    if geometry_type == "MultiPolygon":
        rings = []
        for polygon in coordinates:
            rings.extend(polygon)
        return rings

    return []


def get_geojson_bbox(features):
    """Calculate a simple lon/lat bounding box for GeoJSON features."""
    longitudes = []
    latitudes = []

    for feature in features:
        for ring in get_geometry_rings(feature.get("geometry", {})):
            for longitude, latitude in ring:
                longitudes.append(longitude)
                latitudes.append(latitude)

    if not longitudes or not latitudes:
        return (99, 0, 120, 8)

    return (min(longitudes), min(latitudes), max(longitudes), max(latitudes))


def project_geo_point(longitude, latitude, bbox, width, height, padding):
    """Project lon/lat into the SVG viewport with preserved aspect ratio."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_span = max(max_lon - min_lon, 0.0001)
    lat_span = max(max_lat - min_lat, 0.0001)
    scale = min((width - padding * 2) / lon_span, (height - padding * 2) / lat_span)
    map_width = lon_span * scale
    map_height = lat_span * scale
    offset_x = (width - map_width) / 2
    offset_y = (height - map_height) / 2
    x = offset_x + ((longitude - min_lon) * scale)
    y = offset_y + ((max_lat - latitude) * scale)
    return x, y


def make_geojson_path(geometry, bbox, width, height, padding):
    """Convert GeoJSON polygon coordinates into an SVG path."""
    path_parts = []

    for ring in get_geometry_rings(geometry):
        if not ring:
            continue

        first_x, first_y = project_geo_point(
            ring[0][0],
            ring[0][1],
            bbox,
            width,
            height,
            padding,
        )
        commands = [f"M {first_x:.1f} {first_y:.1f}"]

        for longitude, latitude in ring[1:]:
            x, y = project_geo_point(longitude, latitude, bbox, width, height, padding)
            commands.append(f"L {x:.1f} {y:.1f}")

        commands.append("Z")
        path_parts.append(" ".join(commands))

    return " ".join(path_parts)


def make_malaysia_choropleth_html(map_data, title, metric_label, value_format):
    """Build a real Malaysia state-boundary choropleth as SVG."""
    geojson = load_malaysia_state_geojson()
    values_by_state = {
        row["Malaysia State"]: {
            "average": float(row["Average"]),
            "customers": int(row["Customers"]),
        }
        for _, row in map_data.iterrows()
    }
    features = geojson.get("features", [])
    bbox = get_geojson_bbox(features)
    width = 720
    height = 330
    padding = 18
    values = [state_data["average"] for state_data in values_by_state.values()]
    min_value = min(values) if values else 0
    max_value = max(values) if values else 0
    color_start = "#dbeafe"
    color_end = "#17446b"

    path_parts = []
    for feature in features:
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        state_name = get_geojson_state_name(properties)
        state_data = values_by_state.get(state_name)
        path_data = make_geojson_path(geometry, bbox, width, height, padding)

        if state_data:
            average_value = state_data["average"]
            customers = state_data["customers"]
            fraction = 0.65 if max_value == min_value else (
                (average_value - min_value) / (max_value - min_value)
            )
            fill = interpolate_hex_color(color_start, color_end, fraction)
            tooltip = (
                f"{state_name}: {metric_label} "
                f"{value_format.format(average_value)} from {customers:,} customers"
            )
        else:
            fill = "#e5e7eb"
            tooltip = f"{state_name}: No data"

        path_parts.append(
            f"""
            <path
                d="{path_data}"
                fill="{fill}"
                fill-rule="evenodd"
                stroke="#ffffff"
                stroke-linejoin="round"
                stroke-width="1.1"
            >
                <title>{escape(tooltip)}</title>
            </path>
            """
        )

    top_rows = []
    for _, row in map_data.head(5).iterrows():
        top_rows.append(
            f"""
            <div class="map-rank-row">
                <span>{escape(str(row["Malaysia State"]))}</span>
                <strong>{value_format.format(float(row["Average"]))}</strong>
            </div>
            """
        )

    min_label = value_format.format(min_value) if values else "0"
    max_label = value_format.format(max_value) if values else "0"

    return f"""
    <style>
        body {{
            margin: 0;
            background: #ffffff;
            color: #0f172a;
            font-family: "Source Sans Pro", Arial, sans-serif;
        }}

        .malaysia-map-card {{
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 14px;
            box-sizing: border-box;
            display: grid;
            gap: 1rem;
            grid-template-columns: minmax(0, 1fr) minmax(165px, 25%);
            height: 350px;
            padding: 1rem 1.05rem 1rem 1rem;
            width: 100%;
        }}

        .map-title {{
            color: #0f172a;
            font-size: 0.95rem;
            font-weight: 800;
            margin: 0 0 0.45rem;
        }}

        .map-svg {{
            display: block;
            height: 278px;
            width: 100%;
        }}

        .map-svg path {{
            filter: drop-shadow(0 1px 1px rgba(15, 23, 42, 0.08));
            transition: opacity 0.12s ease, stroke-width 0.12s ease;
        }}

        .map-svg path:hover {{
            opacity: 0.86;
            stroke: #0f172a;
            stroke-width: 1.7;
        }}

        .map-side {{
            align-self: stretch;
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
            justify-content: center;
            min-width: 0;
            padding-right: 0.25rem;
        }}

        .map-legend {{
            display: grid;
            gap: 0.35rem;
        }}

        .map-legend-scale {{
            background: linear-gradient(90deg, {color_start}, {color_end});
            border-radius: 999px;
            height: 12px;
        }}

        .map-legend-labels {{
            color: #334155;
            display: flex;
            font-size: 0.75rem;
            font-weight: 700;
            justify-content: space-between;
        }}

        .map-rank {{
            display: grid;
            gap: 0.42rem;
        }}

        .map-rank-heading {{
            color: #0f172a;
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
        }}

        .map-rank-row {{
            align-items: center;
            border-bottom: 1px solid #e2e8f0;
            color: #1e293b;
            display: flex;
            font-size: 0.76rem;
            gap: 0.45rem;
            justify-content: space-between;
            padding-bottom: 0.32rem;
        }}

        .map-rank-row span {{
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .map-rank-row strong {{
            color: #0f172a;
            white-space: nowrap;
        }}

        @media (max-width: 420px) {{
            .malaysia-map-card {{
                grid-template-columns: 1fr;
                height: auto;
            }}

            .map-svg {{
                height: 260px;
            }}
        }}
    </style>
    <div class="malaysia-map-card">
        <div>
            <h3 class="map-title">{escape(title)}</h3>
            <svg
                class="map-svg"
                viewBox="0 0 {width} {height}"
                role="img"
                aria-label="{escape(title)}"
            >
                <rect x="0" y="0" width="{width}" height="{height}" rx="12" fill="#f8fafc" />
                {"".join(path_parts)}
            </svg>
        </div>
        <div class="map-side">
            <div class="map-legend">
                <div class="map-legend-scale"></div>
                <div class="map-legend-labels">
                    <span>{min_label}</span>
                    <span>{max_label}</span>
                </div>
            </div>
            <div class="map-rank">
                <div class="map-rank-heading">Top average</div>
                {"".join(top_rows)}
            </div>
        </div>
    </div>
    """


def make_pie_chart_svg(labels, values):
    """Create a simple SVG donut chart without extra chart libraries."""
    total = sum(values)
    if total <= 0:
        return "<p>No values available for the pie chart.</p>"

    colors = get_chart_theme()
    current_offset = 0
    circle_parts = []
    legend_parts = []

    for index, label in enumerate(labels):
        value = values[index]
        percent = (value / total) * 100
        color = colors[index % len(colors)]
        safe_label = escape(str(label))

        circle_parts.append(
            f"""
            <circle
                cx="90"
                cy="90"
                fill="transparent"
                r="58"
                stroke="{color}"
                stroke-dasharray="{percent:.4f} {100 - percent:.4f}"
                stroke-dashoffset="{-current_offset:.4f}"
                stroke-linecap="butt"
                stroke-width="28"
                transform="rotate(-90 90 90)"
                pathLength="100"
            />
            """
        )

        legend_parts.append(
            f"""
            <div class="pie-legend-row">
                <span class="pie-legend-dot" style="background:{color};"></span>
                <span>{safe_label}</span>
                <strong>{value:,} ({percent:.1f}%)</strong>
            </div>
            """
        )
        current_offset += percent

    circle_html = "".join(circle_parts)
    legend_html = "".join(legend_parts)

    return f"""
    <style>
        body {{
            margin: 0;
            background: #ffffff;
            color: #0f172a;
            font-family: "Source Sans Pro", Arial, sans-serif;
        }}

        .pie-chart-wrap {{
            align-items: center;
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 14px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            box-sizing: border-box;
            display: flex;
            gap: 1rem;
            height: 330px;
            padding: 1rem;
            width: 100%;
        }}

        .pie-chart {{
            align-items: center;
            display: flex;
            flex: 0 0 170px;
            height: 170px;
            justify-content: center;
            width: 170px;
        }}

        .pie-legend {{
            display: flex;
            flex: 1;
            flex-direction: column;
            gap: 0.45rem;
            min-width: 0;
        }}

        .pie-legend-row {{
            align-items: center;
            color: #1e293b;
            display: grid;
            font-size: 0.9rem;
            gap: 0.5rem;
            grid-template-columns: 12px minmax(0, 1fr) auto;
        }}

        .pie-legend-row span:nth-child(2) {{
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .pie-legend-row strong {{
            color: #1e293b;
            font-weight: 700;
        }}

        .pie-legend-dot {{
            border-radius: 999px;
            height: 10px;
            width: 10px;
        }}

        @media (max-width: 520px) {{
            .pie-chart-wrap {{
                align-items: flex-start;
                flex-direction: column;
                height: auto;
            }}
        }}
    </style>
    <div class="pie-chart-wrap">
        <div class="pie-chart">
                <svg viewBox="0 0 180 180" width="145" height="145" role="img">
                <circle cx="90" cy="90" fill="transparent" r="58" stroke="#e2e8f0" stroke-width="28" />
                {circle_html}
                <circle cx="90" cy="90" fill="#ffffff" r="42" />
                <text x="90" y="84" text-anchor="middle" fill="#0f2f4a" font-size="18" font-weight="700">{total:,}</text>
                <text x="90" y="104" text-anchor="middle" fill="#334155" font-size="11">customers</text>
            </svg>
        </div>
        <div class="pie-legend">{legend_html}</div>
    </div>
    """


def get_pie_counts(customers_df, column_name, top_n=4):
    """Return compact pie labels and values for a dataframe column."""
    counts = customers_df[column_name].fillna("Unknown").astype(str).value_counts()

    if len(counts) > top_n:
        top_counts = counts.head(top_n)
        other_count = counts.iloc[top_n:].sum()
        counts = pd.concat([top_counts, pd.Series({"Other": other_count})])

    return counts.index.tolist(), counts.astype(int).tolist()


def make_pie_section_html(title, labels, values):
    """Create one compact donut chart section."""
    total = sum(values)
    if total <= 0:
        return f"<div class='pie-panel'><h3>{escape(title)}</h3><p>No values available.</p></div>"

    colors = get_chart_theme()
    current_offset = 0
    circle_parts = []
    legend_parts = []

    for index, label in enumerate(labels):
        value = values[index]
        percent = (value / total) * 100
        color = colors[index % len(colors)]
        safe_label = escape(str(label))

        circle_parts.append(
            f"""
            <circle
                cx="90"
                cy="90"
                fill="transparent"
                r="58"
                stroke="{color}"
                stroke-dasharray="{percent:.4f} {100 - percent:.4f}"
                stroke-dashoffset="{-current_offset:.4f}"
                stroke-width="26"
                transform="rotate(-90 90 90)"
                pathLength="100"
            />
            """
        )
        legend_parts.append(
            f"""
            <div class="pie-legend-row">
                <span class="pie-legend-dot" style="background:{color};"></span>
                <span>{safe_label}</span>
                <strong>{value:,} ({percent:.1f}%)</strong>
            </div>
            """
        )
        current_offset += percent

    return f"""
    <div class="pie-panel">
        <h3>{escape(title)}</h3>
        <div class="pie-panel-body">
            <svg viewBox="0 0 180 180" width="126" height="126" role="img">
                <circle cx="90" cy="90" fill="transparent" r="58" stroke="#e2e8f0" stroke-width="26" />
                {"".join(circle_parts)}
                <circle cx="90" cy="90" fill="#ffffff" r="42" />
                <text x="90" y="86" text-anchor="middle" fill="#0f2f4a" font-size="17" font-weight="700">{total:,}</text>
                <text x="90" y="104" text-anchor="middle" fill="#334155" font-size="10">records</text>
            </svg>
            <div class="pie-legend">{"".join(legend_parts)}</div>
        </div>
    </div>
    """


def make_dual_pie_chart_html(first_title, first_labels, first_values, second_title, second_labels, second_values):
    """Create two compact pie charts inside one bordered chart area."""
    first_pie = make_pie_section_html(first_title, first_labels, first_values)
    second_pie = make_pie_section_html(second_title, second_labels, second_values)

    return f"""
    <style>
        body {{
            margin: 0;
            background: #ffffff;
            color: #0f172a;
            font-family: "Source Sans Pro", Arial, sans-serif;
        }}

        .dual-pie-card {{
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 14px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            box-sizing: border-box;
            display: grid;
            gap: 0.75rem;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            height: 330px;
            overflow: hidden;
            padding: 0.85rem;
            width: 100%;
        }}

        .pie-panel {{
            min-width: 0;
        }}

        .pie-panel h3 {{
            color: #0f172a;
            font-size: 0.9rem;
            font-weight: 800;
            line-height: 1.2;
            margin: 0 0 0.55rem;
        }}

        .pie-panel-body {{
            align-items: center;
            display: grid;
            gap: 0.65rem;
            grid-template-columns: 132px minmax(0, 1fr);
        }}

        .pie-legend {{
            display: flex;
            flex-direction: column;
            gap: 0.32rem;
            min-width: 0;
        }}

        .pie-legend-row {{
            align-items: center;
            color: #1e293b;
            display: grid;
            font-size: 0.76rem;
            gap: 0.35rem;
            grid-template-columns: 9px minmax(0, 1fr) auto;
            line-height: 1.2;
        }}

        .pie-legend-row span:nth-child(2) {{
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .pie-legend-row strong {{
            color: #0f172a;
            font-weight: 800;
            white-space: nowrap;
        }}

        .pie-legend-dot {{
            border-radius: 999px;
            height: 8px;
            width: 8px;
        }}

        @media (max-width: 760px) {{
            .dual-pie-card {{
                grid-template-columns: 1fr;
                height: auto;
            }}
        }}
    </style>
    <div class="dual-pie-card">
        {first_pie}
        {second_pie}
    </div>
    """


def get_multi_field_bar_data(customers_df):
    """Prepare grouped bar chart data from several numeric customer fields."""
    numeric_fields = [
        field
        for field in ["total_orders", "loyalty_points", "monthly_income"]
        if field in customers_df.columns
    ]

    if not numeric_fields:
        return pd.DataFrame()

    if "membership_status" in customers_df.columns:
        group_field = "membership_status"
        group_label = "Membership Status"
    elif "state" in customers_df.columns:
        group_field = "state"
        group_label = "State"
    else:
        group_field = "all_customers"
        group_label = "Customer Group"
        customers_df = customers_df.copy()
        customers_df[group_field] = "All Customers"

    chart_df = customers_df.copy()
    chart_df[group_field] = chart_df[group_field].fillna("Unknown")

    for field in numeric_fields:
        chart_df[field] = get_numeric_column(chart_df, field)

    grouped_df = chart_df.groupby(group_field, as_index=False)[numeric_fields].sum()
    grouped_df = grouped_df.melt(
        id_vars=group_field,
        value_vars=numeric_fields,
        var_name="Metric",
        value_name="Value",
    )
    grouped_df = grouped_df.rename(columns={group_field: group_label})
    grouped_df["Metric"] = grouped_df["Metric"].str.replace("_", " ").str.title()

    return grouped_df


def show_kpi_cards(customers_df):
    """Display customer summary numbers at the top of the dashboard."""
    render_section_header("Executive summary", "Customer KPIs")

    total_customers = len(customers_df)
    monthly_income = get_numeric_column(customers_df, "monthly_income")
    total_orders = get_numeric_column(customers_df, "total_orders")
    loyalty_points = get_numeric_column(customers_df, "loyalty_points")

    average_monthly_income = monthly_income.mean() if not monthly_income.empty else 0
    total_orders_sum = int(total_orders.sum()) if not total_orders.empty else 0
    total_loyalty_points = int(loyalty_points.sum()) if not loyalty_points.empty else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", total_customers)
    col2.metric("Average Monthly Income", f"{average_monthly_income:,.2f}")
    col3.metric("Total Orders", f"{total_orders_sum:,}")
    col4.metric("Total Loyalty Points", f"{total_loyalty_points:,}")


def apply_numeric_range_slicer(customers_df, filtered_df, column_name, label, number_format):
    """Apply a numeric range slider when the column has useful values."""
    if column_name not in customers_df.columns:
        st.caption(f"{label} slicer needs a {column_name} column.")
        return filtered_df

    full_values = get_numeric_column(customers_df, column_name)
    min_value = float(full_values.min()) if not full_values.empty else 0
    max_value = float(full_values.max()) if not full_values.empty else 0

    if min_value >= max_value:
        st.caption(f"{label} slicer needs more than one value.")
        return filtered_df

    use_integer_slider = full_values.dropna().mod(1).eq(0).all()

    if use_integer_slider:
        slider_min = int(min_value)
        slider_max = int(max_value)
        selected_range = st.slider(
            label,
            slider_min,
            slider_max,
            (slider_min, slider_max),
        )
    else:
        selected_range = st.slider(
            label,
            min_value,
            max_value,
            (min_value, max_value),
            format=number_format,
        )

    filtered_values = get_numeric_column(filtered_df, column_name)
    return filtered_df[
        filtered_values.between(
            selected_range[0],
            selected_range[1],
        )
    ]


def get_slicer_options(customers_df, column_name):
    """Return clean slicer options, including Unknown for blank values."""
    if column_name not in customers_df.columns:
        return []

    options = customers_df[column_name].fillna("Unknown").astype(str).unique()
    return sorted(options)


def apply_text_multiselect_slicer(customers_df, filtered_df, column_name, label):
    """Apply a text slicer with all values selected by default."""
    if column_name not in customers_df.columns:
        st.caption(f"{label} slicer needs a {column_name} column.")
        return filtered_df

    options = get_slicer_options(customers_df, column_name)
    selected_options = st.multiselect(label, options, default=options)

    return filtered_df[
        filtered_df[column_name].fillna("Unknown").astype(str).isin(selected_options)
    ]


def find_status_column(customers_df):
    """Find a likely active/inactive status column in the customers table."""
    possible_status_columns = [
        "status",
        "customer_status",
        "account_status",
        "is_active",
        "active_status",
    ]

    for column_name in possible_status_columns:
        if column_name in customers_df.columns:
            return column_name

    return None


def find_age_column(customers_df):
    """Find a likely age column in the customers table."""
    possible_age_columns = ["age", "customer_age"]

    for column_name in possible_age_columns:
        if column_name in customers_df.columns:
            return column_name

    return None


def apply_dashboard_filters(customers_df):
    """Add dashboard slicers and return the filtered dataframe."""
    filtered_df = customers_df.copy()

    with st.sidebar:
        st.divider()
        st.caption("Dashboard slicers")

        filtered_df = apply_text_multiselect_slicer(
            customers_df,
            filtered_df,
            "state",
            "State",
        )

        filtered_df = apply_numeric_range_slicer(
            customers_df,
            filtered_df,
            "total_orders",
            "Total Orders Range",
            "%d",
        )

        status_column = find_status_column(customers_df)
        if status_column:
            filtered_df = apply_text_multiselect_slicer(
                customers_df,
                filtered_df,
                status_column,
                "Status",
            )

        age_column = find_age_column(customers_df)
        if age_column:
            filtered_df = apply_numeric_range_slicer(
                customers_df,
                filtered_df,
                age_column,
                "Age Range",
                "%d",
            )
        else:
            st.caption("Age slicer needs an age column.")

        filtered_df = apply_numeric_range_slicer(
            customers_df,
            filtered_df,
            "monthly_income",
            "Monthly Income Range",
            "%.2f",
        )

        filtered_df = apply_numeric_range_slicer(
            customers_df,
            filtered_df,
            "loyalty_points",
            "Loyalty Points Range",
            "%d",
        )

        st.metric("Filtered rows", len(filtered_df))

    return filtered_df


def show_customer_charts(customers_df):
    """Display executive visuals from the customers dataframe."""
    render_section_header("Performance visuals", "Customer Analytics")

    if customers_df.empty:
        st.info("Add customer data to see charts.")
        return

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        show_chart_title("Column Chart: Customers by State")
        if "state" in customers_df.columns:
            state_counts = (
                customers_df["state"]
                .fillna("Unknown")
                .value_counts()
                .reset_index()
            )
            state_counts.columns = ["State", "Customers"]
            column_spec = {
                "mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
                "encoding": {
                    "x": {
                        "field": "State",
                        "type": "nominal",
                        "sort": "-y",
                        "axis": {"labelAngle": -30},
                        "title": "State",
                    },
                    "y": {
                        "field": "Customers",
                        "type": "quantitative",
                        "title": "Customers",
                    },
                    "color": {"value": "#17446b"},
                    "tooltip": [
                        {"field": "State", "type": "nominal"},
                        {"field": "Customers", "type": "quantitative"},
                    ],
                },
                "height": 260,
            }
            show_vega_chart(column_spec, state_counts)
        else:
            st.info("Add a state column to show customers by state.")

    with chart_col2:
        show_chart_title("Pie Charts: Membership and State")
        if "membership_status" in customers_df.columns and "state" in customers_df.columns:
            membership_labels, membership_values = get_pie_counts(
                customers_df,
                "membership_status",
                top_n=4,
            )
            state_labels, state_values = get_pie_counts(
                customers_df,
                "state",
                top_n=4,
            )
            pie_chart = make_dual_pie_chart_html(
                "Membership Status",
                membership_labels,
                membership_values,
                "State Distribution",
                state_labels,
                state_values,
            )
            components.html(pie_chart, height=330)
        else:
            st.info("Add membership_status and state columns to show both pie charts.")

    bar_col, pivot_col = st.columns([1.35, 1])

    with bar_col:
        show_chart_title("Multi-Field Bar Chart: Value Metrics")
        multi_bar_df = get_multi_field_bar_data(customers_df)

        if multi_bar_df.empty:
            st.info("Add total_orders, loyalty_points, or monthly_income to show this chart.")
        else:
            category_column = (
                "Membership Status"
                if "Membership Status" in multi_bar_df.columns
                else "State"
                if "State" in multi_bar_df.columns
                else "Customer Group"
            )
            multi_bar_spec = {
                "mark": {"type": "bar", "cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3},
                "encoding": {
                    "x": {
                        "field": category_column,
                        "type": "nominal",
                        "axis": {"labelAngle": -20},
                        "title": category_column,
                    },
                    "xOffset": {"field": "Metric"},
                    "y": {
                        "field": "Value",
                        "type": "quantitative",
                        "title": "Total Value",
                    },
                    "color": {
                        "field": "Metric",
                        "type": "nominal",
                        "scale": {"range": get_chart_theme()},
                        "title": "Metric",
                    },
                    "tooltip": [
                        {"field": category_column, "type": "nominal"},
                        {"field": "Metric", "type": "nominal"},
                        {"field": "Value", "type": "quantitative", "format": ",.2f"},
                    ],
                },
                "height": 260,
            }
            show_vega_chart(multi_bar_spec, multi_bar_df)

    with pivot_col:
        show_chart_title("Pivot Table: State by Membership")
        if "state" in customers_df.columns and "membership_status" in customers_df.columns:
            pivot_df = customers_df.copy()
            pivot_df["state"] = pivot_df["state"].fillna("Unknown")
            pivot_df["membership_status"] = pivot_df["membership_status"].fillna("Unknown")

            if "total_orders" in pivot_df.columns:
                pivot_df["total_orders"] = get_numeric_column(pivot_df, "total_orders")
                pivot_values = "total_orders"
                pivot_function = "sum"
            else:
                pivot_df["customer_count"] = 1
                pivot_values = "customer_count"
                pivot_function = "sum"

            pivot_table = pd.pivot_table(
                pivot_df,
                index="state",
                columns="membership_status",
                values=pivot_values,
                aggfunc=pivot_function,
                fill_value=0,
                margins=True,
                margins_name="Total",
            )
            st.dataframe(pivot_table, use_container_width=True)
        else:
            st.info("Add state and membership_status columns to show the pivot table.")


def show_comparison_charts(customers_df):
    """Display four colorful comparison analysis charts."""
    render_section_header("Comparison analysis", "Customer Comparison Dashboard")

    if customers_df.empty:
        st.info("Adjust the slicers or add customer data to see comparison charts.")
        return

    map_col1, map_col2 = st.columns(2)

    with map_col1:
        show_chart_title("Malaysia Map: Average Monthly Income")
        income_map_df = get_state_average_map_data(customers_df, "monthly_income")
        if income_map_df.empty:
            st.info("Add state and monthly_income columns for this Malaysia map.")
        else:
            income_map = make_malaysia_choropleth_html(
                income_map_df,
                "Average Monthly Income by State",
                "Average income",
                "RM {:,.2f}",
            )
            components.html(income_map, height=350)

    with map_col2:
        show_chart_title("Malaysia Map: Average Loyalty Points")
        loyalty_map_df = get_state_average_map_data(customers_df, "loyalty_points")
        if loyalty_map_df.empty:
            st.info("Add state and loyalty_points columns for this Malaysia map.")
        else:
            loyalty_map = make_malaysia_choropleth_html(
                loyalty_map_df,
                "Average Loyalty Points by State",
                "Average loyalty points",
                "{:,.0f}",
            )
            components.html(loyalty_map, height=350)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        show_chart_title("Grouped Bar: Orders vs Loyalty by State")
        if all(column in customers_df.columns for column in ["state", "total_orders", "loyalty_points"]):
            grouped_df = customers_df.copy()
            grouped_df["state"] = grouped_df["state"].fillna("Unknown")
            grouped_df["total_orders"] = get_numeric_column(grouped_df, "total_orders")
            grouped_df["loyalty_points"] = get_numeric_column(grouped_df, "loyalty_points")
            grouped_df = grouped_df.groupby("state", as_index=False)[
                ["total_orders", "loyalty_points"]
            ].sum()
            grouped_df = grouped_df.melt(
                id_vars="state",
                value_vars=["total_orders", "loyalty_points"],
                var_name="Metric",
                value_name="Value",
            )
            grouped_df["Metric"] = grouped_df["Metric"].str.replace("_", " ").str.title()

            grouped_bar_spec = {
                "mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
                "encoding": {
                    "x": {
                        "field": "state",
                        "type": "nominal",
                        "sort": "-y",
                        "axis": {"labelAngle": -25},
                        "title": "State",
                    },
                    "xOffset": {"field": "Metric"},
                    "y": {"field": "Value", "type": "quantitative", "title": "Total"},
                    "color": {
                        "field": "Metric",
                        "type": "nominal",
                        "scale": {"range": ["#2563eb", "#f59e0b"]},
                        "legend": {
                            "orient": "bottom",
                            "direction": "horizontal",
                            "title": "Metric",
                        },
                    },
                    "tooltip": [
                        {"field": "state", "type": "nominal", "title": "State"},
                        {"field": "Metric", "type": "nominal"},
                        {"field": "Value", "type": "quantitative", "format": ",.0f"},
                    ],
                },
                "height": 300,
            }
            show_vega_chart(grouped_bar_spec, grouped_df)
        else:
            st.info("Add state, total_orders, and loyalty_points columns for this chart.")

    with chart_col2:
        show_chart_title("Stacked Bar: Membership Mix by State")
        if all(column in customers_df.columns for column in ["state", "membership_status"]):
            stacked_df = (
                customers_df.assign(
                    state=customers_df["state"].fillna("Unknown"),
                    membership_status=customers_df["membership_status"].fillna("Unknown"),
                )
                .groupby(["state", "membership_status"], as_index=False)
                .size()
            )
            stacked_df = stacked_df.rename(columns={"size": "Customers"})

            stacked_bar_spec = {
                "mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
                "encoding": {
                    "x": {
                        "field": "state",
                        "type": "nominal",
                        "axis": {"labelAngle": -25},
                        "title": "State",
                    },
                    "y": {"field": "Customers", "type": "quantitative", "title": "Customers"},
                    "color": {
                        "field": "membership_status",
                        "type": "nominal",
                        "scale": {"range": get_chart_theme()},
                        "legend": {
                            "orient": "bottom",
                            "direction": "horizontal",
                            "title": "Membership",
                        },
                    },
                    "tooltip": [
                        {"field": "state", "type": "nominal", "title": "State"},
                        {"field": "membership_status", "type": "nominal", "title": "Membership"},
                        {"field": "Customers", "type": "quantitative"},
                    ],
                },
                "height": 300,
            }
            show_vega_chart(stacked_bar_spec, stacked_df)
        else:
            st.info("Add state and membership_status columns for this chart.")

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        show_chart_title("Bubble Chart: Income vs Orders")
        if all(
            column in customers_df.columns
            for column in ["monthly_income", "total_orders", "loyalty_points"]
        ):
            bubble_df = customers_df.copy()
            bubble_df["monthly_income"] = get_numeric_column(bubble_df, "monthly_income")
            bubble_df["total_orders"] = get_numeric_column(bubble_df, "total_orders")
            bubble_df["loyalty_points"] = get_numeric_column(bubble_df, "loyalty_points")

            if "membership_status" in bubble_df.columns:
                color_field = "membership_status"
                bubble_df[color_field] = bubble_df[color_field].fillna("Unknown")
            else:
                color_field = "Customer Group"
                bubble_df[color_field] = "All Customers"

            bubble_spec = {
                "mark": {"type": "circle", "opacity": 0.78, "stroke": "#ffffff", "strokeWidth": 1},
                "encoding": {
                    "x": {
                        "field": "monthly_income",
                        "type": "quantitative",
                        "title": "Monthly Income",
                    },
                    "y": {
                        "field": "total_orders",
                        "type": "quantitative",
                        "title": "Total Orders",
                    },
                    "size": {
                        "field": "loyalty_points",
                        "type": "quantitative",
                        "title": "Loyalty Points",
                        "scale": {"range": [70, 900]},
                        "legend": {
                            "orient": "bottom",
                            "direction": "horizontal",
                            "title": "Loyalty Points",
                            "values": [0, 200, 400, 600],
                        },
                    },
                    "color": {
                        "field": color_field,
                        "type": "nominal",
                        "scale": {"range": get_chart_theme()},
                        "legend": {
                            "orient": "bottom",
                            "direction": "horizontal",
                            "title": "Status",
                        },
                    },
                    "tooltip": [
                        {"field": "monthly_income", "type": "quantitative", "format": ",.2f"},
                        {"field": "total_orders", "type": "quantitative", "format": ",.0f"},
                        {"field": "loyalty_points", "type": "quantitative", "format": ",.0f"},
                        {"field": color_field, "type": "nominal"},
                    ],
                },
                "height": 300,
            }
            show_vega_chart(bubble_spec, bubble_df)
        else:
            st.info("Add monthly_income, total_orders, and loyalty_points columns for this chart.")

    with chart_col4:
        show_chart_title("Heatmap: State vs Membership Orders")
        if all(
            column in customers_df.columns
            for column in ["state", "membership_status", "total_orders"]
        ):
            heatmap_df = customers_df.copy()
            heatmap_df["state"] = heatmap_df["state"].fillna("Unknown")
            heatmap_df["membership_status"] = heatmap_df["membership_status"].fillna("Unknown")
            heatmap_df["total_orders"] = get_numeric_column(heatmap_df, "total_orders")
            heatmap_df = heatmap_df.groupby(
                ["state", "membership_status"],
                as_index=False,
            )["total_orders"].sum()

            heatmap_spec = {
                "mark": {"type": "rect", "cornerRadius": 3},
                "encoding": {
                    "x": {
                        "field": "membership_status",
                        "type": "nominal",
                        "title": "Membership",
                    },
                    "y": {
                        "field": "state",
                        "type": "nominal",
                        "title": "State",
                    },
                    "color": {
                        "field": "total_orders",
                        "type": "quantitative",
                        "title": "Orders",
                        "scale": {"range": ["#eff6ff", "#60a5fa", "#7c3aed", "#be123c"]},
                        "legend": {
                            "orient": "bottom",
                            "direction": "horizontal",
                            "title": "Orders",
                        },
                    },
                    "tooltip": [
                        {"field": "state", "type": "nominal", "title": "State"},
                        {"field": "membership_status", "type": "nominal", "title": "Membership"},
                        {"field": "total_orders", "type": "quantitative", "format": ",.0f"},
                    ],
                },
                "height": 300,
            }
            show_vega_chart(heatmap_spec, heatmap_df)
        else:
            st.info("Add state, membership_status, and total_orders columns for this chart.")


def show_add_customer_form(editable_fields):
    """Create a new customer row."""
    render_section_header("Create", "Add Customer")

    with st.form("add_customer_form"):
        new_customer = {}

        for field in editable_fields:
            new_customer[field] = st.text_input(field.replace("_", " ").title())

        submitted = st.form_submit_button("Add Customer")

        if submitted:
            if not new_customer.get("customer_code"):
                st.warning("Customer code is required.")
                return

            try:
                supabase.table("customers").insert(new_customer).execute()
                refresh_with_message("Customer added successfully.")
            except Exception as error:
                st.error(f"Could not add customer: {error}")


def show_customers_dataframe(customers_df):
    """Display all customers in a dataframe."""
    render_section_header("Customer records", "View Customers")

    if customers_df.empty:
        st.info("The customers table is empty.")
    else:
        st.dataframe(customers_df, use_container_width=True)


def show_update_customer_form(customers_df, editable_fields):
    """Update an existing customer row using customer_code."""
    render_section_header("Edit", "Update Customer")

    if customers_df.empty:
        st.info("Add a customer first before updating.")
        return

    if "customer_code" not in customers_df.columns:
        st.warning("The customers table needs a customer_code column for updates.")
        return

    customer_codes = customers_df["customer_code"].dropna().astype(str).tolist()
    if not customer_codes:
        st.warning("No customer codes are available for updates.")
        return

    selected_code = st.selectbox("Select customer code to update", customer_codes)

    selected_customer = customers_df[
        customers_df["customer_code"].astype(str) == selected_code
    ].iloc[0]

    with st.form("update_customer_form"):
        updated_customer = {}

        for field in editable_fields:
            current_value = selected_customer.get(field, "")
            updated_customer[field] = st.text_input(
                field.replace("_", " ").title(),
                value="" if pd.isna(current_value) else str(current_value),
            )

        submitted = st.form_submit_button("Update Customer")

        if submitted:
            if not updated_customer.get("customer_code"):
                st.warning("Customer code is required.")
                return

            try:
                supabase.table("customers").update(updated_customer).eq(
                    "customer_code",
                    selected_code,
                ).execute()
                refresh_with_message("Customer updated successfully.")
            except Exception as error:
                st.error(f"Could not update customer: {error}")


def show_delete_customer_form(customers_df):
    """Delete an existing customer row using customer_code."""
    render_section_header("Remove", "Delete Customer")

    if customers_df.empty:
        st.info("Add a customer first before deleting.")
        return

    if "customer_code" not in customers_df.columns:
        st.warning("The customers table needs a customer_code column for deletes.")
        return

    customer_codes = customers_df["customer_code"].dropna().astype(str).tolist()
    if not customer_codes:
        st.warning("No customer codes are available for deletes.")
        return

    with st.form("delete_customer_form"):
        selected_code = st.selectbox("Select customer code to delete", customer_codes)
        confirm_delete = st.checkbox("I understand this will delete the customer.")
        submitted = st.form_submit_button("Delete Customer")

        if submitted:
            if not confirm_delete:
                st.warning("Please tick the checkbox before deleting.")
                return

            try:
                supabase.table("customers").delete().eq(
                    "customer_code",
                    selected_code,
                ).execute()
                refresh_with_message("Customer deleted successfully.")
            except Exception as error:
                st.error(f"Could not delete customer: {error}")


@st.cache_resource
def load_existing_ml_model(model_path_text, modified_time):
    """Load an existing ML model and refresh the cache when the file changes."""
    import joblib

    return joblib.load(model_path_text)


def load_ml_model(model_filename):
    """Load a saved ML model from the models folder."""
    model_path = MODELS_DIR / model_filename

    # Do not cache missing models. This lets the app detect newly trained files.
    if not model_path.exists():
        return None, MODEL_NOT_FOUND_MESSAGE

    try:
        model = load_existing_ml_model(str(model_path), model_path.stat().st_mtime)
        return model, None
    except Exception as error:
        return None, f"Could not load model: {error}"


def get_model_feature_columns(model):
    """Find the original feature columns expected by a trained model pipeline."""
    if hasattr(model, "feature_names_in_"):
        return remove_identifier_features(list(model.feature_names_in_))

    preprocessor = getattr(model, "named_steps", {}).get("preprocessor")
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        return remove_identifier_features(list(preprocessor.feature_names_in_))

    if preprocessor is not None and hasattr(preprocessor, "transformers_"):
        columns = []
        for _, _, transformer_columns in preprocessor.transformers_:
            if isinstance(transformer_columns, str) and transformer_columns == "drop":
                continue
            if isinstance(transformer_columns, slice):
                continue
            if isinstance(transformer_columns, str):
                columns.append(transformer_columns)
            else:
                columns.extend([column for column in transformer_columns])
        return remove_identifier_features(list(dict.fromkeys(columns)))

    return []


def remove_identifier_features(feature_columns):
    """Remove identifier/contact columns that should not be manually predicted."""
    return [
        column
        for column in feature_columns
        if str(column).lower() not in ML_IDENTIFIER_FIELDS
    ]


def is_numeric_feature(customers_df, feature_name):
    """Decide whether a feature should use a number input."""
    if feature_name in customers_df.columns:
        series = customers_df[feature_name]
        if pd.api.types.is_numeric_dtype(series):
            return True

        converted = pd.to_numeric(series, errors="coerce")
        return converted.notna().sum() >= max(1, series.notna().sum() * 0.8)

    numeric_words = [
        "age",
        "amount",
        "count",
        "days",
        "frequency",
        "income",
        "loyalty",
        "order",
        "point",
        "postcode",
        "purchase",
        "revenue",
        "sale",
        "total",
    ]
    return any(word in feature_name.lower() for word in numeric_words)


def get_default_numeric_value(customers_df, feature_name):
    """Choose a friendly default value for numeric prediction inputs."""
    if feature_name not in customers_df.columns:
        return 0.0

    values = pd.to_numeric(customers_df[feature_name], errors="coerce").dropna()
    if values.empty:
        return 0.0

    return float(values.median())


def get_categorical_options(customers_df, feature_name):
    """Get existing category values from the customer data for select boxes."""
    if feature_name not in customers_df.columns:
        return []

    options = (
        customers_df[feature_name]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )
    return options[:100]


def render_prediction_inputs(model, customers_df, form_key):
    """Create Streamlit inputs based on the model's original feature columns."""
    feature_columns = get_model_feature_columns(model)

    if not feature_columns:
        st.warning("Could not detect model feature columns.")
        return None

    input_values = {}
    columns = st.columns(2)

    for index, feature_name in enumerate(feature_columns):
        with columns[index % 2]:
            label = str(feature_name).replace("_", " ").title()

            if is_numeric_feature(customers_df, feature_name):
                input_values[feature_name] = st.number_input(
                    label,
                    value=get_default_numeric_value(customers_df, feature_name),
                    key=f"{form_key}_{feature_name}",
                )
            else:
                options = get_categorical_options(customers_df, feature_name)
                if options:
                    input_values[feature_name] = st.selectbox(
                        label,
                        options,
                        key=f"{form_key}_{feature_name}",
                    )
                else:
                    input_values[feature_name] = st.text_input(
                        label,
                        key=f"{form_key}_{feature_name}",
                    )

    return input_values


def predict_probability_if_available(model, input_df):
    """Return high-risk probability when the model supports probabilities."""
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(input_df)
    if probabilities.shape[1] < 2:
        return None

    high_risk_index = 1
    classes = getattr(model, "classes_", None)
    if classes is not None:
        for index, class_name in enumerate(classes):
            if is_high_churn_prediction(class_name):
                high_risk_index = index
                break

    return float(probabilities[0][high_risk_index])


def is_high_churn_prediction(prediction):
    """Convert common model outputs into a high-risk True/False value."""
    if isinstance(prediction, str):
        return prediction.strip().lower() in {"1", "yes", "true", "high", "high risk"}

    return bool(prediction)


def interpret_churn_result(is_high_risk, probability):
    """Create a short business explanation for churn predictions."""
    if is_high_risk:
        if probability is None:
            return "This customer may need retention attention. Consider a personal offer, follow-up, or loyalty incentive."
        return "This customer may need retention attention. Prioritize follow-up if the probability is high."

    return "This customer appears more stable. Keep engagement consistent and continue monitoring purchase behaviour."


def interpret_total_order_prediction(predicted_orders):
    """Create a short business explanation for total order predictions."""
    if predicted_orders >= 20:
        return "This customer is likely to be a strong repeat buyer. Consider VIP treatment or bundle offers."
    if predicted_orders >= 8:
        return "This customer has moderate order potential. Targeted promotions may increase repeat purchases."
    return "This customer may need more nurturing. Consider onboarding messages, discounts, or reminders."


def show_churn_prediction_section(customers_df):
    """Display the churn risk prediction tool."""
    render_section_header("Section 1", "Churn Risk Prediction")

    model, error_message = load_ml_model("churn_model.pkl")
    if model is None:
        st.warning(error_message or MODEL_NOT_FOUND_MESSAGE)
        return

    st.write(
        "Enter customer profile and behaviour details to estimate whether the "
        "customer has low or high churn risk."
    )

    with st.form("churn_prediction_form"):
        input_values = render_prediction_inputs(model, customers_df, "churn")
        submitted = st.form_submit_button("Predict Churn Risk")

    if submitted and input_values is not None:
        input_df = pd.DataFrame([input_values])
        try:
            prediction = model.predict(input_df)[0]
            probability = predict_probability_if_available(model, input_df)
        except Exception as error:
            st.error(
                "Could not predict churn risk with the current model. "
                f"Please retrain the churn model. Details: {error}"
            )
            return

        is_high_risk = is_high_churn_prediction(prediction)
        risk_label = "High Risk" if is_high_risk else "Low Risk"

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Churn Risk", risk_label)
        with col2:
            if probability is not None:
                st.metric("High Risk Probability", f"{probability:.1%}")
            else:
                st.metric("Probability", "Not available")

        st.info(interpret_churn_result(is_high_risk, probability))


def show_total_order_prediction_section(customers_df):
    """Display the total order prediction tool."""
    render_section_header("Section 2", "Total Order Prediction")

    model, error_message = load_ml_model("total_order_model.pkl")
    if model is None:
        st.warning(error_message or MODEL_NOT_FOUND_MESSAGE)
        return

    st.write(
        "Enter customer profile and behaviour details to estimate the expected "
        "total number of customer orders."
    )

    with st.form("total_order_prediction_form"):
        input_values = render_prediction_inputs(model, customers_df, "total_order")
        submitted = st.form_submit_button("Predict Total Order")

    if submitted and input_values is not None:
        input_df = pd.DataFrame([input_values])
        try:
            predicted_orders = float(model.predict(input_df)[0])
            predicted_orders = max(0, predicted_orders)
        except Exception as error:
            st.error(
                "Could not predict total orders with the current model. "
                f"Please retrain the total order model. Details: {error}"
            )
            return

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Predicted Total Orders", f"{predicted_orders:,.2f}")
        with col2:
            st.info(interpret_total_order_prediction(predicted_orders))


def show_ml_prediction_page(customers_df):
    """Display ML prediction tools for churn risk and total orders."""
    render_page_header(
        "ML Prediction",
        "Use trained machine learning models to estimate churn risk and expected total orders.",
    )

    show_churn_prediction_section(customers_df)
    st.markdown("<br>", unsafe_allow_html=True)
    show_total_order_prediction_section(customers_df)


def load_evaluation_csv(filename):
    """Load a model evaluation CSV from the models folder."""
    file_path = MODELS_DIR / filename

    if not file_path.exists():
        return None

    try:
        return pd.read_csv(file_path)
    except Exception as error:
        st.error(f"Could not read {filename}: {error}")
        return None


def run_model_training(module_name):
    """
    Run one of the ML training scripts from the Streamlit app.

    The training files live inside the ml folder and already contain the full
    beginner-friendly training pipeline, including loading the latest customer
    data and saving model/evaluation files.
    """
    ml_path = str(ML_DIR)
    if ml_path not in sys.path:
        sys.path.insert(0, ml_path)

    training_module = importlib.import_module(module_name)
    training_module.train()

    # Clear cached models so the prediction page uses the latest trained files.
    load_existing_ml_model.clear()


def show_retrain_button(button_label, module_name, success_message):
    """Show a retrain button and run the requested training pipeline."""
    retrained = False

    if st.button(button_label, type="primary"):
        with st.spinner("Retraining model with latest customer data..."):
            try:
                run_model_training(module_name)
                retrained = True
            except Exception as error:
                st.error(f"Retraining failed: {error}")

        if retrained:
            st.success(success_message)

    return retrained


def format_metric_table(df, metric_columns):
    """Return a display-friendly copy of an evaluation table."""
    display_df = df.copy()
    available_metrics = [
        column for column in metric_columns if column in display_df.columns
    ]

    for column in available_metrics:
        display_df[column] = pd.to_numeric(display_df[column], errors="coerce")

    return display_df


def get_best_model_name(df, sort_columns, ascending):
    """Find the best model using the same ranking logic as training."""
    available_sort_columns = [column for column in sort_columns if column in df.columns]
    if not available_sort_columns or "model" not in df.columns:
        return None

    sort_order = ascending[: len(available_sort_columns)]
    ranked_df = df.sort_values(
        by=available_sort_columns,
        ascending=sort_order,
        na_position="last",
    )

    if ranked_df.empty:
        return None

    return ranked_df.iloc[0]["model"]


def show_best_model_metric(best_model_name):
    """Display the selected best model clearly."""
    if best_model_name:
        st.metric("Best Selected Model", best_model_name)
    else:
        st.info("Best model could not be determined from this evaluation file.")


def show_metric_bar_chart(df, metric_columns, title):
    """Show a compact bar chart for the available model metrics."""
    if "model" not in df.columns:
        return

    chart_columns = [column for column in metric_columns if column in df.columns]
    if not chart_columns:
        return

    chart_df = df[["model"] + chart_columns].copy()
    for column in chart_columns:
        chart_df[column] = pd.to_numeric(chart_df[column], errors="coerce")

    chart_df = chart_df.set_index("model")
    st.caption(title)
    st.bar_chart(chart_df)


def highlight_selected_model_row(row, best_model_name):
    """Highlight the row for the model selected by the training logic."""
    if best_model_name and row.get("model") == best_model_name:
        return ["background-color: #fef9c3"] * len(row)

    return [""] * len(row)


def show_feature_importance_chart(filename, title):
    """Show a feature importance chart if the CSV file exists."""
    importance_df = load_evaluation_csv(filename)
    if importance_df is None:
        st.info("Feature importance file not found yet.")
        return

    value_column = None
    for candidate in ["importance", "absolute_coefficient"]:
        if candidate in importance_df.columns:
            value_column = candidate
            break

    if value_column is None or "feature" not in importance_df.columns:
        st.info("Feature importance data is not available in the expected format.")
        return

    importance_df[value_column] = pd.to_numeric(
        importance_df[value_column],
        errors="coerce",
    )
    chart_df = (
        importance_df.dropna(subset=[value_column])
        .sort_values(value_column, ascending=False)
        .head(15)
        .set_index("feature")[[value_column]]
    )

    if chart_df.empty:
        st.info("Feature importance data is empty.")
        return

    st.caption(title)
    st.bar_chart(chart_df)


def show_churn_model_evaluation():
    """Display churn model comparison and feature importance."""
    render_section_header("Churn model", "Churn Model Evaluation")

    retrained = show_retrain_button(
        "Retrain Churn Model",
        "train_churn_model",
        "Churn model retrained successfully.",
    )

    evaluation_df = load_evaluation_csv("churn_evaluation.csv")
    if evaluation_df is None:
        st.warning("Churn evaluation file not found. Please train the churn model first.")
        return

    metric_columns = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    display_df = format_metric_table(evaluation_df, metric_columns)
    best_model_name = get_best_model_name(
        display_df,
        ["f1_score", "roc_auc", "accuracy"],
        [False, False, False],
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        show_best_model_metric(best_model_name)
    with col2:
        st.write("Comparison using Accuracy, Precision, Recall, F1-score, and ROC-AUC.")

    if retrained:
        st.caption("Latest metrics from the newly saved churn evaluation file.")

    st.dataframe(
        display_df.style.apply(
            highlight_selected_model_row,
            axis=1,
            best_model_name=best_model_name,
        ).highlight_max(
            subset=[column for column in metric_columns if column in display_df.columns],
            color="#dcfce7",
        ),
        use_container_width=True,
    )
    show_metric_bar_chart(display_df, metric_columns, "Churn model metric comparison")
    show_feature_importance_chart(
        "churn_feature_importance.csv",
        "Top churn model feature importance",
    )


def show_total_order_model_evaluation():
    """Display total order model comparison and feature importance."""
    render_section_header("Total order model", "Total Order Model Evaluation")

    retrained = show_retrain_button(
        "Retrain Total Order Model",
        "train_orders_model",
        "Total order model retrained successfully.",
    )

    evaluation_df = load_evaluation_csv("total_order_evaluation.csv")
    if evaluation_df is None:
        st.warning(
            "Total order evaluation file not found. Please train the total order model first."
        )
        return

    metric_columns = ["mae", "rmse", "r2_score", "cv_rmse"]
    display_df = format_metric_table(evaluation_df, metric_columns)
    best_model_name = get_best_model_name(
        display_df,
        ["rmse", "r2_score"],
        [True, False],
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        show_best_model_metric(best_model_name)
    with col2:
        st.write("Comparison using MAE, RMSE, R2 score, and cross-validation RMSE.")

    if retrained:
        st.caption("Latest metrics from the newly saved total order evaluation file.")

    st.dataframe(
        display_df.style.apply(
            highlight_selected_model_row,
            axis=1,
            best_model_name=best_model_name,
        ).highlight_min(
            subset=[column for column in ["mae", "rmse", "cv_rmse"] if column in display_df.columns],
            color="#dcfce7",
        ).highlight_max(
            subset=[column for column in ["r2_score"] if column in display_df.columns],
            color="#dbeafe",
        ),
        use_container_width=True,
    )
    show_metric_bar_chart(
        display_df,
        metric_columns,
        "Total order model metric comparison",
    )
    show_feature_importance_chart(
        "total_order_feature_importance.csv",
        "Top total order model feature importance",
    )


def show_ml_model_evaluation_page():
    """Display evaluation results for trained ML models."""
    render_page_header(
        "ML Model Evaluation",
        "Compare model techniques, review selected winners, and inspect important features.",
    )

    show_churn_model_evaluation()
    st.markdown("<br>", unsafe_allow_html=True)
    show_total_order_model_evaluation()


def show_sidebar(customers_df):
    """Display app navigation and a few compact sidebar details."""
    with st.sidebar:
        st.markdown("<div class='brand-logo'>AFJ</div>", unsafe_allow_html=True)
        st.markdown("<div class='brand-name'>afj_analytics</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='brand-tagline'>Cultivating data-driven culture</div>",
            unsafe_allow_html=True,
        )
        st.title("Customer Analytics")
        st.caption("Executive dashboard")

        selected_page = st.radio(
            "Navigation",
            [
                "Dashboard",
                "Comparison Dashboard",
                "Customer Data",
                "CRUD Management",
                "ML Prediction",
                "ML Model Evaluation",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("Dataset status")
        st.metric("Total rows", len(customers_df))
        st.metric("Columns", len(customers_df.columns))

    return selected_page


def show_dashboard_page(customers_df):
    """Display the executive dashboard page."""
    render_page_header(
        "EXECUTIVE CUSTOMER DASHBOARD",
        "A clean overview of customer volume, value, loyalty, and ordering behavior.",
    )

    show_kpi_cards(customers_df)
    st.markdown("<br>", unsafe_allow_html=True)

    show_customer_charts(customers_df)


def show_comparison_dashboard_page(customers_df):
    """Display a dashboard focused on side-by-side comparison analysis."""
    render_page_header(
        "COMPARISON DASHBOARD",
        "Compare customer segments, states, orders, income, and loyalty patterns.",
    )

    show_comparison_charts(customers_df)


def show_customer_data_page(customers_df):
    """Display the customer data page."""
    render_page_header(
        "Customer Data",
        "Review the full dataframe from Supabase with quick dataset context.",
    )

    total_rows = len(customers_df)
    total_columns = len(customers_df.columns)
    editable_columns = len(get_editable_fields(customers_df))

    col1, col2, col3 = st.columns(3)
    with col1:
        render_mini_card("Rows", f"{total_rows:,}")
    with col2:
        render_mini_card("Columns", f"{total_columns:,}")
    with col3:
        render_mini_card("Editable Fields", f"{editable_columns:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    show_customers_dataframe(customers_df)


def show_crud_management_page(customers_df, editable_fields):
    """Display all CRUD tools in a compact management area."""
    render_page_header(
        "CRUD Management",
        "Add, update, and delete customer records while keeping the workflow simple.",
    )

    add_tab, update_tab, delete_tab = st.tabs(
        ["Add Customer", "Update Customer", "Delete Customer"]
    )

    with add_tab:
        show_add_customer_form(editable_fields)

    with update_tab:
        show_update_customer_form(customers_df, editable_fields)

    with delete_tab:
        show_delete_customer_form(customers_df)


def run_app():
    """Load data once, then route the user to the selected page."""
    apply_custom_styles()
    show_message()

    # Authentication will be added later.
    # For now, the dashboard is shown directly when the app opens.
    try:
        customers_df = fetch_customers()
    except Exception as error:
        st.error(f"Could not load customers data: {error}")
        return

    editable_fields = get_editable_fields(customers_df)
    selected_page = show_sidebar(customers_df)
    filtered_customers_df = apply_dashboard_filters(customers_df)

    if selected_page == "Dashboard":
        show_dashboard_page(filtered_customers_df)
    elif selected_page == "Comparison Dashboard":
        show_comparison_dashboard_page(filtered_customers_df)
    elif selected_page == "Customer Data":
        show_customer_data_page(filtered_customers_df)
    elif selected_page == "CRUD Management":
        show_crud_management_page(customers_df, editable_fields)
    elif selected_page == "ML Prediction":
        show_ml_prediction_page(customers_df)
    elif selected_page == "ML Model Evaluation":
        show_ml_model_evaluation_page()


# --------------------------------------------------
# Main app routing
# --------------------------------------------------
run_app()
