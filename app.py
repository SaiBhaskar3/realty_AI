
import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import plotly.express as px
from typing import Dict, Any
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import openai


# Page config
st.set_page_config(
    page_title="US Neighborhood Comparison",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# trend indicators
def trend_indicator(value: str) -> str:
    """Generate HTML for trend indicators with arrows and colors"""
    try:
        # Extract numeric value, handling percentage signs
        numeric_str = ''.join(filter(lambda x: x.isdigit() or x == '.' or x == '-', value))
        numeric = float(numeric_str)
        
        # Determine if it's positive and the magnitude
        is_positive = numeric > 0
        magnitude = abs(numeric)
        
        # Define colors and arrows based on trend and magnitude
        if is_positive:
            if magnitude > 10:
                color = "#15803d"  
                arrow = "↑↑"
            else:
                color = "#22c55e"  
                arrow = "↑"
        else:
            if magnitude > 10:
                color = "#b91c1c"  
                arrow = "↓↓"
            else:
                color = "#ef4444"  # light red for moderate negative
                arrow = "↓"
        
        # Add background color for emphasis
        bg_color = f"{color}15"  # Add 15% opacity version of the color
        
        return f"""
            <div style='
                color: {color};
                font-weight: 600;
                background-color: {bg_color};
                padding: 0.25rem 0.5rem;
                border-radius: 0.25rem;
                display: inline-block;
                margin: 0.25rem 0;
            '>
                {arrow} {value}
            </div>
        """
    except:
        return f"<div>{value}</div>"

def metric_with_indicator(value: str, label: str, reverse: bool = False) -> str:
    """Generate HTML for metrics with colored indicators"""
    try:
        # Extract numeric value, handling percentage signs
        numeric_str = ''.join(filter(lambda x: x.isdigit() or x == '.' or x == '-', value))
        numeric = float(numeric_str)
        
        # Determine if this is good or bad (reverse logic if needed)
        is_good = numeric < 50 if reverse else numeric > 50
        
        # Define colors based on value ranges
        if reverse:
            if numeric < 25:
                color = "#15803d"  # dark green for very good
                indicator = "●●●"
            elif numeric < 50:
                color = "#22c55e"  # light green for good
                indicator = "●●"
            elif numeric < 75:
                color = "#ef4444"  # light red for poor
                indicator = "●"
            else:
                color = "#b91c1c"  # dark red for very poor
                indicator = "○"
        else:
            if numeric > 75:
                color = "#15803d"  # dark green for very good
                indicator = "●●●"
            elif numeric > 50:
                color = "#22c55e"  # light green for good
                indicator = "●●"
            elif numeric > 25:
                color = "#ef4444"  # light red for poor
                indicator = "●"
            else:
                color = "#b91c1c"  # dark red for very poor
                indicator = "○"
        
        # Add background color for emphasis
        bg_color = f"{color}15"  # Add 15% opacity version of the color
        
        return f"""
            <div style='
                background-color: {bg_color};
                padding: 0.5rem;
                border-radius: 0.25rem;
                text-align: center;
            '>
                <div style='color: {color}; font-weight: 600;'>{value}</div>
                <div style='color: {color}; font-size: 0.75rem; margin: 0.25rem 0;'>{indicator}</div>
                <div style='color: #6b7280; font-size: 0.875rem;'>{label}</div>
            </div>
        """
    except:
        return f"""
            <div style='text-align: center;'>
                <div>{value}</div>
                <div style='color: #6b7280; font-size: 0.875rem;'>{label}</div>
            </div>
        """

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 0.75rem;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    .stButton>button:active {
        transform: translateY(0);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .comparison-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }
    .metric-title {
        color: #4b5563;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        color: #111827;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .section-title {
        color: #111827;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    .stMarkdown {
        padding: 0.5rem 0;
    }
    .metrics-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    .metric-item {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
    }
    .metric-title {
        color: #6b7280;
        font-size: 0.875rem;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        color: #111827;
        font-size: 1.25rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Header with improved styling
st.markdown("""
    <h1 style='text-align: center; color: #111827; margin-bottom: 2rem;'>
        🏘️ US Neighborhood Comparison Tool
    </h1>
""", unsafe_allow_html=True)

st.markdown("""
    <p style='text-align: center; font-size: 1.1rem; color: #4b5563; margin-bottom: 2rem;'>
        Compare any two locations across the United States based on key metrics:
    </p>
""", unsafe_allow_html=True)

# Create metrics list with improved styling
metrics_list = """
<div style='display: flex; justify-content: center; gap: 2rem; margin-bottom: 2rem;'>
    <div>📚 Education & Schools</div>
    <div>🏠 Real Estate Market</div>
    <div>🚓 Safety & Crime</div>
    <div>✨ Quality of Life</div>
</div>
"""
st.markdown(metrics_list, unsafe_allow_html=True)

# Input fields in two columns with improved styling
col1, col2 = st.columns(2)
with col1:
    location1 = st.text_input(
        "First Location",
        "Seattle, WA",
        help="Enter city and state (e.g., Seattle, WA)"
    )

with col2:
    location2 = st.text_input(
        "Second Location",
        "Portland, OR",
        help="Enter city and state (e.g., Portland, OR)"
    )

import yfinance as yf
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from bs4 import BeautifulSoup
import json
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Key for air quality data
WAQI_API_KEY = os.getenv('WAQI_API_KEY')

# Check if we're in demo mode
DEMO_MODE = not WAQI_API_KEY
if DEMO_MODE:
    st.warning("""
        ⚠️ Running in demo mode with estimated data. 
        Some metrics will be approximated based on historical trends and patterns.
        To get real-time data, please add your API keys to the .env file.
    """)

def get_location_data(location: str) -> Dict[Any, Any]:
    """
    Get data for a location using APIs or demo data
    """
    try:
        # Parse city and state
        city, state = location.split(',')
        city = city.strip()
        state = state.strip()

        # Get coordinates using Nominatim with retry logic
        geolocator = Nominatim(user_agent="neighborhood_comparison_tool")
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                location_data = geolocator.geocode(f"{city}, {state}, USA", timeout=10)
                if location_data:
                    lat, lon = location_data.latitude, location_data.longitude
                    st.success(f"Successfully found coordinates for {city}, {state}")
                    break
                else:
                    if attempt == max_retries - 1:
                        st.error(f"Could not find coordinates for {city}, {state}")
                        lat, lon = 0, 0  # Default coordinates
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Error getting coordinates: {str(e)}")
                    lat, lon = 0, 0  # Default coordinates
                time.sleep(retry_delay)

        # Get real estate data
        real_estate_data = get_real_estate_data(city, state)
        
        # Get safety data from FBI UCR API simulation
        safety_data = get_safety_data(city, state)
        
        # Get quality of life data
        quality_data = get_quality_data(lat, lon)
        
        # Get education data with highest ranked school
        education_data = get_education_data(city, state)
        
        # Combine all data
        return {
            "education": education_data,
            "real_estate": real_estate_data,
            "safety": safety_data,
            "quality_of_life": quality_data
        }
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def test_census_api():
    """Test Census API connectivity"""
    try:
        # Test with a simple query for median household income
        CENSUS_API_KEY = '8eaa824d600a0405a510f7675105ab2e95ac139d'
        test_url = f"https://api.census.gov/data/2021/acs/acs5?get=NAME,B19013_001E&for=state:*&key={CENSUS_API_KEY}"
        
        st.info(f"Testing Census API connection with URL: {test_url}")
        response = requests.get(test_url, timeout=30)
        
        st.info(f"""Response details:
            Status: {response.status_code}
            Content-Type: {response.headers.get('content-type')}
            Response: {response.text[:1000]}
        """)
        
        if response.status_code == 200:
            st.success("Census API test successful!")
            return True
        else:
            st.error("Census API test failed")
            return False
            
    except Exception as e:
        st.error(f"Census API test error: {str(e)}")
        return False

def get_real_estate_data(city: str, state: str) -> Dict[str, Any]:
    """Get real estate data from data_gov_bldg_rexus.csv for the given city and state"""
    try:
        df = pd.read_csv("data_gov_bldg_rexus.csv", dtype=str)
        # Filter by city and state (ignore case and possible spaces)
        city = city.strip().upper()
        state = state.strip().upper()
        matches = df[(df["Bldg City"].str.strip().str.upper() == city) & (df["Bldg State"].str.strip().str.upper() == state)]
        
        if matches.empty:
            return {
                "median_price": "No data",
                "median_rent": "No data",
                "total_units": "No data",
                "occupancy_rate": "No data",
                "market_health": "No data",
                "first_address": "No building found in database."
            }
        else:
            # For demo, just take the first matching row
            row = matches.iloc[0]
            return {
                "first_address": row["Bldg Address1"],
                "building_status": row["Bldg Status"],
                "property_type": row["Property Type"],
                "usable_sqft": row["Bldg ANSI Usable"],
                "total_parking": row["Total Parking Spaces"],
                "owned_leased": row["Owned/Leased"],
                "construction_date": row["Construction Date"],
                "historical_status": row["Historical Status"],
                "aba_accessibility": row.get("ABA Accessibility Flag", "Unknown"),
                "city": row["Bldg City"],
                "state": row["Bldg State"]
            }
    except Exception as e:
        st.error(f"Error reading real estate data from CSV: {str(e)}")
        return {
            "first_address": "Error",
            "building_status": "Error",
            "property_type": "Error",
            "usable_sqft": "Error",
            "total_parking": "Error",
            "owned_leased": "Error",
            "construction_date": "Error",
            "historical_status": "Error",
            "aba_accessibility": "Error",
            "city": city,
            "state": state
        }
def get_safety_data(city: str, state: str) -> Dict[str, Any]:
    """Get safety data from FBI UCR and local sources"""
    try:
        # Simulate FBI UCR API data
        base_score = 75
        city_adjustments = {
            "new york": 5, "san francisco": -5, "seattle": 3, "portland": -2,
            "los angeles": -3, "chicago": -8, "boston": 7, "austin": 4
        }
        
        adjustment = city_adjustments.get(city.lower(), 0)
        safety_score = base_score + adjustment
        
        return {
            "crime_index": safety_score,
            "safety_score": f"{safety_score}%",
            "violent_crime_rate": f"{(100-safety_score)/20:.1f} per 1,000",
            "property_crime_rate": f"{(100-safety_score)/4:.1f} per 1,000",
            "police_response": f"{int(5 + (100-safety_score)/10)} min avg",
            "crime_trend": f"{(safety_score-70)/2:.1f}% YoY",
            "neighborhood_watch": f"{int(safety_score/3)} active groups"
        }
    except Exception as e:
        st.warning(f"Using estimated safety data: {str(e)}")
        return {
            "crime_index": "N/A",
            "safety_score": "N/A",
            "violent_crime_rate": "N/A",
            "property_crime_rate": "N/A",
            "police_response": "N/A",
            "crime_trend": "N/A",
            "neighborhood_watch": "N/A"
        }

def get_quality_data(lat: float, lon: float) -> Dict[str, Any]:
    """Get quality of life data using various APIs"""
    try:
        # Calculate quality scores based on location characteristics
        distance_from_coast = abs(lon + 100) / 20  # Coastal proximity factor
        urban_density = abs(40 - lat) / 10  # Urban density approximation
        elevation_factor = max(0, min(10, abs(lat - 40)))  # Climate/elevation influence
        
        # Base score calculation with multiple factors
        base_score = max(50, min(95, 75 - distance_from_coast + urban_density - elevation_factor))
        
        # Calculate specific metrics based on the base score
        walkability = int(base_score * (1 + urban_density/20))
        air_quality = int(base_score + (distance_from_coast - urban_density))
        parks = int(base_score/8 + elevation_factor)
        restaurants = int(base_score * (2 + urban_density/10))
        commute = int(35 - base_score/4 + urban_density)
        transit = int(base_score * (0.8 + urban_density/15))
        healthcare = int(base_score * (0.9 + urban_density/20))
        
        return {
            "walkability": f"{min(100, walkability)}/100",
            "air_quality": f"{min(100, air_quality)}/100",
            "parks_nearby": parks,
            "restaurants": restaurants,
            "commute_time": f"{commute} min avg",
            "public_transit": f"{min(100, transit)}/100",
            "healthcare_access": f"{min(100, healthcare)}/100"
        }
    except Exception as e:
        st.warning(f"Using estimated quality of life data: {str(e)}")
        return {
            "walkability": "N/A",
            "air_quality": "N/A",
            "parks_nearby": "N/A",
            "restaurants": "N/A",
            "commute_time": "N/A",
            "public_transit": "N/A",
            "healthcare_access": "N/A"
        }

def get_education_data(city: str, state: str) -> Dict[str, Any]:
    """Get education data with highest ranked school in the district"""
    try:
        # Demo data based on city characteristics
        city_ratings = {
            "seattle": {"rating": 8.5, "rank": 12},
            "portland": {"rating": 7.8, "rank": 24},
            "san francisco": {"rating": 8.9, "rank": 8},
            "new york": {"rating": 8.7, "rank": 10},
            "chicago": {"rating": 7.5, "rank": 35},
            "boston": {"rating": 9.1, "rank": 5},
            "austin": {"rating": 8.2, "rank": 18},
            "denver": {"rating": 7.9, "rank": 22}
        }
        
        city_data = city_ratings.get(city.lower(), {"rating": 7.0, "rank": 50})
        total_schools = int(30 + (city_data["rating"] * 5))
        
        return {
            "district_name": f"{city} School District",
            "highest_ranked_school": f"{city} High School",
            "school_rank": f"#{city_data['rank']} in {state}",
            "school_rating": f"{city_data['rating']}/10",
            "total_schools": str(total_schools)
        }
    except Exception as e:
        st.warning(f"Using estimated education data: {str(e)}")
        return {
            "district_name": f"{city} School District",
            "highest_ranked_school": "Local High School",
            "school_rank": "N/A",
            "school_rating": "7.0/10",
            "total_schools": "35"
        }
def semantic_retrieve_rexus(user_question, df_rexus, rexus_embeddings, emb_model, top_k=3):
    """
    Retrieve the top K rows from df_rexus most semantically similar to the user_question.
    """

    question_emb = emb_model.encode([user_question])
    # Compute cosine similarity
    similarities = np.dot(rexus_embeddings, question_emb.T).squeeze()
    # Get top K indices
    top_indices = similarities.argsort()[-top_k:][::-1]
    return df_rexus.iloc[top_indices]
# Initialize data in session state
if 'data1' not in st.session_state:
    st.session_state.data1 = None
if 'data2' not in st.session_state:
    st.session_state.data2 = None

# prominent comparison section
st.markdown("""
    <div style='text-align: center; background-color: #f0f9ff; padding: 2rem; border-radius: 1rem; margin: 1rem 0; border: 2px solid #bae6fd; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);'>
        <h3 style='color: #0369a1; font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem;'>
            🎯 Ready to Compare These Locations?
        </h3>
        <p style='color: #0c4a6e; font-size: 1.1rem; margin-bottom: 2rem;'>
            Get detailed insights about schools, housing, safety, and quality of life
        </p>
    </div>
""", unsafe_allow_html=True)

# Make the button more prominent
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# container for the loading animation and button
loading_container = st.empty()

# comparison button with a key
if st.button("🔄 Compare Locations Now", 
            key="compare_button",
            help="Click to load comparison data", 
            use_container_width=True,
            type="primary"):  # Make it primary to stand out
    try:
        # loading animation with progress
        with loading_container.container():
            st.markdown("""
                <div style='text-align: center; padding: 2rem;'>
                    <div style='display: inline-block; padding: 1rem 2rem; background-color: #e0f2fe; border-radius: 0.5rem; margin-bottom: 1rem;'>
                        <p style='color: #0369a1; font-size: 1.1rem; margin-bottom: 0.5rem;'>🔄 Loading comparison data...</p>
                        <div style='color: #0c4a6e; font-size: 0.9rem;'>This may take a few moments</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Get first location data
            status_text.text("Loading data for " + location1 + "...")
            progress_bar.progress(25)
            st.session_state.data1 = get_location_data(location1)
            progress_bar.progress(50)
            
            # Get second location data
            status_text.text("Loading data for " + location2 + "...")
            progress_bar.progress(75)
            st.session_state.data2 = get_location_data(location2)
            progress_bar.progress(100)
            status_text.text("Data loaded successfully!")
            
        # Clear the loading animation
        loading_container.empty()
        
    except Exception as e:
        st.error(f"An error occurred while comparing locations: {str(e)}")
        st.session_state.data1 = None
        st.session_state.data2 = None

# Display comparison if data exists in session state
if st.session_state.data1 and st.session_state.data2:
    # Display comparison sections
    sections = ["education", "real_estate", "safety", "quality_of_life"]
    section_icons = {
        "education": "📚", 
        "real_estate": "🏠",
        "safety": "🚓",
        "quality_of_life": "✨"
    }

def display_real_estate_section(data1, data2, location1, location2):
    st.markdown("""
        <h2 style='text-align: center; color: #111827; margin: 2rem 0 1rem;'>
            🏠 Best available home
        </h2>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    def display_market_metrics(data, location):
        try:
            real_estate = data.get('real_estate', {})
            st.markdown(f"""
                <div style='background-color: white; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                    <h4 style='color: #111827; margin-bottom: 1rem;'>{location}</h4>
                    <div class='metrics-container'>
                        <div class='metric-item'><span class='metric-title'>Address:</span> <span class='metric-value'>{real_estate.get('first_address', 'N/A')}</span></div>
                        <div class='metric-item'><span class='metric-title'>Building Status:</span> <span class='metric-value'>{real_estate.get('building_status', 'N/A')}</span></div>
                        <div class='metric-item'><span class='metric-title'>Property Type:</span> <span class='metric-value'>{real_estate.get('property_type', 'N/A')}</span></div>
                        <div class='metric-item'><span class='metric-title'>Usable SqFt:</span> <span class='metric-value'>{real_estate.get('usable_sqft', 'N/A')}</span></div>
                        <div class='metric-item'><span class='metric-title'>Parking Spaces:</span> <span class='metric-value'>{real_estate.get('total_parking', 'N/A')}</span></div>
                        <div class='metric-item'><span class='metric-title'>Owned/Leased:</span> <span class='metric-value'>{real_estate.get('owned_leased', 'N/A')}</span></div>
                        <div class='metric-item'><span class='metric-title'>Construction Date:</span> <span class='metric-value'>{real_estate.get('construction_date', 'N/A')}</span></div>
                        <div class='metric-item'><span class='metric-title'>Historical Status:</span> <span class='metric-value'>{real_estate.get('historical_status', 'N/A')}</span></div>
                        <div class='metric-item'><span class='metric-title'>ABA Accessibility:</span> <span class='metric-value'>{real_estate.get('aba_accessibility', 'N/A')}</span></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying market metrics: {str(e)}")
    with col1:
        display_market_metrics(data1, location1)
    with col2:
        display_market_metrics(data2, location2)
    
    def display_market_metrics(data, location):
        try:
            real_estate = data.get('real_estate', {})
            st.markdown(f"""
                <div style='background-color: white; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                    <h4 style='color: #111827; margin-bottom: 1rem;'>{location}</h4>
                    <div class='metrics-container'>
                        <div class='metric-item'><span class='metric-title'>Address:</span> <span class='metric-value'>{real_estate.get('first_address', 'N/A')}</span></div>
                        <div class='metric-item'><span class='metric-title'>Building Status:</span> <span class='metric-value'>{real_estate.get('building_status', 'N/A')}</span></div>
                        <div class='metric-item'><span class='metric-title'>Property Type:</span> <span class='metric-value'>{real_estate.get('property_type', 'N/A')}</span></div>
                        <div class='metric-item'><span class='metric-title'>Usable SqFt:</span> <span class='metric-value'>{real_estate.get('usable_sqft', 'N/A')}</span></div>
                        <div class='metric-item'><span class='metric-title'>Parking Spaces:</span> <span class='metric-value'>{real_estate.get('total_parking', 'N/A')}</span></div>
                        <div class='metric-item'><span class='metric-title'>Owned/Leased:</span> <span class='metric-value'>{real_estate.get('owned_leased', 'N/A')}</span></div>
                        <div class='metric-item'><span class='metric-title'>Construction Date:</span> <span class='metric-value'>{real_estate.get('construction_date', 'N/A')}</span></div>
                        <div class='metric-item'><span class='metric-title'>Historical Status:</span> <span class='metric-value'>{real_estate.get('historical_status', 'N/A')}</span></div>
                        <div class='metric-item'><span class='metric-title'>ABA Accessibility:</span> <span class='metric-value'>{real_estate.get('aba_accessibility', 'N/A')}</span></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying market metrics: {str(e)}")
    with col1:
        display_market_metrics(data1, location1)
    
    with col2:
        display_market_metrics(data2, location2)

# Define sections and icons
sections = ["education", "real_estate", "safety", "quality_of_life"]
section_icons = {
    "education": "📚", 
    "real_estate": "🏠",
    "safety": "🚓",
    "quality_of_life": "✨"
}

# Display sections
if st.session_state.data1 and st.session_state.data2:
    try:
        # Display timestamp
        st.markdown(f"""
            <p style='text-align: center; color: #6b7280; font-size: 0.875rem; margin: 2rem 0;'>
                Data updated: {datetime.now().strftime('%B %d, %Y %I:%M %p')}
            </p>
        """, unsafe_allow_html=True)

        # sections and their display order
        sections_order = [
            ("education", "Education & Schools", "📚"),
            ("real_estate", "Real Estate Market", "🏠"),
            ("safety", "Safety & Crime", "🚓"),
            ("quality_of_life", "Quality of Life", "✨")
        ]
        
        # sections in order
        for section, title, icon in sections_order:
            try:
                st.markdown(
                    f"<h2 class='section-title'>{icon} {title}</h2>",
                    unsafe_allow_html=True
                )
                
                if section == "real_estate":
                    display_real_estate_section(st.session_state.data1, st.session_state.data2, location1, location2)
                else:
                    # three columns for better layout
                    col1, col_spacer, col2 = st.columns([10, 1, 10])
                    
                    # Display data for location 1
                    with col1:
                        st.markdown(
                            f"""
                            <div class='comparison-card'>
                                <h3 style='color: #111827; margin-bottom: 1rem;'>{location1}</h3>
                                <div class='metrics-container'>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        if section in st.session_state.data1 and st.session_state.data1[section]:
                            for key, value in st.session_state.data1[section].items():
                                st.markdown(
                                    f"""
                                    <div class='metric-item'>
                                        <div class='metric-title'>{key.replace('_', ' ').title()}</div>
                                        <div class='metric-value'>{value}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                        else:
                            st.warning(f"No {section} data available for {location1}")
                        
                        st.markdown("</div></div>", unsafe_allow_html=True)
                    
                    # Display data for location 2
                    with col2:
                        st.markdown(
                            f"""
                            <div class='comparison-card'>
                                <h3 style='color: #111827; margin-bottom: 1rem;'>{location2}</h3>
                                <div class='metrics-container'>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        if section in st.session_state.data2 and st.session_state.data2[section]:
                            for key, value in st.session_state.data2[section].items():
                                st.markdown(
                                    f"""
                                    <div class='metric-item'>
                                        <div class='metric-title'>{key.replace('_', ' ').title()}</div>
                                        <div class='metric-value'>{value}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                        else:
                            st.warning(f"No {section} data available for {location2}")
                        
                        st.markdown("</div></div>", unsafe_allow_html=True)
                
                # Add spacing between sections
                if section != "quality_of_life":
                    st.markdown("<br>", unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error displaying {section} data: {str(e)}")
                continue
                
    except Exception as e:
        st.error(f"Error displaying comparison sections: {str(e)}")
# --- Initialization for AI Assistant (Chatbot) ---

# DataFrame (adjust the path as needed)
df_rexus = pd.read_csv("price.csv", dtype=str)

# embeddings for address/city/state columns
emb_model = SentenceTransformer('all-MiniLM-L6-v2')
df_rexus = pd.read_csv("data_gov_bldg_rexus.csv", dtype=str)
# Computing embeddings for the address+city+state columns
rexus_embeddings = emb_model.encode(
    df_rexus["Bldg Address1"] + " " + df_rexus["Bldg City"] + " " + df_rexus["Bldg State"]
)
# chatbot interface in sidebar with improved styling
st.sidebar.markdown("""
    <h2 style='color: #111827; margin-bottom: 1rem;'>🤖 AI Assistant</h2>
""", unsafe_allow_html=True)

# chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Show different messages based on comparison state
st.sidebar.markdown("""
    <div style='background-color: #f8fafc; padding: 0.75rem; border-radius: 0.5rem; margin-top: 1rem;'>
        <p style='color: #4b5563; font-size: 0.875rem; margin: 0;'>
            Example questions:
            <br>• Which city has better schools?
            <br>• How do home prices compare?
            <br>• Which area is safer?
        </p>
    </div>
""", unsafe_allow_html=True)

user_question = st.sidebar.text_input(
    "Ask me anything about these locations:",
    placeholder="E.g., Which location has better schools?",
    key="user_input"
)
retrieved_rows = semantic_retrieve_rexus(
    user_question, df_rexus, rexus_embeddings, emb_model, top_k=3
)
# submit button to prevent auto-refresh
if st.sidebar.button("Ask", disabled=not (st.session_state.data1 and st.session_state.data2)):
    if user_question:
        # 1. Retrieve top relevant building rows using semantic search
        

        # 2. context for the LLM
        context_snippets = []
        for idx, row in retrieved_rows.iterrows():
            snippet = (
                f"Address: {row['Bldg Address1']}, {row['Bldg City']}, {row['Bldg State']} | "
                f"Status: {row['Bldg Status']} | Type: {row['Property Type']} | "
                f"Usable SqFt: {row['Bldg ANSI Usable']} | Parking: {row['Total Parking Spaces']} | "
                f"Owned/Leased: {row['Owned/Leased']} | Built: {row['Construction Date']} | "
                f"Historical: {row['Historical Status']} | ABA Accessibility: {row.get('ABA Accessibility Flag', 'Unknown')}"
            )
            context_snippets.append(snippet)
        context = "\n".join(context_snippets)

        # prompt for the LLM
        prompt = (
            "You are an expert assistant answering questions about US government buildings. "
            "Use ONLY the data provided below to answer the user's question. If the answer is not in the data, say so.\n\n"
            f"Building Data:\n{context}\n\n"
            f"User Question: {user_question}\n"
            "Answer:"
        )

        # OpenAI for an answer
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.2,
                max_tokens=350
            )
            response = completion["choices"][0]["message"]["content"].strip()
        except Exception as e:
            response = f"Error with OpenAI API: {e}"

        # Add to chat history
        st.session_state.chat_history.append({"question": user_question, "answer": response})

    elif user_question:
        st.sidebar.error("⚠️ Please click 'Compare Locations' first to load the data!")
# Display chat history
if st.session_state.chat_history:
    st.sidebar.markdown("### Previous Questions")
    for chat in st.session_state.chat_history:
        st.sidebar.markdown(f"""
            <div style='background-color: #f8fafc; padding: 0.5rem; border-radius: 0.5rem; margin-bottom: 0.5rem;'>
                <p style='color: #4b5563; font-size: 0.875rem; margin-bottom: 0.25rem;'><strong>Q:</strong> {chat['question']}</p>
                <p style='color: #111827; margin: 0;'><strong>A:</strong> {chat['answer']}</p>
            </div>
        """, unsafe_allow_html=True)
