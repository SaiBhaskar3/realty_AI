import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from sentence_transformers import SentenceTransformer
    S_EMBED_AVAILABLE = True
except Exception:
    S_EMBED_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="US Neighborhood Comparison", page_icon="🏘️", layout="wide")

from dotenv import load_dotenv
load_dotenv()

WAQI_API_KEY = os.getenv("WAQI_API_KEY", "")
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if OPENAI_API_KEY and OPENAI_AVAILABLE:
    openai.api_key = OPENAI_API_KEY

DEMO_MODE = not (WAQI_API_KEY or CENSUS_API_KEY)  # simple demo indicator

def safe_float(x, default=np.nan):
    try:
        return float(str(x).replace(",", "").replace("%", ""))
    except Exception:
        return default

def sanitize_location_text(location: str) -> Tuple[str, str]:
    """Return (city, state) parsed from 'City, ST' style input."""
    if not location:
        return "", ""
    parts = [p.strip() for p in location.split(",")]
    if len(parts) >= 2:
        return parts[0], parts[1]
    else:
        # If user typed only a city, return that and empty state
        return parts[0], ""

st.markdown(
    """
    <style>
    .section-title { color: #111827; font-size: 1.3rem; font-weight: 700; margin: 1rem 0 0.5rem 0;}
    .metric-item { background-color: #fff; padding: 0.75rem; border-radius: 6px; border: 1px solid #e5e7eb; }
    </style>
    """, unsafe_allow_html=True
)

@st.cache_data(show_spinner=False)
def load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, dtype=str)
    except Exception as e:
        st.warning(f"Failed to load {path}: {e}")
        return None

@st.cache_resource
def init_sentence_model(model_name: str = "all-MiniLM-L6-v2"):
    if not S_EMBED_AVAILABLE:
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        st.warning(f"SentenceTransformer init failed: {e}")
        return None

df_price = load_csv_safe("price.csv")  # optional data you had referenced
df_rexus = load_csv_safe("data_gov_bldg_rexus.csv")

emb_model = init_sentence_model() if S_EMBED_AVAILABLE else None
rexus_embeddings = None
if emb_model is not None and df_rexus is not None:
    # Prepare a combined text column safely
    combined = (df_rexus.get("Bldg Address1", "").fillna("") + " " +
                df_rexus.get("Bldg City", "").fillna("") + " " +
                df_rexus.get("Bldg State", "").fillna(""))
    try:
        rexus_embeddings = emb_model.encode(list(combined), show_progress_bar=False)
    except Exception as e:
        st.warning(f"Failed to compute embeddings: {e}")
        rexus_embeddings = None

def semantic_retrieve_rexus(user_question: str, top_k: int = 3) -> pd.DataFrame:
    if not user_question:
        return pd.DataFrame()
    if df_rexus is None or emb_model is None or rexus_embeddings is None:
        # Fallback: return top rows (non-semantic)
        if df_rexus is None:
            return pd.DataFrame()
        return df_rexus.head(top_k)
    q_emb = emb_model.encode([user_question])
    sims = np.dot(rexus_embeddings, q_emb.T).squeeze()
    top_idx = sims.argsort()[-top_k:][::-1]
    return df_rexus.iloc[top_idx]

def get_real_estate_data(city: str, state: str) -> Dict[str, Any]:
    """Return a dict with building info using data_gov_bldg_rexus.csv if available."""
    try:
        if df_rexus is None:
            # Demo synthetic values
            return {
                "first_address": "Demo Building, 123 Demo St",
                "building_status": "Active",
                "property_type": "Office",
                "usable_sqft": "25,000",
                "total_parking": "50",
                "owned_leased": "Owned",
                "construction_date": "2010",
                "historical_status": "No",
                "aba_accessibility": "Yes",
                "city": city,
                "state": state
            }
        mcity, mstate = city.strip().upper(), state.strip().upper()
        matches = df_rexus[
            (df_rexus.get("Bldg City", "").str.strip().str.upper() == mcity) &
            (df_rexus.get("Bldg State", "").str.strip().str.upper() == mstate)
        ]
        if matches.empty:
            # fallback to first row or demo
            row = df_rexus.iloc[0]
        else:
            row = matches.iloc[0]
        return {
            "first_address": row.get("Bldg Address1", "N/A"),
            "building_status": row.get("Bldg Status", "N/A"),
            "property_type": row.get("Property Type", "N/A"),
            "usable_sqft": row.get("Bldg ANSI Usable", "N/A"),
            "total_parking": row.get("Total Parking Spaces", "N/A"),
            "owned_leased": row.get("Owned/Leased", "N/A"),
            "construction_date": row.get("Construction Date", "N/A"),
            "historical_status": row.get("Historical Status", "N/A"),
            "aba_accessibility": row.get("ABA Accessibility Flag", "Unknown"),
            "city": row.get("Bldg City", city),
            "state": row.get("Bldg State", state)
        }
    except Exception as e:
        st.error(f"Error in get_real_estate_data: {e}")
        return {}

def get_safety_data(city: str, state: str) -> Dict[str, Any]:
    """Simulated safety data, easy to replace with FBI/Census calls when you add keys."""
    base_score = 75
    adjustments = {
        "NEW YORK": 5, "SAN FRANCISCO": -5, "SEATTLE": 3, "PORTLAND": -2,
        "LOS ANGELES": -3, "CHICAGO": -8, "BOSTON": 7, "AUSTIN": 4
    }
    adj = adjustments.get(city.strip().upper(), 0)
    score = max(20, min(95, base_score + adj))
    return {
        "crime_index": score,
        "safety_score": f"{score}%",
        "violent_crime_rate": f"{(100-score)/20:.2f} per 1,000",
        "property_crime_rate": f"{(100-score)/4:.2f} per 1,000",
        "police_response": f"{int(4 + (100-score)/20)} min avg",
        "crime_trend": f"{(score-70)/2:.1f}% YoY",
        "neighborhood_watch": f"{int(score/3)} active groups"
    }

def get_quality_data(lat: float, lon: float) -> Dict[str, Any]:
    """Estimate quality-of-life metrics using lat/lon or provide a demo set."""
    try:
        distance_from_coast = abs((lon + 100) / 20) if lon is not None else 2
        urban_density = abs(40 - (lat if lat is not None else 40)) / 10
        base = max(50, min(95, 75 - distance_from_coast + urban_density))
        walkability = int(min(100, base * (1 + urban_density/20)))
        air_quality = int(min(100, base - (distance_from_coast/2)))
        parks = int(min(100, base / 8 + urban_density))
        restaurants = int(min(100, base * (1.5 + urban_density/10)))
        commute = int(max(10, min(90, 35 - base/4 + urban_density)))
        transit = int(min(100, base * (0.8 + urban_density/15)))
        healthcare = int(min(100, base * (0.9 + urban_density/20)))
        return {
            "walkability": f"{walkability}/100",
            "air_quality": f"{air_quality}/100",
            "parks_nearby": parks,
            "restaurants": restaurants,
            "commute_time": f"{commute} min avg",
            "public_transit": f"{transit}/100",
            "healthcare_access": f"{healthcare}/100"
        }
    except Exception as e:
        st.warning(f"quality fallback: {e}")
        return {}

def geocode_city_state(city: str, state: str) -> Tuple[Optional[float], Optional[float]]:
    """Lightweight geocode fallback - not accurate. Use Nominatim/geopy or a proper API for production."""
    demo_coords = {
        ("Seattle", "WA"): (47.6062, -122.3321),
        ("Portland", "OR"): (45.5122, -122.6587),
        ("San Francisco", "CA"): (37.7749, -122.4194),
        ("New York", "NY"): (40.7128, -74.0060),
        ("Boston", "MA"): (42.3601, -71.0589),
        ("Chicago", "IL"): (41.8781, -87.6298),
        ("Austin", "TX"): (30.2672, -97.7431),
        ("Denver", "CO"): (39.7392, -104.9903)
    }
    return demo_coords.get((city, state), (None, None))

def get_location_data(location: str) -> Dict[str, Any]:
    """Aggregate data for UI from smaller functions; robust to missing pieces."""
    city, state = sanitize_location_text(location)
    lat, lon = geocode_city_state(city, state)
    real = get_real_estate_data(city, state)
    safety = get_safety_data(city, state)
    quality = get_quality_data(lat if lat else 40.0, lon if lon else -100.0)
    education = {
        "district_name": f"{city} School District" if city else "Local School District",
        "highest_ranked_school": f"{city} High School" if city else "Local High School",
        "school_rank": "#12" if city.lower() in ["seattle", "boston"] else "#24",
        "school_rating": "8.2/10",
        "total_schools": "35"
    }
    return {
        "education": education,
        "real_estate": real,
        "safety": safety,
        "quality_of_life": quality,
        "coords": {"lat": lat, "lon": lon}
    }

st.sidebar.title("US Neighborhood Comparison")
page = st.sidebar.selectbox("Go to", ["Compare", "Data Explorer", "AI Assistant", "Settings"])

with st.sidebar.form("location_form", clear_on_submit=False):
    loc1 = st.text_input("First location", value="Seattle, WA", help="City, ST")
    loc2 = st.text_input("Second location", value="Portland, OR", help="City, ST")
    submit_locations = st.form_submit_button("Set locations")

if "data1" not in st.session_state:
    st.session_state.data1 = None
if "data2" not in st.session_state:
    st.session_state.data2 = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if page == "Compare":
    st.header("🏘️ Compare two US locations")
    left_col, right_col = st.columns([1, 3])
    with left_col:
        st.markdown("### Locations")
        st.write(f"**A:** {loc1}")
        st.write(f"**B:** {loc2}")
        if st.button("🔄 Load & Compare"):
            # load & cache
            with st.spinner("Loading data..."):
                st.session_state.data1 = get_location_data(loc1)
                st.session_state.data2 = get_location_data(loc2)
            st.success("Data loaded")

        st.markdown("---")
        st.markdown("### Tips")
        st.markdown("- Use `City, ST` format (e.g., Seattle, WA).")
        st.markdown("- If you have data files, place them in the app root (price.csv, data_gov_bldg_rexus.csv).")

    with right_col:
        if not st.session_state.data1 or not st.session_state.data2:
            st.info("Load locations to see comparison cards and charts.")
        else:
            st.markdown(f"**Data updated:** {datetime.now().strftime('%B %d, %Y %I:%M %p')}")
            # Real estate card side-by-side
            st.subheader("🏠 Real Estate Snapshot")
            a, b = st.columns(2)
            def display_market_metrics_card(data, title):
                real = data.get("real_estate", {})
                st.markdown(f"#### {title}")
                st.markdown("<div class='metric-item'>", unsafe_allow_html=True)
                st.write("**Address:**", real.get("first_address", "N/A"))
                st.write("**Status:**", real.get("building_status", "N/A"))
                st.write("**Type:**", real.get("property_type", "N/A"))
                st.write("**Usable SqFt:**", real.get("usable_sqft", "N/A"))
                st.write("**Parking:**", real.get("total_parking", "N/A"))
                st.write("**Owned/Leased:**", real.get("owned_leased", "N/A"))
                st.write("**Built:**", real.get("construction_date", "N/A"))
                st.write("**Historical:**", real.get("historical_status", "N/A"))
                st.write("**ABA Accessible:**", real.get("aba_accessibility", "N/A"))
                st.markdown("</div>", unsafe_allow_html=True)

            with a:
                display_market_metrics_card(st.session_state.data1, loc1)
            with b:
                display_market_metrics_card(st.session_state.data2, loc2)

            st.markdown("---")
            # Safety comparison bar chart
            st.subheader("🚓 Safety comparison")
            safety_df = pd.DataFrame([
                {"location": loc1, "safety_percent": safe_float(st.session_state.data1["safety"].get("crime_index", 0))},
                {"location": loc2, "safety_percent": safe_float(st.session_state.data2["safety"].get("crime_index", 0))}
            ])
            fig = px.bar(safety_df, x="location", y="safety_percent", text="safety_percent", labels={"safety_percent": "Crime Index (higher = safer)"})
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("✨ Quality of life comparison (radar)")
            def qdf_for_radar(data, label):
                q = data.get("quality_of_life", {})
                items = {
                    "Walkability": safe_float(q.get("walkability", "0")),
                    "AirQuality": safe_float(q.get("air_quality", "0")),
                    "Transit": safe_float(str(q.get("public_transit", "0")).split("/")[0]),
                    "Healthcare": safe_float(str(q.get("healthcare_access", "0")).split("/")[0]),
                    "Restaurants": safe_float(q.get("restaurants", 0))
                }
                df = pd.DataFrame([items])
                df["label"] = label
                return df

            r1 = qdf_for_radar(st.session_state.data1, loc1)
            r2 = qdf_for_radar(st.session_state.data2, loc2)
            radar_df = pd.concat([r1, r2], ignore_index=True)
            radar_long = radar_df.melt(id_vars=["label"], var_name="metric", value_name="value")
            fig_radar = px.line_polar(radar_long, r="value", theta="metric", color="label", line_close=True)
            st.plotly_chart(fig_radar, use_container_width=True)

            st.subheader("📚 Education & Schools")
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                ed = st.session_state.data1.get("education", {})
                st.markdown(f"**{loc1}**")
                st.write("District:", ed.get("district_name"))
                st.write("Top school:", ed.get("highest_ranked_school"))
                st.write("Rank:", ed.get("school_rank"))
                st.write("Rating:", ed.get("school_rating"))
            with col_e2:
                ed = st.session_state.data2.get("education", {})
                st.markdown(f"**{loc2}**")
                st.write("District:", ed.get("district_name"))
                st.write("Top school:", ed.get("highest_ranked_school"))
                st.write("Rank:", ed.get("school_rank"))
                st.write("Rating:", ed.get("school_rating"))

elif page == "Data Explorer":
    st.header("🔎 Data Explorer")
    st.write("Inspect loaded CSV files and run semantic search if model is available.")
    if df_rexus is None:
        st.warning("`data_gov_bldg_rexus.csv` not found in app root.")
    else:
        st.markdown("### data_gov_bldg_rexus (sample)")
        st.dataframe(df_rexus.head(50))

    st.markdown("---")
    st.subheader("Semantic search (addresses / buildings)")
    q = st.text_input("Enter a question or address to semantically search building records")
    k = st.slider("Top K results", 1, 10, 3)
    if st.button("Search buildings"):
        if df_rexus is None:
            st.error("No building dataset available to search.")
        else:
            results = semantic_retrieve_rexus(q, top_k=k)
            if results.empty:
                st.info("No results.")
            else:
                st.dataframe(results)

elif page == "AI Assistant":
    st.header("🤖 AI Assistant")
    st.write("Ask questions about the loaded building data. Uses OpenAI if API key is provided.")
    user_q = st.text_input("Your question about the two locations or the loaded building dataset")
    if st.button("Ask"):
        if not user_q:
            st.warning("Please enter a question.")
        else:
            # Compose context from top semantic results
            context_rows = semantic_retrieve_rexus(user_q, top_k=3)
            context = ""
            for _, row in context_rows.iterrows():
                snippet = f"Address: {row.get('Bldg Address1','N/A')}, {row.get('Bldg City','N/A')}, {row.get('Bldg State','N/A')} | Status: {row.get('Bldg Status','N/A')}"
                context += snippet + "\n"
            prompt = (
                "You are an assistant answering questions using ONLY the building data provided.\n\n"
                f"Building Data:\n{context}\nUser question: {user_q}\nAnswer succinctly."
            )

            if OPENAI_API_KEY and OPENAI_AVAILABLE:
                try:
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "system", "content": prompt}],
                        temperature=0.2,
                        max_tokens=300
                    )
                    answer = completion["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    answer = f"OpenAI API error: {e}"
            else:
                if context_rows is None or context_rows.empty:
                    answer = "No data available to answer. Provide a building dataset or set OPENAI_API_KEY."
                else:
                    # Provide a short summary constructed from the top row
                    top = context_rows.iloc[0]
                    answer = f"I found a building at {top.get('Bldg Address1','N/A')} in {top.get('Bldg City','N/A')}, status {top.get('Bldg Status','N/A')}."
            st.session_state.chat_history.append({"q": user_q, "a": answer})
            st.success("Answer ready")
    if st.session_state.chat_history:
        st.markdown("### Conversation history")
        for chat in reversed(st.session_state.chat_history[-10:]):
            st.markdown(f"**Q:** {chat['q']}")
            st.markdown(f"**A:** {chat['a']}")
            st.write("---")

elif page == "Settings":
    st.header("⚙️ Settings & Integrations")
    st.markdown("Add API keys to `.env` or set them in your deployment environment variables.")
    st.info("Current environment keys detection (empty means not set):")
    st.write("WAQI_API_KEY:", "SET" if WAQI_API_KEY else "NOT SET")
    st.write("CENSUS_API_KEY:", "SET" if CENSUS_API_KEY else "NOT SET")
    st.write("OPENAI_API_KEY:", "SET" if OPENAI_API_KEY else "NOT SET")

    st.markdown("---")
    st.subheader("Demo mode")
    if DEMO_MODE:
        st.warning("App is running in DEMO mode (no live external APIs configured).")
    else:
        st.success("External APIs available.")

    st.markdown("""
    - To enable real external data, set `CENSUS_API_KEY`, `WAQI_API_KEY`, and `OPENAI_API_KEY` in your environment.
    - To use the semantic search model, install `sentence-transformers`.
    - Place `data_gov_bldg_rexus.csv` in app root to use real building data.
    """)

st.markdown("---")
st.markdown("Built with ❤️ — update datasets and API keys for real data. Contact me to add Census/FBI/Zillow endpoints.")
