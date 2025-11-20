import os
import re
import numpy as np
import pandas as pd
from typing import Optional, Tuple

def safe_float(x, default=float("nan")):
    try:
        if pd.isna(x):
            return default
        return float(str(x).replace(",", "").replace("%", ""))
    except Exception:
        return default

def sanitize_location_text(location: str) -> Tuple[str, str]:
    """Return (city, state) parsed from 'City, ST'."""
    if not location:
        return "", ""
    parts = [p.strip() for p in location.split(",")]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return parts[0], ""

def load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    """Load CSV returning DataFrame or None. Use low_memory to handle wide files."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, dtype=str, low_memory=False)
    except Exception:
        try:
            # second attempt with default engine
            return pd.read_csv(path, dtype=str, engine='python')
        except Exception:
            return None

def geocode_city_state(city: str, state: str) -> Tuple[Optional[float], Optional[float]]:
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

def get_safety_data(city: str, state: str = "") -> dict:
    city_key = (city or "").strip().lower()
    BASE_SAFE_SCORE = 72
    CITY_CRIME_PROFILE = {
        "seattle":      {"violent": 5.1, "property": 55.0, "adj": +3},
        "portland":     {"violent": 5.2, "property": 63.0, "adj": -2},
        "new york":     {"violent": 3.1, "property": 23.0, "adj": +5},
        "san francisco":{"violent": 6.5, "property": 80.0, "adj": -5},
        "los angeles":  {"violent": 5.9, "property": 54.0, "adj": -3},
        "chicago":      {"violent": 9.9, "property": 46.0, "adj": -8},
        "boston":       {"violent": 3.9, "property": 24.0, "adj": +7},
        "austin":       {"violent": 4.3, "property": 36.0, "adj": +4},
        "denver":       {"violent": 5.5, "property": 53.0, "adj": +1}
    }

    default_profile = {"violent": 4.5, "property": 30.0, "adj": 0}
    profile = CITY_CRIME_PROFILE.get(city_key, default_profile)

    violent_score = max(0, 100 - (profile["violent"] * 7))
    property_score = max(0, 100 - (profile["property"] * 1.2))

    total_safety = int(
        violent_score * 0.55 +
        property_score * 0.40 +
        BASE_SAFE_SCORE * 0.05
    )
    total_safety = max(1, min(95, total_safety + profile["adj"]))

    # Crime trend simulation
    trend = round((profile["violent"] - 4.0) * 3, 1)
    trend_str = (f"+{trend}%" if trend >= 0 else f"{trend}%") + " YoY"

    # Severity label
    if total_safety > 80:
        severity = "Very Safe"
    elif total_safety > 70:
        severity = "Safe"
    elif total_safety > 60:
        severity = "Moderately Safe"
    elif total_safety > 50:
        severity = "Some Risk"
    else:
        severity = "High Risk"

    return {
        "crime_index": total_safety,
        "severity": severity,
        "violent_crime_rate": f"{profile['violent']} per 1,000",
        "property_crime_rate": f"{profile['property']} per 1,000",
        "crime_trend": trend_str,
        "police_response": f"{8 + int(profile['violent'])} min avg",
        "neighborhood_watch": f"{int(total_safety/2)} groups"
    }

def get_quality_data(lat: float, lon: float) -> dict:
    try:
        distance_from_coast = abs((lon + 100) / 20)
        urban_density = abs(40 - lat) / 10
        base = max(50, min(95, 75 - distance_from_coast + urban_density))

        walkability = int(min(100, base * (1 + urban_density/20)))
        air_quality = int(min(100, base - (distance_from_coast/2)))
        parks = int(min(100, base / 8 + urban_density))
        restaurants = int(min(100, base * (1.5 + urban_density/10)))
        commute = int(max(10, min(90, 35 - base/4 + urban_density)))
        transit = int(min(100, base * (0.8 + urban_density/15)))
        healthcare = int(min(100, base * (0.9 + urban_density/20)))

        return {
            "walkability": walkability,
            "air_quality": air_quality,
            "parks": parks,
            "restaurants": restaurants,
            "commute_time": f"{commute} min avg",
            "transit": transit,
            "healthcare": healthcare
        }
    except Exception:
        return {}

def get_education(city: str) -> dict:
    return {
        "district_name": f"{city} School District" if city else "Local School District",
        "highest_ranked_school": f"{city} High School" if city else "Local High School",
        "school_rank": "#12",
        "school_rating": "8.2/10",
        "total_schools": "35"
    }

MONTH_COL_REGEX = re.compile(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s\-]?\d{2,4}$', re.IGNORECASE)

def _identify_date_columns(df: pd.DataFrame):
    """Return list of column names that look like month-year (e.g., 'Oct-16' or 'October 2016')."""
    date_cols = []
    for c in df.columns:
        if MONTH_COL_REGEX.match(c.strip()):
            date_cols.append(c)
            continue
        # also try to parse columns that look like MMM-YYYY or MMM-YY numeric-ish
        try:
            # try converting sample header by appending '1 ' to become a date string
            pd.to_datetime("1 " + c, errors='coerce')
            date_cols.append(c)
        except Exception:
            pass
    # fallback: if many columns and first 6 are metadata, assume rest are time series
    if not date_cols and df.shape[1] > 6:
        date_cols = list(df.columns[6:])
    return date_cols

def _parse_timeseries_from_row(row: pd.Series, date_cols):
    """Return pd.Series indexed by datetime constructed from date_cols for this row."""
    vals = row.loc[date_cols].replace("", np.nan).map(lambda x: str(x).replace(",", "").strip())
    # convert to numeric
    numeric = pd.to_numeric(vals, errors='coerce')
    # parse index to dates robustly
    # try multiple parse strategies:
    parsed = pd.to_datetime(date_cols, errors='coerce', infer_datetime_format=True)
    if parsed.isna().all():
        # try adding day '1 ' prefix
        parsed = pd.to_datetime(["1 " + c for c in date_cols], errors='coerce', infer_datetime_format=True)
    # final fallback: use integer positions as index
    if parsed.isna().all():
        idx = pd.Index(range(len(date_cols)))
    else:
        idx = parsed
    s = pd.Series(data=numeric.values, index=idx)
    # drop missing entirely
    s = s.dropna()
    return s

def get_price_data_for_city(city: str, state: str, df_price: pd.DataFrame) -> dict:
    """
    Returns:
      - latest_price: last available non-NaN value (string)
      - median_price: median of the timeseries (string)
      - price_timeseries: pd.Series indexed by datetime (or numeric index) with floats
    """
    if df_price is None:
        return {"latest_price": "No data", "median_price": "No data", "price_timeseries": None}

    df = df_price.copy()

    # Normalize column names to avoid leading/trailing spaces
    df.columns = [c.strip() for c in df.columns]

    # Identify date columns automatically
    date_cols = _identify_date_columns(df)

    # Candidate matching columns for City and State
    city_cols = [c for c in df.columns if c.lower() in ("city", "city name", "place", "city_name", "citycode", "city code") or "city" in c.lower()]
    state_cols = [c for c in df.columns if c.lower() in ("state", "st", "state_code") or "state" in c.lower()]

    # Fallback to common names
    city_col = city_cols[0] if city_cols else (df.columns[0] if len(df.columns) > 0 else None)
    state_col = state_cols[0] if state_cols else (df.columns[4] if df.shape[1] > 4 else None)

    # Create boolean masks safely
    mask_city = False
    mask_state = False
    try:
        if city_col:
            mask_city = df[city_col].fillna("").astype(str).str.strip().str.lower() == (city or "").strip().lower()
        if state_col:
            mask_state = df[state_col].fillna("").astype(str).str.strip().str.lower() == (state or "").strip().lower()
    except Exception:
        mask_city = False
        mask_state = False

    matches = df[mask_city & mask_state] if (isinstance(mask_city, (pd.Series, np.ndarray)) and isinstance(mask_state, (pd.Series, np.ndarray))) else pd.DataFrame()

    if matches.empty and isinstance(mask_city, (pd.Series, np.ndarray)):
        matches = df[mask_city]

    # If still empty, fallback to nearest by city substring match
    if matches.empty and city:
        try:
            matches = df[df[city_col].fillna("").astype(str).str.lower().str.contains(city.strip().lower(), na=False)]
        except Exception:
            matches = pd.DataFrame()

    # If still empty, use first row as fallback
    if matches.empty:
        if df.shape[0] > 0:
            row = df.iloc[0]
        else:
            return {"latest_price": "No data", "median_price": "No data", "price_timeseries": None}
    else:
        row = matches.iloc[0]

    # Build timeseries Series
    if date_cols:
        ts = _parse_timeseries_from_row(row, date_cols)
    else:
        ts = pd.Series(dtype=float)

    # compute latest and median
    latest_val = None
    if not ts.empty:
        latest_val = ts.iloc[-1]
        median_val = float(ts.median(skipna=True)) if not ts.empty else None
    else:
        # fallback to explicit LatestPrice / MedianPrice columns if present
        latest_val = row.get("Latest Price") or row.get("LatestPrice") or row.get("Latest_Price") or None
        median_val = row.get("Median Price") or row.get("MedianPrice") or None
        if latest_val is not None:
            try:
                latest_val = float(str(latest_val).replace(",", ""))
            except Exception:
                pass

    # Return clean results
    latest_str = f"{latest_val:.2f}" if isinstance(latest_val, (int, float, np.floating)) else (str(latest_val) if latest_val is not None else "No data")
    median_str = f"{median_val:.2f}" if isinstance(median_val, (int, float, np.floating)) else (str(median_val) if median_val is not None else "No data")

    return {
        "latest_price": latest_str,
        "median_price": median_str,
        "price_timeseries": ts
    }

def get_real_estate_data(city: str, state: str, df_rexus: pd.DataFrame) -> dict:
    """
    Placeholder for building registry retrieval.
    Returns first matching row from df_rexus or empty dict.
    """
    if df_rexus is None or df_rexus.empty:
        return {}

    # normalize columns
    cols = {c: c for c in df_rexus.columns}
    if "Bldg City" not in df_rexus.columns:
        # try to find reasonable city column
        possible = [c for c in df_rexus.columns if "city" in c.lower()]
        if possible:
            cols["Bldg City"] = possible[0]
    if "Bldg State" not in df_rexus.columns:
        possible = [c for c in df_rexus.columns if "state" in c.lower()]
        if possible:
            cols["Bldg State"] = possible[0]

    bcity_col = cols.get("Bldg City", None)
    bstate_col = cols.get("Bldg State", None)

    mask_city = df_rexus[bcity_col].fillna("").astype(str).str.strip().str.upper() == city.strip().upper() if bcity_col else False
    mask_state = df_rexus[bstate_col].fillna("").astype(str).str.strip().str.upper() == state.strip().upper() if bstate_col else False

    try:
        matches = df_rexus[mask_city & mask_state] if isinstance(mask_city, (pd.Series, np.ndarray)) and isinstance(mask_state, (pd.Series, np.ndarray)) else pd.DataFrame()
    except Exception:
        matches = pd.DataFrame()

    if matches.empty and isinstance(mask_city, (pd.Series, np.ndarray)):
        matches = df_rexus[mask_city]

    if matches.empty:
        return df_rexus.iloc[0].to_dict()
    return matches.iloc[0].to_dict()

def semantic_retrieve_rexus(query: str, df: pd.DataFrame, embeddings=None, model=None, top_k: int = 5) -> pd.DataFrame:
    """
    If embeddings + model are provided, a real vector similarity should be used.
    Otherwise do a simple case-insensitive substring match across all fields.
    """
    if df is None or not query:
        return pd.DataFrame()

    if embeddings is not None and model is not None:
        # Placeholder: return top_k rows (replace with actual similarity search)
        return df.head(top_k)

    mask = df.apply(lambda row: row.astype(str).str.contains(query, case=False, na=False).any(), axis=1)
    return df[mask].head(top_k)
