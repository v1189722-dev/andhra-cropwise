from pathlib import Path
import sys
import json

import folium
import joblib
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.yield_lookup import get_yield_comparison


MODEL_PATH = PROJECT_DIR / "models" / "crop_recommender.joblib"
PRICE_MODEL_PATH = PROJECT_DIR / "models" / "price_predictor.joblib"
PRICE_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "andhra_pradesh_prices_available.csv"
)
MAP_PATH = (
    PROJECT_DIR
    / "data"
    / "maps"
    / "andhra_pradesh_districts.geojson"
)

DISTRICTS = [
    "Alluri Sitharama Raju",
    "Anakapalli",
    "Ananthapuramu",
    "Annamayya",
    "Bapatla",
    "Chittoor",
    "East Godavari",
    "Eluru",
    "Guntur",
    "Kadapa",
    "Kakinada",
    "Konaseema",
    "Krishna",
    "Kurnool",
    "NTR",
    "Nandyal",
    "Palnadu",
    "Parvathipuram Manyam",
    "Prakasam",
    "Nellore",
    "Srikakulam",
    "Srisathyasai",
    "Tirupati",
    "Visakhapatnam",
    "Vizianagaram",
    "West Godavari",
]

SEASONS = ["Kharif", "Rabi", "Summer"]


@st.cache_resource
def load_crop_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train it first."
        )
    return joblib.load(MODEL_PATH)
@st.cache_resource
def load_price_model():
    if not PRICE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Price model not found at {PRICE_MODEL_PATH}. "
            "Run src\\train_price_model.py first."
        )
    return joblib.load(PRICE_MODEL_PATH)


@st.cache_data
def load_prices():
    if not PRICE_PATH.exists():
        return pd.DataFrame()

    prices = pd.read_csv(PRICE_PATH)
    prices.columns = prices.columns.astype(str).str.strip()

    required_columns = {
        "Arrival_Date",
        "District",
        "Market",
        "Commodity",
        "Modal Price",
    }
    missing = required_columns - set(prices.columns)
    if missing:
        raise ValueError(f"Price file is missing columns: {sorted(missing)}")

    prices["Arrival_Date"] = pd.to_datetime(
        prices["Arrival_Date"],
        errors="coerce",
    )
    prices["Modal Price"] = pd.to_numeric(
        prices["Modal Price"],
        errors="coerce",
    )
    prices = prices.dropna(
        subset=["Arrival_Date", "District", "Commodity", "Modal Price"]
    )
    return prices[prices["Modal Price"] > 0].copy()


def normalize_key(value: object) -> str:
    return "".join(
        character
        for character in str(value).casefold()
        if character.isalnum()
    )


CROP_PRICE_TERMS = {
    "rice": ["paddy", "rice"],
    "maize": ["maize", "corn"],
    "chickpea": ["bengalgram", "chickpea"],
    "kidneybeans": ["kidneybean", "rajma"],
    "pigeonpeas": ["pigeonpea", "arhar", "redgram"],
    "mothbeans": ["mothbean"],
    "mungbean": ["mungbean", "greengram", "moong"],
    "blackgram": ["blackgram", "urd"],
    "lentil": ["lentil", "masoor"],
    "pomegranate": ["pomegranate"],
    "banana": ["banana"],
    "mango": ["mango"],
    "grapes": ["grape"],
    "watermelon": ["watermelon"],
    "muskmelon": ["muskmelon"],
    "apple": ["apple"],
    "orange": ["orange"],
    "papaya": ["papaya"],
    "coconut": ["coconut"],
    "cotton": ["cotton", "kapas"],
    "jute": ["jute"],
    "coffee": ["coffee"],
}

DISTRICT_PRICE_NAMES = {
    "chittoor": ["chittoor", "chittor"],
    "kadapa": ["kadapa", "cuddapah"],
    "ananthapuramu": ["ananthapuramu", "anantapur"],
    "nellore": ["nellore", "spsr nellore"],
    "vizianagaram": ["vizianagaram", "vijayanagaram"],
}


def find_crop_observations(
    prices: pd.DataFrame,
    crop: str,
) -> pd.DataFrame:
    if prices.empty:
        return prices

    crop_key = normalize_key(crop)
    terms = CROP_PRICE_TERMS.get(crop_key, [crop_key])
    terms = [normalize_key(term) for term in terms]

    commodity_keys = prices["Commodity"].fillna("").map(normalize_key)
    matches = commodity_keys.map(
        lambda commodity: any(term in commodity for term in terms)
    )
    return prices[matches].copy()


def find_price_observations(
    prices: pd.DataFrame,
    district: str,
    crop: str,
) -> pd.DataFrame:
    if prices.empty:
        return prices

    district_key = normalize_key(district)
    district_names = DISTRICT_PRICE_NAMES.get(
        district_key,
        [district_key],
    )
    district_names = [normalize_key(name) for name in district_names]

    district_matches = prices["District"].fillna("").map(normalize_key).isin(
        district_names
    )
    crop_matches = find_crop_observations(prices, crop)

    return prices.loc[
        district_matches & prices.index.isin(crop_matches.index)
    ].sort_values("Arrival_Date")


def get_price_estimate(
    prices: pd.DataFrame,
    district: str,
    crop: str,
) -> tuple[float, str, pd.DataFrame]:
    if prices.empty:
        raise ValueError(
            "The prepared Andhra Pradesh price file has no usable price rows. "
            "Run src\\prepare_data.py first."
        )

    district_crop_prices = find_price_observations(
        prices,
        district,
        crop,
    )

    if not district_crop_prices.empty:
        candidates = district_crop_prices
        source = f"Recent market records matching {district} and {crop}"
    else:
        ap_crop_prices = find_crop_observations(prices, crop)

        if not ap_crop_prices.empty:
            candidates = ap_crop_prices
            source = (
                f"Estimated from Andhra Pradesh market records for {crop}; "
                f"no matching district record was found for {district}"
            )
        else:
            candidates = prices
            source = (
                "Low-confidence Andhra Pradesh market benchmark; "
                f"no matching {crop} price was found"
            )

    candidates = candidates[
        pd.to_numeric(candidates["Modal Price"], errors="coerce") > 0
    ].copy()

    if candidates.empty:
        candidates = prices[prices["Modal Price"] > 0].copy()
        source = "Low-confidence estimate from available AP market prices"

    latest_date = candidates["Arrival_Date"].max()
    recent_start = latest_date - pd.Timedelta(days=365)
    recent = candidates[candidates["Arrival_Date"] >= recent_start]

    if not recent.empty:
        candidates = recent

    estimate = float(candidates["Modal Price"].median())
    return estimate, source, candidates.sort_values("Arrival_Date")
def predict_price_with_model(
    prices: pd.DataFrame,
    district: str,
    crop: str,
) -> tuple[float, str, pd.DataFrame]:
    if prices.empty:
        raise ValueError("No usable Andhra Pradesh market price rows were found.")

    bundle = load_price_model()
    model = bundle["model"]
    features = bundle["features"]

    # Use price-file crop names the model learned from.
    crop_rows = find_crop_observations(prices, crop)

    if crop_rows.empty:
        # No matching crop prices exist, so a crop-specific model prediction
        # cannot be made. Use the existing broad market estimate instead.
        return get_price_estimate(prices, district, crop)

    # Match the app district to its spelling in the price file when possible.
    district_key = normalize_key(district)
    district_names = DISTRICT_PRICE_NAMES.get(district_key, [district])
    district_names = [normalize_key(name) for name in district_names]

    district_rows = prices[
        prices["District"].fillna("").map(normalize_key).isin(district_names)
    ]

    if not district_rows.empty:
        model_district = str(district_rows["District"].iloc[0])
        district_note = (
            f"XGBoost estimate using price-file district name "
            f"{model_district}."
        )
    else:
        model_district = district
        district_note = (
            f"XGBoost estimate for {district}; the price file has no direct "
            "records for this district, so this is less reliable."
        )

    # Predict for the crop commodity names in the prepared price file, then
    # use their median prediction for the card.
    commodities = sorted(
        crop_rows["Commodity"].dropna().astype(str).unique().tolist()
    )

    latest_date = prices["Arrival_Date"].max()
    prediction_date = latest_date + pd.Timedelta(days=1)

    prediction_rows = pd.DataFrame(
        {
            "District": [model_district] * len(commodities),
            "Commodity": commodities,
            "Month": [prediction_date.month] * len(commodities),
            "DayOfYear": [prediction_date.dayofyear] * len(commodities),
        }
    )

    prediction_rows = prediction_rows[features]
    predictions = model.predict(prediction_rows)
    estimate = max(0.0, float(pd.Series(predictions).median()))

    if not district_rows.empty:
        chart_rows = find_price_observations(prices, district, crop)
    else:
        chart_rows = crop_rows.sort_values("Arrival_Date")

    if chart_rows.empty:
        chart_rows = crop_rows.sort_values("Arrival_Date")

    source = (
        f"{district_note} Predicted for {prediction_date.date()}. "
        "Validation MAE was ₹1,608.71/quintal."
    )

    return estimate, source, chart_rows


st.set_page_config(
    page_title="CropWise AP",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 CropWise AP")
st.caption("Crop recommendation and historical agriculture information")

with st.sidebar:
    st.header("Field details")

    with st.form("field_inputs"):
        district = st.selectbox("District", DISTRICTS)
        season = st.selectbox("Season", SEASONS)

        st.subheader("Soil nutrients")
        n_value = st.number_input("Nitrogen (N)", min_value=0.0, value=90.0)
        p_value = st.number_input("Phosphorus (P)", min_value=0.0, value=42.0)
        k_value = st.number_input("Potassium (K)", min_value=0.0, value=43.0)

        st.subheader("Soil and weather")
        ph_value = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5,
            step=0.1,
        )
        temperature = st.number_input(
            "Temperature (°C)",
            value=27.5,
            step=0.1,
        )
        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
        )
        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=203.0,
        )

        submitted = st.form_submit_button(
            "Get crop recommendation",
            type="primary",
            use_container_width=True,
        )


if submitted:
    try:
        saved = load_crop_model()
        model = saved["model"]
        features = saved["features"]

        input_data = pd.DataFrame(
            [[
                n_value,
                p_value,
                k_value,
                temperature,
                humidity,
                ph_value,
                rainfall,
            ]],
            columns=features,
        )

        recommended_crop = str(model.predict(input_data)[0])
        st.session_state["result"] = {
            "district": district,
            "season": season,
            "crop": recommended_crop,
        }
    except Exception as error:
        st.error(f"Could not make a recommendation: {error}")


if "result" in st.session_state:
    result = st.session_state["result"]
    district = result["district"]
    season = result["season"]
    recommended_crop = result["crop"]

    st.subheader("Your results")

    recommendation_col, yield_col, price_col = st.columns(3)

    with recommendation_col:
        st.metric("Recommended crop", recommended_crop.title())
        st.caption("Based on the soil and weather values you entered.")

    with yield_col:
        try:
            yield_result = get_yield_comparison(
                district=district,
                crop=recommended_crop,
                season=season,
            )

            district_yield = yield_result[
                "district_yield_tonnes_per_hectare"
            ]
            ap_average = yield_result[
                "ap_historical_average_tonnes_per_hectare"
            ]
            yield_message = yield_result["message"]

            if district_yield is None:
                if ap_average is not None:
                    st.metric(
                        "Estimated yield (AP average)",
                        f"{ap_average:.3f} t/ha",
                    )

                    if "coffee board" in yield_message.casefold():
                        st.caption(
                            "Using the statewide Coffee Board benchmark; "
                            "it is not district-specific."
                        )
                    elif "across seasons" in yield_message.casefold():
                        st.caption(
                            "Using the AP historical average for this crop "
                            "across seasons."
                        )
                    else:
                        st.caption(
                            "Using the AP historical average for this crop "
                            "and season because district records are missing."
                        )
                else:
                    st.metric("Historic district yield", "Unavailable")
            else:
                st.metric(
                    "Historic district yield",
                    f"{district_yield:.3f} t/ha",
                )
                if ap_average is not None:
                    st.caption(
                        f"AP historical average for this crop and season: "
                        f"{ap_average:.3f} t/ha"
                    )

            st.caption(yield_message)

        except Exception as error:
            st.warning(f"Could not load historic yield: {error}")

    prices = load_prices()

    try:
        price_estimate, price_source, price_rows = predict_price_with_model(
            prices,
            district=district,
            crop=recommended_crop,
        )

        with price_col:
            st.metric(
                "Estimated market price",
                f"₹{price_estimate:,.0f}/quintal",
            )
            st.caption(price_source)
            st.caption(
                "Estimate from historical market observations; "
                "not a guaranteed future price."
            )

        st.subheader("Price trend and supporting market records")

        daily_median = (
            price_rows.groupby("Arrival_Date")["Modal Price"]
            .median()
            .sort_index()
        )
        st.line_chart(daily_median)

        display = price_rows[
            ["Arrival_Date", "District", "Market", "Commodity", "Modal Price"]
        ].sort_values("Arrival_Date", ascending=False).head(20).copy()

        display = display.rename(
            columns={
                "Arrival_Date": "Date",
                "Commodity": "Crop in price file",
                "Modal Price": "Modal price (₹/quintal)",
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

    except Exception as error:
        with price_col:
            st.error(f"Could not calculate a price estimate: {error}")

    st.subheader("Andhra Pradesh district map")

    if not MAP_PATH.exists():
        st.warning(f"Map boundary file not found: {MAP_PATH}")
    else:
        with MAP_PATH.open("r", encoding="utf-8") as map_file:
            district_geojson = json.load(map_file)

        selected_key = normalize_key(district)

        boundary_name_aliases = {
            "konaseema": "drbrambedkarkonaseema",
            "nellore": "sripottisriramulunellore",
            "kadapa": "ysrkadapa",
        }
        selected_key = boundary_name_aliases.get(selected_key, selected_key)

        for feature in district_geojson.get("features", []):
            properties = feature.setdefault("properties", {})
            boundary_name = properties.get(
                "district_name",
                properties.get("NEW_DIST", "Unknown district"),
            )
            properties["district_name"] = boundary_name
            properties["_is_selected"] = (
                normalize_key(boundary_name) == selected_key
            )

        ap_map = folium.Map(
            location=[15.8, 79.7],
            zoom_start=6,
            tiles="OpenStreetMap",
        )

        folium.GeoJson(
            district_geojson,
            name="Andhra Pradesh districts",
            style_function=lambda feature: {
                "fillColor": (
                    "#16a34a"
                    if feature["properties"].get("_is_selected")
                    else "#93c5fd"
                ),
                "color": "#334155",
                "weight": 1,
                "fillOpacity": 0.75,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["district_name"],
                aliases=["District:"],
            ),
        ).add_to(ap_map)

        st_folium(
            ap_map,
            height=520,
            use_container_width=True,
            key="andhra_pradesh_district_map",
        )

else:
    st.info(
        "Choose your district, season, and field measurements in the sidebar, "
        "then select **Get crop recommendation**."
    )

st.divider()
st.caption(
    "Historic yield records cover 13 older district names through 2019. "
    "Newer districts may not have separate historic yield records."
)