"""
Feature module for the Farm AI Assistant (Plant Diagnoser) application.
Integrates Gemini multimodal analysis, EXIF GPS extraction, Google Maps reverse
geocoding, and Google Maps Platform Weather API for crop, soil, pest, and irrigation advice.
"""
import base64
from datetime import datetime, timedelta, timezone
import io
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import google.generativeai as genai
import googlemaps
from PIL import ExifTags, Image
import requests
import streamlit as st

load_dotenv()


def load_api_keys() -> TupleKeys:
    """
    Loads GEMINI_API_KEY and GOOGLE_MAPS_API_KEY from environment variables
    or Streamlit secrets.

    Returns:
        tuple: (gemini_api_key, google_maps_api_key)
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    maps_key = os.getenv("GOOGLE_MAPS_API_KEY")

    if not gemini_key or not maps_key:
        try:
            gemini_key = gemini_key or st.secrets.get("GEMINI_API_KEY")
            maps_key = maps_key or st.secrets.get("GOOGLE_MAPS_API_KEY")
        except Exception:
            pass

    return gemini_key, maps_key


# Type alias for key pair
TupleKeys = tuple[Optional[str], Optional[str]]


def get_image_geotag(image: Image.Image) -> Optional[Dict[str, float]]:
    """
    Extracts GPS latitude and longitude from an image's EXIF metadata.

    Args:
        image (Image.Image): Opened PIL Image object.

    Returns:
        Optional[Dict[str, float]]: Dictionary with 'lat' and 'lon', or None.
    """
    try:
        exif_data = image._getexif()  # noqa: SLF001
        if not exif_data:
            return None
        exif = {
            ExifTags.TAGS[k]: v for k, v in exif_data.items() if k in ExifTags.TAGS
        }
        gps_info = exif.get("GPSInfo")
        if not gps_info:
            return None

        def dms_to_decimal(dms: Any, ref: str) -> float:
            decimal = float(dms[0]) + float(dms[1]) / 60.0 + float(dms[2]) / 3600.0
            if ref in ["S", "W"]:
                decimal = -decimal
            return decimal

        lat = dms_to_decimal(gps_info[2], gps_info[1])
        lon = dms_to_decimal(gps_info[4], gps_info[3])
        return {"lat": lat, "lon": lon}
    except (AttributeError, KeyError, IndexError, TypeError):
        return None


def get_location_from_coords(
    gmaps: googlemaps.Client, lat: float, lon: float
) -> Dict[str, str]:
    """
    Uses Google Maps Geocoding API to convert coordinates into a human-readable address.

    Args:
        gmaps (googlemaps.Client): Initialized Google Maps client.
        lat (float): Latitude.
        lon (float): Longitude.

    Returns:
        Dict[str, str]: Dictionary containing 'address' or 'error'.
    """
    try:
        reverse_geocode_result = gmaps.reverse_geocode((lat, lon))
        if reverse_geocode_result:
            return {"address": reverse_geocode_result[0]["formatted_address"]}
        return {"error": "Could not determine address."}
    except Exception as exc:
        return {"error": f"Google Maps API error: {exc}"}


def get_weather(lat: float, lon: float, api_key: str) -> Dict[str, Any]:
    """
    Fetches current weather conditions from the Google Maps Platform Weather API.

    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        api_key (str): Google Maps Platform API key.

    Returns:
        Dict[str, Any]: Weather metrics or error details.
    """
    url = (
        f"https://weather.googleapis.com/v1/currentConditions:lookup"
        f"?key={api_key}&location.latitude={lat}&location.longitude={lon}"
    )
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        temp = data.get("temperature", {}).get("degrees")
        humidity = data.get("relativeHumidity")
        description = (
            data.get("weatherCondition", {})
            .get("description", {})
            .get("text", "N/A")
            .capitalize()
        )

        return {
            "temperature": temp,
            "humidity": humidity,
            "description": description,
        }
    except requests.exceptions.HTTPError:
        error_details = "No specific error details found."
        try:
            error_json = response.json()
            if "error" in error_json and "message" in error_json["error"]:
                error_details = error_json["error"]["message"]
        except ValueError:
            pass
        return {"error": f"({response.status_code}) {error_details}"}
    except Exception as exc:
        return {"error": f"Could not fetch Google weather data. Error: {exc}"}


def get_historical_weather(
    lat: float, lon: float, api_key: str
) -> Optional[List[Dict[str, Any]]]:
    """
    Fetches historical weather data for the past 7 days.

    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        api_key (str): Google Maps Platform API key.

    Returns:
        Optional[List[Dict[str, Any]]]: List of historical weather records, or None.
    """
    today = datetime.now(timezone.utc)
    past_date = today - timedelta(days=7)
    url = (
        f"https://weather.googleapis.com/v1/history:lookup"
        f"?key={api_key}&location.latitude={lat}&location.longitude={lon}"
        f"&from.year={past_date.year}&from.month={past_date.month}&from.day={past_date.day}"
        f"&to.year={today.year}&to.month={today.month}&to.day={today.day}"
    )
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get("history", [])
    except Exception:
        return None


def get_gemini_response(image_data: Optional[Dict[str, str]], prompt: str) -> str:
    """
    Generates content from Gemini 2.5 Pro given an optional image payload and text prompt.

    Args:
        image_data (Optional[Dict[str, str]]): Base64-encoded JPEG dictionary or None.
        prompt (str): User prompt.

    Returns:
        str: Model response text.
    """
    model = genai.GenerativeModel("gemini-2.5-pro")
    if image_data:
        response = model.generate_content([prompt, image_data])
    else:
        response = model.generate_content(prompt)
    return response.text


def image_to_dict(image: Image.Image) -> Dict[str, str]:
    """
    Serializes a PIL Image into a base64-encoded JPEG dictionary for Gemini.

    Args:
        image (Image.Image): PIL Image instance.

    Returns:
        Dict[str, str]: Dict with 'mime_type' and base64 'data'.
    """
    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"mime_type": "image/jpeg", "data": img_str}


def render_farm_assistant_app() -> None:
    """
    Renders the Farm AI Assistant Streamlit interface.
    """
    st.set_page_config(
        page_title="Farm AI Assistant", page_icon="🧑‍🌾", layout="wide"
    )
    st.title("🧑‍🌾 Farm AI Assistant")
    st.markdown(
        "Your comprehensive tool for crop diagnosis, soil analysis, and farm management."
    )

    gemini_api_key, google_maps_api_key = load_api_keys()
    if not gemini_api_key or not google_maps_api_key:
        st.error(
            "API Keys not found. Please set `GEMINI_API_KEY` and `GOOGLE_MAPS_API_KEY` "
            "in `.env` or `.streamlit/secrets.toml`."
        )
        st.stop()

    genai.configure(api_key=gemini_api_key)
    gmaps = googlemaps.Client(key=google_maps_api_key)

    if "context_info" not in st.session_state:
        st.session_state.context_info = ""
    if "lat" not in st.session_state:
        st.session_state.lat = None
    if "lon" not in st.session_state:
        st.session_state.lon = None
    if "address" not in st.session_state:
        st.session_state.address = None
    if "plant_prompt" not in st.session_state:
        st.session_state.plant_prompt = ""

    image = None
    with st.sidebar:
        st.header("1. Upload Image")
        uploaded_file = st.file_uploader(
            "Upload an image of a plant, soil, or pest.",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

        st.header("2. Set Location")
        if uploaded_file and image:
            with st.spinner("Checking for image location data..."):
                geotag = get_image_geotag(image)
                if geotag:
                    st.session_state.lat = geotag["lat"]
                    st.session_state.lon = geotag["lon"]
                    location_data = get_location_from_coords(
                        gmaps, st.session_state.lat, st.session_state.lon
                    )
                    if "address" in location_data:
                        st.session_state.address = location_data["address"]
                        st.success(
                            f"📍 Location found in image: {st.session_state.address}"
                        )

        if not st.session_state.address:
            st.warning("No GPS data in image. Please provide location manually.")
            location_input = st.text_input(
                "Enter your city or address:", "Gurugram, India"
            )
            if st.button("Confirm Location"):
                geocode_result = gmaps.geocode(location_input)
                if not geocode_result:
                    st.error(f"Could not find location: '{location_input}'.")
                else:
                    st.session_state.lat = geocode_result[0]["geometry"]["location"][
                        "lat"
                    ]
                    st.session_state.lon = geocode_result[0]["geometry"]["location"][
                        "lng"
                    ]
                    st.session_state.address = geocode_result[0]["formatted_address"]
                    st.success(f"📍 Location confirmed: {st.session_state.address}")

        if st.session_state.address:
            with st.expander("Current Weather", expanded=True):
                weather_data = get_weather(
                    st.session_state.lat, st.session_state.lon, google_maps_api_key
                )
                if "error" in weather_data:
                    st.error(weather_data["error"])
                else:
                    temp = weather_data.get("temperature")
                    humidity = weather_data.get("humidity")
                    temp_str = f"{temp:.1f}°C" if temp is not None else "N/A"
                    humidity_str = (
                        f"{humidity:.0f}%" if humidity is not None else "N/A"
                    )
                    st.metric("Temperature", temp_str)
                    st.metric("Humidity", humidity_str)
                    st.info(f"Conditions: {weather_data['description']}")

                    st.session_state.context_info = (
                        f"For context, the location is near {st.session_state.address}. "
                        f"The current weather is: {temp_str}, {humidity_str} humidity, "
                        f"'{weather_data['description']}'."
                    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🩺 Plant Diagnosis",
            "🌱 Soil Analysis",
            "🐞 Pest & Weed ID",
            "💧 Irrigation Calculator",
            "📈 Climate Analysis",
        ]
    )

    with tab1:
        st.header("Plant Health Diagnosis")
        st.markdown(
            "Upload an image of a plant to diagnose diseases or nutrient deficiencies."
        )

        st.subheader("Example Prompts")
        b_col1, b_col2 = st.columns(2)
        prompts = {
            "Diagnose Health": (
                "What appears to be wrong with this plant? Provide a clear diagnosis "
                "and suggest practical ways to improve its health (watering, light, "
                "nutrients, pest control)."
            ),
            "Identify Plant": (
                "What is the species of this plant? Provide its common and scientific names."
            ),
            "Care Instructions": (
                "Provide detailed care instructions for this plant, including watering "
                "schedule, light requirements, and ideal soil type."
            ),
            "Check Toxicity": (
                "Is this plant toxic to pets (cats, dogs) or humans? If so, what are "
                "the symptoms of poisoning?"
            ),
        }

        if b_col1.button("Diagnose Health", key="diag_health_btn"):
            st.session_state.plant_prompt = prompts["Diagnose Health"]
        if b_col2.button("Identify Plant", key="id_plant_btn"):
            st.session_state.plant_prompt = prompts["Identify Plant"]
        if b_col1.button("Care Instructions", key="care_btn"):
            st.session_state.plant_prompt = prompts["Care Instructions"]
        if b_col2.button("Check Toxicity", key="toxic_btn"):
            st.session_state.plant_prompt = prompts["Check Toxicity"]

        prompt = st.text_area(
            "Describe the problem or ask a question about the plant:",
            key="plant_prompt",
            height=150,
        )

        if st.button("Get Plant Analysis", key="plant_btn"):
            if not uploaded_file or not image:
                st.warning("Please upload an image first.")
            elif not st.session_state.address:
                st.warning("Please set a location first.")
            elif not prompt:
                st.warning("Please enter a question or select an example prompt.")
            else:
                with st.spinner("Analyzing plant..."):
                    image_data = image_to_dict(image)
                    full_prompt = (
                        f"Diagnose the issue with this plant. {prompt}. "
                        f"{st.session_state.context_info}"
                    )
                    response = get_gemini_response(image_data, full_prompt)
                    st.success("Diagnosis Complete")
                    st.markdown(response)

    with tab2:
        st.header("Soil Health Analysis")
        st.markdown(
            "Upload a close-up image of your soil to analyze its type and health."
        )
        if st.button("Analyze Soil", key="soil_btn"):
            if not uploaded_file or not image:
                st.warning("Please upload a soil image first.")
            elif not st.session_state.address:
                st.warning("Please set a location first.")
            else:
                with st.spinner("Analyzing soil..."):
                    image_data = image_to_dict(image)
                    prompt = (
                        "Analyze this soil image. What is its likely composition "
                        "(e.g., sandy, clay, loam), what are its properties for "
                        "agriculture, and what crops would grow well here? "
                        f"{st.session_state.context_info}"
                    )
                    response = get_gemini_response(image_data, prompt)
                    st.success("Soil Analysis Complete")
                    st.markdown(response)

    with tab3:
        st.header("Pest and Weed Identification")
        st.markdown(
            "Upload an image of an insect or a weed for identification and control recommendations."
        )
        col_pest, col_weed = st.columns(2)
        if col_pest.button("Identify Pest", key="pest_btn"):
            if not uploaded_file or not image:
                st.warning("Please upload an image of the pest.")
            else:
                with st.spinner("Identifying pest..."):
                    image_data = image_to_dict(image)
                    prompt = (
                        "Identify this insect or pest. Is it harmful to crops? "
                        "What are the recommended organic and chemical control methods? "
                        f"{st.session_state.context_info}"
                    )
                    response = get_gemini_response(image_data, prompt)
                    st.success("Pest Identified")
                    st.markdown(response)

        if col_weed.button("Identify Weed", key="weed_btn"):
            if not uploaded_file or not image:
                st.warning("Please upload an image of the weed.")
            else:
                with st.spinner("Identifying weed..."):
                    image_data = image_to_dict(image)
                    prompt = (
                        "Identify this plant. Is it a weed? If so, how does it compete "
                        "with crops and what is the best way to remove it? "
                        f"{st.session_state.context_info}"
                    )
                    response = get_gemini_response(image_data, prompt)
                    st.success("Weed Identified")
                    st.markdown(response)

    with tab4:
        st.header("Irrigation Calculator")
        st.markdown(
            "Get a recommended watering schedule based on your crop and local weather."
        )
        crop_type = st.text_input("Enter your crop type (e.g., tomatoes, corn):")
        if st.button("Calculate Irrigation", key="irrigation_btn"):
            if not crop_type:
                st.warning("Please enter a crop type.")
            elif not st.session_state.address:
                st.warning("Please set a location first.")
            else:
                with st.spinner("Calculating schedule..."):
                    prompt = (
                        f"Create a 7-day irrigation plan for a {crop_type} field. "
                        f"Consider the provided context. {st.session_state.context_info}"
                    )
                    response = get_gemini_response(None, prompt)
                    st.success("Irrigation Plan Generated")
                    st.markdown(response)

    with tab5:
        st.header("Historical Climate Analysis")
        st.markdown(
            "Analyze weather trends for your location over the past week to inform decisions."
        )
        if st.button("Analyze Past Weather", key="history_btn"):
            if not st.session_state.address:
                st.warning("Please set a location first.")
            else:
                with st.spinner("Fetching historical data..."):
                    history = get_historical_weather(
                        st.session_state.lat,
                        st.session_state.lon,
                        google_maps_api_key,
                    )
                    if history:
                        prompt = (
                            "Analyze the following historical weather data for the past "
                            "week and provide insights for a farmer. What trends do you "
                            "see, and what impact could this have on crops? "
                            f"Data: {history}. {st.session_state.context_info}"
                        )
                        response = get_gemini_response(None, prompt)
                        st.success("Historical Analysis Complete")
                        st.markdown(response)
                    else:
                        st.error("Could not fetch historical weather data.")

    st.markdown("---")
    st.markdown(
        "**Disclaimer:** This tool provides AI-generated suggestions for informational purposes only."
    )

