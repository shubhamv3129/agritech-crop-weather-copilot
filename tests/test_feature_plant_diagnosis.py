"""
Unit tests for feature_plant_diagnosis.py.
Validates API key loading, EXIF GPS extraction, Google Maps reverse geocoding,
weather API lookups, image serialization, Gemini calls, and Streamlit UI flows
with mocked external API calls.
"""

import base64
import os
from unittest.mock import MagicMock, patch

import pytest

import feature_plant_diagnosis as fpd


def test_load_api_keys_from_env():
    """Verify API keys are loaded from environment variables when present."""
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "gemini-env-key",
            "GOOGLE_MAPS_API_KEY": "maps-env-key",
        },
        clear=True,
    ):
        gemini_key, maps_key = fpd.load_api_keys()
        assert gemini_key == "gemini-env-key"
        assert maps_key == "maps-env-key"


def test_load_api_keys_fallback_to_secrets():
    """Verify fallback to Streamlit secrets when environment variables are missing."""
    with patch.dict(os.environ, {}, clear=True), patch.object(
        fpd.st,
        "secrets",
        {"GEMINI_API_KEY": "gemini-secret", "GOOGLE_MAPS_API_KEY": "maps-secret"},
    ):
        gemini_key, maps_key = fpd.load_api_keys()
        assert gemini_key == "gemini-secret"
        assert maps_key == "maps-secret"


def test_load_api_keys_partial_env_and_secrets():
    """Verify environment variable takes precedence and missing key falls back to secrets."""
    with patch.dict(
        os.environ, {"GEMINI_API_KEY": "gemini-env-only"}, clear=True
    ), patch.object(
        fpd.st,
        "secrets",
        {"GEMINI_API_KEY": "gemini-secret", "GOOGLE_MAPS_API_KEY": "maps-secret"},
    ):
        gemini_key, maps_key = fpd.load_api_keys()
        assert gemini_key == "gemini-env-only"
        assert maps_key == "maps-secret"


def test_load_api_keys_secrets_exception():
    """Verify graceful handling when accessing st.secrets raises an exception."""
    broken_secrets = MagicMock()
    broken_secrets.get.side_effect = RuntimeError("No secrets file")
    with patch.dict(os.environ, {}, clear=True), patch.object(
        fpd.st, "secrets", broken_secrets
    ):
        gemini_key, maps_key = fpd.load_api_keys()
        assert gemini_key is None
        assert maps_key is None


@pytest.mark.parametrize(
    "lat_ref,lat_dms,lon_ref,lon_dms,expected_lat,expected_lon",
    [
        ("N", (28, 30, 0), "E", (77, 15, 36), 28.5, 77.26),
        ("S", (33, 51, 54), "W", (151, 12, 36), -33.865, -151.21),
    ],
)
def test_get_image_geotag_valid_exif(
    lat_ref, lat_dms, lon_ref, lon_dms, expected_lat, expected_lon
):
    """Verify DMS EXIF GPS extraction for N/E and S/W hemispheres."""
    mock_image = MagicMock()
    mock_image._getexif.return_value = {
        34853: {
            1: lat_ref,
            2: lat_dms,
            3: lon_ref,
            4: lon_dms,
        }
    }
    with patch.object(fpd.ExifTags, "TAGS", {34853: "GPSInfo"}):
        coords = fpd.get_image_geotag(mock_image)

    assert coords is not None
    assert coords["lat"] == pytest.approx(expected_lat, rel=1e-5)
    assert coords["lon"] == pytest.approx(expected_lon, rel=1e-5)


def test_get_image_geotag_no_exif():
    """Verify None is returned when the image has no EXIF metadata."""
    mock_image = MagicMock()
    mock_image._getexif.return_value = None
    assert fpd.get_image_geotag(mock_image) is None


def test_get_image_geotag_no_gps_info():
    """Verify None is returned when EXIF metadata exists without GPSInfo."""
    mock_image = MagicMock()
    mock_image._getexif.return_value = {271: "Canon"}
    with patch.object(fpd.ExifTags, "TAGS", {271: "Make", 34853: "GPSInfo"}):
        assert fpd.get_image_geotag(mock_image) is None


def test_get_image_geotag_malformed_gps():
    """Verify None is returned on malformed GPSInfo tuples."""
    mock_image = MagicMock()
    mock_image._getexif.return_value = {34853: {1: "N"}}
    with patch.object(fpd.ExifTags, "TAGS", {34853: "GPSInfo"}):
        assert fpd.get_image_geotag(mock_image) is None


def test_get_location_from_coords_success():
    """Verify reverse geocoding extracts formatted_address from Google Maps response."""
    mock_gmaps = MagicMock()
    mock_gmaps.reverse_geocode.return_value = [
        {"formatted_address": "Gurugram, Haryana, India"}
    ]
    result = fpd.get_location_from_coords(mock_gmaps, 28.4595, 77.0266)
    mock_gmaps.reverse_geocode.assert_called_once_with((28.4595, 77.0266))
    assert result == {"address": "Gurugram, Haryana, India"}


def test_get_location_from_coords_empty_result():
    """Verify error message when reverse geocoding returns no matches."""
    mock_gmaps = MagicMock()
    mock_gmaps.reverse_geocode.return_value = []
    result = fpd.get_location_from_coords(mock_gmaps, 0.0, 0.0)
    assert result == {"error": "Could not determine address."}


def test_get_location_from_coords_api_exception():
    """Verify Google Maps exceptions are caught and returned as error dicts."""
    mock_gmaps = MagicMock()
    mock_gmaps.reverse_geocode.side_effect = RuntimeError("OVER_QUERY_LIMIT")
    result = fpd.get_location_from_coords(mock_gmaps, 28.45, 77.02)
    assert "error" in result
    assert "OVER_QUERY_LIMIT" in result["error"]


def test_get_weather_success():
    """Verify current weather parsing from mocked Google Maps Weather API response."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "temperature": {"degrees": 29.4},
        "relativeHumidity": 62,
        "weatherCondition": {"description": {"text": "partly cloudy"}},
    }
    with patch.object(fpd.requests, "get", return_value=mock_response) as mock_get:
        result = fpd.get_weather(28.45, 77.02, "test-maps-key")

    mock_get.assert_called_once_with(
        "https://weather.googleapis.com/v1/currentConditions:lookup"
        "?key=test-maps-key&location.latitude=28.45&location.longitude=77.02",
        timeout=15,
    )
    mock_response.raise_for_status.assert_called_once()
    assert result == {
        "temperature": 29.4,
        "humidity": 62,
        "description": "Partly cloudy",
    }


def test_get_weather_http_error_with_json_details():
    """Verify HTTPError extracts status code and API error message."""
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.raise_for_status.side_effect = fpd.requests.exceptions.HTTPError(
        "Forbidden"
    )
    mock_response.json.return_value = {"error": {"message": "API key invalid"}}

    with patch.object(fpd.requests, "get", return_value=mock_response):
        result = fpd.get_weather(28.45, 77.02, "bad-key")

    assert result == {"error": "(403) API key invalid"}


def test_get_weather_http_error_without_json():
    """Verify HTTPError fallback message when response body is not valid JSON."""
    mock_response = MagicMock()
    mock_response.status_code = 502
    mock_response.raise_for_status.side_effect = fpd.requests.exceptions.HTTPError(
        "Bad Gateway"
    )
    mock_response.json.side_effect = ValueError("No JSON")

    with patch.object(fpd.requests, "get", return_value=mock_response):
        result = fpd.get_weather(28.45, 77.02, "test-key")

    assert result == {"error": "(502) No specific error details found."}


def test_get_weather_network_exception():
    """Verify general request exceptions return descriptive error dict."""
    with patch.object(
        fpd.requests, "get", side_effect=ConnectionError("Network unreachable")
    ):
        result = fpd.get_weather(28.45, 77.02, "test-key")

    assert "error" in result
    assert "Network unreachable" in result["error"]


def test_get_historical_weather_success():
    """Verify historical weather lookup returns parsed history records."""
    mock_history = [{"date": "2026-11-15", "maxTemp": 27.0}]
    mock_response = MagicMock()
    mock_response.json.return_value = {"history": mock_history}

    with patch.object(fpd.requests, "get", return_value=mock_response) as mock_get:
        result = fpd.get_historical_weather(28.45, 77.02, "test-maps-key")

    called_url = mock_get.call_args[0][0]
    assert "https://weather.googleapis.com/v1/history:lookup" in called_url
    assert "key=test-maps-key" in called_url
    assert "location.latitude=28.45&location.longitude=77.02" in called_url
    assert mock_get.call_args[1]["timeout"] == 15
    assert result == mock_history


def test_get_historical_weather_exception_returns_none():
    """Verify historical weather lookup returns None when request fails."""
    with patch.object(fpd.requests, "get", side_effect=TimeoutError("Timed out")):
        result = fpd.get_historical_weather(28.45, 77.02, "test-maps-key")
    assert result is None


def test_get_gemini_response_with_image():
    """Verify multimodal Gemini invocation includes both prompt and image payload."""
    mock_model = MagicMock()
    mock_model.generate_content.return_value = MagicMock(text="Possible nitrogen deficiency.")
    image_payload = {"mime_type": "image/jpeg", "data": "abc123"}

    with patch.object(
        fpd.genai, "GenerativeModel", return_value=mock_model
    ) as mock_model_cls:
        result = fpd.get_gemini_response(image_payload, "Diagnose this leaf")

    mock_model_cls.assert_called_once_with("gemini-2.5-pro")
    mock_model.generate_content.assert_called_once_with(
        ["Diagnose this leaf", image_payload]
    )
    assert result == "Possible nitrogen deficiency."


def test_get_gemini_response_text_only():
    """Verify text-only Gemini invocation passes prompt string directly."""
    mock_model = MagicMock()
    mock_model.generate_content.return_value = MagicMock(text="7-day irrigation schedule")

    with patch.object(
        fpd.genai, "GenerativeModel", return_value=mock_model
    ) as mock_model_cls:
        result = fpd.get_gemini_response(None, "Create irrigation plan")

    mock_model_cls.assert_called_once_with("gemini-2.5-pro")
    mock_model.generate_content.assert_called_once_with("Create irrigation plan")
    assert result == "7-day irrigation schedule"


def test_image_to_dict_serializes_jpeg_base64():
    """Verify image_to_dict converts image to RGB JPEG and base64-encodes bytes."""
    mock_rgb_image = MagicMock()
    mock_rgb_image.save.side_effect = lambda buf, format: buf.write(b"fake-jpeg-bytes")
    mock_image = MagicMock()
    mock_image.convert.return_value = mock_rgb_image

    payload = fpd.image_to_dict(mock_image)

    mock_image.convert.assert_called_once_with("RGB")
    mock_rgb_image.save.assert_called_once()
    assert mock_rgb_image.save.call_args.kwargs["format"] == "JPEG"
    assert payload == {
        "mime_type": "image/jpeg",
        "data": base64.b64encode(b"fake-jpeg-bytes").decode("utf-8"),
    }


def test_render_farm_assistant_app_missing_api_keys():
    """Verify app shows error and calls st.stop when API keys are missing."""
    with patch.object(fpd, "load_api_keys", return_value=(None, None)), patch.object(
        fpd.st, "stop", side_effect=SystemExit(0)
    ) as mock_stop:
        with pytest.raises(SystemExit):
            fpd.render_farm_assistant_app()

        fpd.st.error.assert_called_once()
        mock_stop.assert_called_once()


def test_render_farm_assistant_app_initializes_clients():
    """Verify app configures Gemini and Google Maps clients when API keys are present."""
    with patch.object(
        fpd, "load_api_keys", return_value=("gemini-key", "maps-key")
    ), patch.object(fpd.genai, "configure") as mock_configure, patch.object(
        fpd.googlemaps, "Client"
    ) as mock_maps_cls:
        fpd.render_farm_assistant_app()

    mock_configure.assert_called_once_with(api_key="gemini-key")
    mock_maps_cls.assert_called_once_with(key="maps-key")
    fpd.st.set_page_config.assert_called_once()
    fpd.st.title.assert_called_once()
