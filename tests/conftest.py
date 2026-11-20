"""
Pytest configuration and environment fixtures for agritech-crop-weather-copilot.
Ensures external dependencies are mocked gracefully if not installed in the test environment.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

# Ensure repository root is importable regardless of pytest invocation mode
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class _SessionState(dict):
    """Dict-like session state supporting Streamlit attribute-style access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _context_manager_mock():
    """Create a MagicMock usable in Python `with` statements."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=None)
    return cm


# Ensure missing third-party modules can be imported during tests
MOCK_MODULES = [
    "dotenv",
    "google",
    "google.generativeai",
    "googlemaps",
    "PIL",
    "PIL.ExifTags",
    "PIL.Image",
    "requests",
    "streamlit",
]

for mod in MOCK_MODULES:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            mock_obj = MagicMock()
            sys.modules[mod] = mock_obj
            if mod == "google.generativeai" and "google" in sys.modules:
                sys.modules["google"].generativeai = mock_obj
            elif mod == "PIL.ExifTags" and "PIL" in sys.modules:
                sys.modules["PIL"].ExifTags = mock_obj
            elif mod == "PIL.Image" and "PIL" in sys.modules:
                sys.modules["PIL"].Image = mock_obj
            elif mod == "requests":
                mock_obj.exceptions = MagicMock()
                mock_obj.exceptions.HTTPError = type("HTTPError", (Exception,), {})
                sys.modules["requests.exceptions"] = mock_obj.exceptions


@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock Streamlit UI calls so tests run cleanly in headless mode."""
    import streamlit as st

    st.set_page_config = MagicMock()
    st.title = MagicMock()
    st.header = MagicMock()
    st.subheader = MagicMock()
    st.markdown = MagicMock()
    st.write = MagicMock()
    st.info = MagicMock()
    st.success = MagicMock()
    st.warning = MagicMock()
    st.error = MagicMock()
    st.metric = MagicMock()
    st.image = MagicMock()
    st.button = MagicMock(return_value=False)
    st.text_input = MagicMock(return_value="")
    st.text_area = MagicMock(return_value="")
    st.file_uploader = MagicMock(return_value=None)
    st.spinner = MagicMock(side_effect=lambda *args, **kwargs: _context_manager_mock())
    st.sidebar = _context_manager_mock()
    st.expander = MagicMock(side_effect=lambda *args, **kwargs: _context_manager_mock())
    st.columns = MagicMock(
        side_effect=lambda spec, **kwargs: [
            _context_manager_mock()
            for _ in range(spec if isinstance(spec, int) else len(spec))
        ]
    )
    st.tabs = MagicMock(
        side_effect=lambda labels: [_context_manager_mock() for _ in labels]
    )
    st.stop = MagicMock(side_effect=SystemExit(0))
    st.session_state = _SessionState()
    st.secrets = {}
    yield st
