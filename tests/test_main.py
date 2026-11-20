"""
Unit tests for main.py application entry points.
Validates entry point wiring and delegation to feature_plant_diagnosis.
"""

from unittest.mock import patch

import main


def test_example_plant_diagnoser_invokes_render():
    """Verify example_plant_diagnoser triggers render_farm_assistant_app."""
    with patch.object(main, "render_farm_assistant_app") as mock_render:
        main.example_plant_diagnoser()
        mock_render.assert_called_once()


def test_main_invokes_example_plant_diagnoser():
    """Verify main invokes example_plant_diagnoser."""
    with patch.object(main, "example_plant_diagnoser") as mock_example:
        main.main()
        mock_example.assert_called_once()
