"""
Main entry point for the Plant Diagnoser (Farm AI Assistant) application.
"""
from feature_plant_diagnosis import render_farm_assistant_app


def example_plant_diagnoser() -> None:
    """
    Example function launching the Farm AI Assistant Streamlit interface.
    """
    render_farm_assistant_app()


def main() -> None:
    """
    Main entry point.
    """
    example_plant_diagnoser()


if __name__ == "__main__":
    main()

