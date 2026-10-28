# 🧑‍🌾 AgriTech Crop & Weather Copilot (`agritech-crop-weather-copilot`)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Pro-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)
![Google Maps](https://img.shields.io/badge/Google_Maps-Geocoding_%26_Weather_API-34A853?style=for-the-badge&logo=google-maps&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

**AgriTech Crop & Weather Copilot** is a multimodal precision-agriculture decision-support system built with **Streamlit**, **Google Gemini 2.5 Pro**, **Google Maps Geocoding API**, and the **Google Maps Platform Weather API**. It fuses computer vision on crop/soil/pest photography with real-time and 7-day historical hyper-local weather telemetry extracted from image EXIF GPS tags.

---

## 📐 System Architecture

```mermaid
flowchart TD
    Upload["Farmer Uploads Crop / Soil / Pest Photo"] --> EXIF["EXIF GPS Geotag Extractor\n(get_image_geotag)"]
    EXIF -->|Coords Found| ReverseGeo["Google Maps Reverse Geocoding"]
    EXIF -->|No EXIF| ManualGeo["Manual City / Address Geocoding"]

    ReverseGeo --> Weather["Google Maps Weather API\nCurrent Conditions & 7-Day History"]
    ManualGeo --> Weather

    Upload --> Vision["Gemini 2.5 Pro Multimodal Engine"]
    Weather --> Vision

    Vision --> T1["🩺 Tab 1: Plant Disease & Toxicity Diagnosis"]
    Vision --> T2["🌱 Tab 2: Soil Composition & Crop Suitability"]
    Vision --> T3["🐞 Tab 3: Pest & Invasive Weed Control"]
    Vision --> T4["💧 Tab 4: 7-Day Crop Irrigation Schedule"]
    Vision --> T5["📈 Tab 5: Historical Agronomic Climate Analysis"]
```

---

## 📁 Project Structure

```text
agritech-crop-weather-copilot/
├── main.py                      # Primary application entry point
├── app.py                       # Streamlit entry point alias
├── feature_plant_diagnosis.py   # Multimodal Gemini, EXIF GPS, and Weather API implementation
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── .streamlit/
│   └── secrets.toml.example     # Streamlit secrets template
└── README.md                    # Project documentation
```

---

## ⚙️ Configuration & Secrets

You can provide your API keys via either `.env` or `.streamlit/secrets.toml`:

| Key | Required | Description |
| :--- | :---: | :--- |
| `GEMINI_API_KEY` | Yes | Google Gemini API key for multimodal diagnosis (`gemini-2.5-pro`) |
| `GOOGLE_MAPS_API_KEY` | Yes | Google Maps Platform key with Geocoding API and Weather API enabled |

---

## 🚀 Quickstart

```bash
git clone https://github.com/shubhamv3129/agritech-crop-weather-copilot.git
   cd agritech-crop-weather-copilot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Or: cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Add your GEMINI_API_KEY and GOOGLE_MAPS_API_KEY

streamlit run main.py
```
