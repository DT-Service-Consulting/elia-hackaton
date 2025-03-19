from pathlib import Path

# Get the base directory (directory where this file resides)
BASE_DIR = Path(__file__).parent

# Paths to various resources
IMAGES_DIR = BASE_DIR / 'images'
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
SAVED_MODELS_DIR = BASE_DIR / 'data' / 'saved_models'

# ELIA PROVIDED INFORMATION

# Define the API endpoint and your API key
api_url = "https://api-nprd.traxes.io/hackathon/"
api_key = "a47c28cc-9401-450f-b052-db23dccb26c5"

# Set up the headers with your API key
headers = {
    "x-api-key": api_key,
    "Content-Type": "application/json"
}

start_date_x = '2023-05-01'
end_date_x = '2025-02-28T22:00:00'

start_date_y = '2023-05-01'
end_date_y = '2024-12-30T23:45:00'
