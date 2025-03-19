from pathlib import Path

# Get the base directory (directory where this file resides)
BASE_DIR = Path(__file__).parent

# Paths to various resources
IMAGES_DIR = BASE_DIR / 'images'
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'

# Define the API endpoint and your API key
api_url = "https://api-nprd.traxes.io/hackathon/"
api_key = "a47c28cc-9401-450f-b052-db23dccb26c5"

# Set up the headers with your API key
headers = {
    "x-api-key": api_key,
    "Content-Type": "application/json"
}