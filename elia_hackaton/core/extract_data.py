import os.path

import pandas as pd
import requests
import warnings
import pickle
import json

warnings.filterwarnings("ignore")
from elia_hackaton.models.models import Prometheus
from elia_hackaton.config import api_url, headers
from elia_hackaton.config import DATA_DIR
from elia_hackaton.config import start_date_x, end_date_x, start_date_y, end_date_y
from elia_hackaton.core.utils import setup_gpu

device = setup_gpu()


def get_data_from_api(api_url, data_extension, headers):
    """
    Fetch data from the specified API endpoint.

    This function makes a GET request to the given API URL with the provided data extension and headers.
    It checks the response status and returns the parsed JSON data if the request is successful.

    Parameters:
    api_url (str): The base URL of the API.
    data_extension (str): The specific endpoint extension to append to the base URL.
    headers (dict): A dictionary of HTTP headers to include in the request.

    Returns:
    dict or None: The parsed JSON data from the API response if successful, otherwise None.
    """
    # Make the GET request to the API
    response = requests.get(api_url + data_extension, headers=headers)

    # Check the response status code
    if response.status_code == 200:
        # If the request was successful, parse the JSON response
        data = response.json()
        return data
    else:
        # If the request failed
        return None


def merge_dataframes(equipment_id, load, hotspot_t, outside_t):
    """
    Merge load, hotspot temperature, and outside temperature dataframes for a given equipment.

    This function merges the provided dataframes on the 'dateTime' column, fills missing values
    using forward fill, and saves the merged dataframe to a CSV file.

    Parameters:
    equipment_id (int): The identifier of the equipment.
    load (pd.DataFrame): DataFrame containing the load data.
    hotspot_t (pd.DataFrame): DataFrame containing the hotspot temperature data.
    outside_t (pd.DataFrame): DataFrame containing the outside temperature data.

    Returns:
    pd.DataFrame: The merged dataframe containing all the data.
    """
    if load.empty or hotspot_t.empty or outside_t.empty:
        print('Merge is not possible due to empty dataframes.')
        return None

    # Drop unnecessary columns
    load = load.drop(columns='equipmentId', errors='ignore')
    hotspot_t = hotspot_t.drop(columns='equipmentId', errors='ignore')
    outside_t = outside_t.drop(columns='locationId', errors='ignore')

    # Merge dataframes on 'dateTime' column
    merged_df = pd.merge(load, outside_t, on='dateTime', how='left')
    merged_df = merged_df.fillna(method='ffill')
    merged_df = pd.merge(merged_df, hotspot_t, on='dateTime', how='left')

    # Save the merged dataframe to a CSV file
    output_path = DATA_DIR / 'all_time_series' / f'{equipment_id}.csv'
    merged_df.to_csv(output_path, index=False)
    print(f'Merged dataframe saved to {output_path}')

    return merged_df


def get_load(equipment_id, end_date_x):
    """
    Fetch and save the load data for a given equipment.

    This function requests the load data from the API for the specified equipment and date range.
    It saves the data to a CSV file if the request is successful.

    Parameters:
    equipment_id (int): The identifier of the equipment.
    end_date_x (str): The end date for the data request in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: The DataFrame containing the load data if successful, otherwise None.
    """
    data_requested = f'equipment/GetTransformerLoad?equipmentId={equipment_id}&fromDate={start_date_x}&toDate={end_date_x}'
    load_data = get_data_from_api(api_url, data_requested, headers)

    if load_data is None:
        print(f'{equipment_id}: ERROR Load!')
        df_load = None
    else:
        print(f'{equipment_id}: Success Load!')
        df_load = pd.DataFrame(load_data)
        df_load.to_csv(DATA_DIR / 'load' / f'{equipment_id}.csv', index=False)

    return df_load


def get_outside_temperature(location_id, end_date_x, equipment_id):
    # Get Outside Temperature

    global df_outside_temperature
    data_requested = f'weather/GetOutsideTemperature?locationId={location_id}&fromDate={start_date_x}&toDate={end_date_x}'

    temperature_outside_data = get_data_from_api(api_url, data_requested, headers)

    if temperature_outside_data is not None:
        print(str(equipment_id) + ': Success Outside Temperature!')
        df_outside_temperature = pd.DataFrame(temperature_outside_data)
        df_outside_temperature.to_csv(DATA_DIR / 'outside_temperature' / f'{str(equipment_id)}.csv')
    else:
        print(str(equipment_id) + ': ERROR Outside Temperature!')

    return df_outside_temperature


def get_hotspot_temperature(equipment_id, end_date_y):
    # Get Hotspot Temperature
    global df_hotspot_temperature
    data_requested = f"equipment/GetTransformerTemperature?equipmentId={equipment_id}&fromDate={start_date_y}&toDate={end_date_y}"

    hotspot_temperature_data = get_data_from_api(api_url, data_requested, headers)

    if hotspot_temperature_data is not None:
        print(str(equipment_id) + ': Success Hotspot Temperature!')
        df_hotspot_temperature = pd.DataFrame(hotspot_temperature_data)
        df_hotspot_temperature.to_csv(DATA_DIR / 'hotspot_temperature' / f'{str(equipment_id)}.csv')
    else:
        print(str(equipment_id) + ': ERROR Hotspot Temperature!')

    return df_hotspot_temperature


def get_data_locally(df_tfo):
    for _, row in df_tfo.iterrows():
        equipment_csv = DATA_DIR / 'all_time_series' / f'{str(row["equipmentId"])}.csv'
        if not os.path.exists(equipment_csv):
            equipment_id = row['equipmentId']
            location_id = row['locationId']

            df_load = get_load(equipment_id, end_date_x)
            df_outside_temperature = get_outside_temperature(location_id, end_date_x, equipment_id)
            df_hotspot_temperature = get_hotspot_temperature(equipment_id, end_date_y)

            merged_df = merge_dataframes(equipment_id, df_load, df_hotspot_temperature, df_outside_temperature)
            merged_df.to_csv(DATA_DIR / 'all_time_series' / f'{str(equipment_id)}.csv')
    return


def post_data(json_data, api_url="https://your-api-endpoint.com/data"):
    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer YOUR_API_KEY'  # Replace with your actual API key
        }

        response = requests.post(api_url, data=json_data, headers=headers)

        if response.status_code == 200 or response.status_code == 201:
            print("Data successfully uploaded to API")
            return True
        else:
            print(f"Error uploading data: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"Exception occurred during API upload: {e}")

        return False


def split_dataframe(df, chunk_size):
    chunks = []
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunks.append(df[start_idx:end_idx].copy())

    return chunks


import torch
from elia_hackaton.config import SAVED_MODELS_DIR


# Function to save model with all necessary components for prediction
def save_model_for_prediction(model, scaler, equipment_name, parameters):
    # Create models directory if it doesn't exist

    # Save model state dictionary
    model_path = SAVED_MODELS_DIR / f'model_{equipment_name}.pth'
    torch.save(model.state_dict(), model_path)

    # Save model architecture configuration
    config_path = SAVED_MODELS_DIR / f'config_{equipment_name}.json'
    with open(config_path, 'w') as f:
        json.dump(model.get_config(), f)

    # Save scaler
    scaler_path = SAVED_MODELS_DIR / f'scaler_{equipment_name}.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Save physics parameters
    params_path = SAVED_MODELS_DIR / f'params_{equipment_name}.json'
    with open(params_path, 'w') as f:
        json.dump(parameters, f)

    print(f"Model and associated components saved for {equipment_name}")


# Function to load model and make predictions
def load_model_and_predict(equipment_name, input_data):
    # Load model configuration
    config_path = SAVED_MODELS_DIR / f'config_{equipment_name}.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Initialize model with saved configuration
    model = Prometheus(**config).to(device)

    # Load model weights
    model_path = SAVED_MODELS_DIR / f'model_{equipment_name}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load scaler
    scaler_path = SAVED_MODELS_DIR / f'scaler_{equipment_name}.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Scale input data
    X_scaled = scaler.transform(input_data)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    # Make prediction
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()

    return predictions
