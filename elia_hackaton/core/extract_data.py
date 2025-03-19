import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore")

# Define the API endpoint and your API key
api_url = "https://api-nprd.traxes.io/hackathon/" 
api_key = "a47c28cc-9401-450f-b052-db23dccb26c5"

# Set up the headers with your API key
headers = {
    "x-api-key": api_key,
    "Content-Type": "application/json"
}
def get_data(api_url, data_extension, headers):
    # Make the GET request to the API
    response = requests.get(api_url + data_extension, headers=headers)
    # Check the response status code
    if response.status_code == 200:
        # If the request was successful, parse the JSON response
        data = response.json()
        print(data)
        return data
    else:
        # If the request failed
        print(response)
        return None
data_requested = 'equipment/GetAllTransformers/'

tfo_data = get_data(api_url, data_requested, headers)

if tfo_data != None:
    df_tfo = pd.DataFrame(tfo_data)
else:
    print('Error')

print("Success")

df_tfo_extra = pd.json_normalize(df_tfo['heatRunTest'])

df_tfo = pd.concat([df_tfo.drop(columns=['heatRunTest']), df_tfo_extra], axis=1)
df_tfo

df_tfo.to_csv('tfo_parameters.csv')

def merge_dataframes(equipment_id, load,hotspot_t,outside_t):
    if (load.empty) or (hotspot_t.empty) or (outside_t.empty):
        print('Merge is not possible')

    else:
        load = load.drop(columns = 'equipmentId')
        hotspot_t = hotspot_t.drop(columns = 'equipmentId')
        outside_t = outside_t.drop(columns = 'locationId')
        merged_df = pd.merge(load, outside_t, on = 'dateTime', how = 'left')
        merged_df = merged_df.fillna(method='ffill')
        merged_df = pd.merge(merged_df, hotspot_t, on=['dateTime'], how = 'left')
        merged_df.to_csv('all_time_series/' + str(equipment_id) + '.csv')



start_date_x = '2023-05-01'
end_date_x = '2025-02-28T22:00:00'

start_date_y = '2023-05-01'
end_date_y = '2024-12-30T23:45:00'

for _, row in df_tfo.iterrows():
    print('\n')

    equipment_id = row['equipmentId']
    location_id = row['locationId']
    
    # Get Load

    data_requested = f'equipment/GetTransformerLoad?equipmentId={equipment_id}&fromDate={start_date_x}&toDate={end_date_x}'
    
    load_data = get_data(api_url, data_requested, headers)

    if load_data != None:
        print(str(equipment_id) + ': Success Load!')
        df_load = pd.DataFrame(load_data)
        df_load.to_csv('load/' + str(equipment_id) + '.csv')
    else:
        print(str(equipment_id) + ': ERROR Load!')

    # Get Outside Temperature

    data_requested = f'weather/GetOutsideTemperature?locationId={location_id}&fromDate={start_date_x}&toDate={end_date_x}'
    
    temperature_outside_data = get_data(api_url, data_requested, headers)

    if temperature_outside_data != None:
        print(str(equipment_id) + ': Success Outside Temperature!')
        df_outside_temperature = pd.DataFrame(temperature_outside_data)
        df_outside_temperature.to_csv('outside_temperature/' + str(equipment_id) + '.csv')
    else:
        print(str(equipment_id) + ': ERROR Outside Temperature!')

    # Get Hotspot Temperature
    data_requested = f"equipment/GetTransformerTemperature?equipmentId={equipment_id}&fromDate={start_date_y}&toDate={end_date_y}"

    hotspot_temperature_data = get_data(api_url, data_requested, headers)

    if hotspot_temperature_data != None:
        print(str(equipment_id) + ': Success Hotspot Temperature!')
        df_hotspot_temperature = pd.DataFrame(hotspot_temperature_data)
        df_hotspot_temperature.to_csv('hotspot_temperature/' + str(equipment_id) + '.csv')
    else:
        print(str(equipment_id) + ': ERROR Hotspot Temperature!')
        


    merge_dataframes(equipment_id, df_load, df_hotspot_temperature, df_outside_temperature)


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


# Function to save model with all necessary components for prediction
def save_model_for_prediction(model, scaler, equipment_name, parameters):
    # Create models directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)

    # Save model state dictionary
    model_path = os.path.join('saved_models', f'model_{equipment_name}.pth')
    torch.save(model.state_dict(), model_path)

    # Save model architecture configuration
    config_path = os.path.join('saved_models', f'config_{equipment_name}.json')
    with open(config_path, 'w') as f:
        json.dump(model.get_config(), f)

    # Save scaler
    scaler_path = os.path.join('saved_models', f'scaler_{equipment_name}.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Save physics parameters
    params_path = os.path.join('saved_models', f'params_{equipment_name}.json')
    with open(params_path, 'w') as f:
        json.dump(parameters, f)

    print(f"Model and associated components saved for {equipment_name}")


# Function to load model and make predictions
def load_model_and_predict(equipment_name, input_data):
    # Load model configuration
    config_path = os.path.join('saved_models', f'config_{equipment_name}.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Initialize model with saved configuration
    model = ImprovedPINN(**config).to(device)

    # Load model weights
    model_path = os.path.join('saved_models', f'model_{equipment_name}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load scaler
    scaler_path = os.path.join('saved_models', f'scaler_{equipment_name}.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Scale input data
    X_scaled = scaler.transform(input_data)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    # Make prediction
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()

    return predictions
