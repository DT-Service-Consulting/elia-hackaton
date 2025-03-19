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


   