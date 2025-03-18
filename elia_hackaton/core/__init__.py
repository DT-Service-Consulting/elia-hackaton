# Fetch Real-Time Data
def fetch_data(substation):
    try:
        transformer_response = requests.get(
            f"{TRANSFORMER_API_URL}?substation={substation}")
        weather_response = requests.get(WEATHER_API_URL)

        if transformer_response.status_code == 200 and weather_response.status_code == 200:
            transformer_data = transformer_response.json()
            weather_data = weather_response.json()

            load_factor = transformer_data["load_factor"]
            max_load = transformer_data["max_load"]
            ambient_temp = weather_data["ambient_temp"]
            oil_temp = transformer_data["oil_temp"]
            humidity = weather_data["humidity"]
            wind_speed = weather_data["wind_speed"]

            return np.array([load_factor, ambient_temp, oil_temp, humidity, wind_speed]), max_load
        else:
            print(f"Error fetching data for {substation}")
            return None, None
    except Exception as e:
        print(f"API Request Failed for {substation}: {e}")
        return None, None
