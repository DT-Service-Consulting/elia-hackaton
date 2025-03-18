# Compute Transformer Temperature & Risk Level
def predict_temperature(data):
    load_factor, ambient_temp, oil_temp, humidity, wind_speed = data
    R, x, y = 0.8, 1.2, 0.9  # Transformer Constants
    base_temp = ((1 + R * load_factor**2) / (1 + R))**x * \
        (oil_temp - ambient_temp) + load_factor**y * (110 - oil_temp)

    X_input = torch.tensor([data], dtype=torch.float32).to(device)
    correction = model(X_input).item()
    final_temp = base_temp + correction
    return final_temp


# Determine Risk Level & Strategy
def evaluate_substation(substation):
    data, max_load = fetch_data(substation)
    if data is None:
        return None

    predicted_temp = predict_temperature(data)

    # Define Safety Thresholds
    status = "Safe"
    if predicted_temp >= 90:
        status = "Warning"
    if predicted_temp >= 105 or data[0] * max_load > max_load:
        status = "Critical"

    return {
        "Substation": substation,
        "Predicted Temperature": round(predicted_temp, 2),
        "Max Load": max_load,
        "Current Load": data[0] * max_load,
        "Status": status
    }
