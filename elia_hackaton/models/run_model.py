import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from itertools import product
from elia_hackaton.models.classes import PINN
from elia_hackaton.config import DATA_DIR
from elia_hackaton.core.utils import white_box_model


if torch.cuda.is_available():
    device = torch.device("cuda")  # Use NVIDIA GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Use Apple Metal (macOS)
else:
    device = torch.device("cpu")  # Default to CPU

print(f"Using device: {device}")

df = pd.read_csv( DATA_DIR / 'final_merged_data.csv')
df = df.sample(1000)

# Selecting relevant features, including weather data
features = ["nominalLoad", "heatRunTest_noLoadLosses", "heatRunTest_copperLosses",
            "heatRunTest_ambiantTemperature", "heatRunTest_deltaTopOil",
            "heatRunTest_x", "heatRunTest_y", "heatRunTest_h", "heatRunTest_gradient",
            "load", "heatRunTest_ambiantTemperature"]  # Including ambient temperature as weather data
target = "hotspotTemperature"


# Normalize data
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])
X = df[features].values
y = df[target].values.reshape(-1, 1)
y_scaler = MinMaxScaler()
y = y_scaler.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# Grid search for hyperparameter tuning
hidden_layer_options = [2, 3]
neuron_options = [16, 32]
learning_rate_options = [0.001, 0.01]
best_rmse = float("inf")
best_params = None

for hidden_layers, neurons, lr in product(hidden_layer_options, neuron_options, learning_rate_options):
    model = PINN(input_size=len(features), hidden_layers=hidden_layers, neurons=neurons)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train model
    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate model
    y_pred = model(X_test_tensor).detach().numpy()
    y_pred = y_scaler.inverse_transform(y_pred)
    y_test_actual = y_scaler.inverse_transform(y_test)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))

    if rmse < best_rmse:
        best_rmse = rmse
        best_params = (hidden_layers, neurons, lr)

print(f"Best parameters: {best_params}, Best RMSE: {best_rmse}")

white_box_pred = white_box_model(X_test)
white_box_rmse = np.sqrt(mean_squared_error(y_test_actual, white_box_pred))

print(f"Best PINN parameters: {best_params}, PINN RMSE: {best_rmse}")
print(f"White-box model RMSE: {white_box_rmse}")