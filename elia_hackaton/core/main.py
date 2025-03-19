from elia_hackaton.core.utils import setup_gpu
from elia_hackaton.config import DATA_DIR
from elia_hackaton.core.extract_data import get_data
from elia_hackaton.config import api_url, headers

import pandas as pd

device = setup_gpu()

# Load transformer parameters
tfo_parameters_df = pd.read_csv( DATA_DIR / 'tfo_parameters.csv', index_col=0)
#equipment = pd.read_csv(DATA_DIR / 'Equipment.csv', index_col=0)
print(tfo_parameters_df)

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




import time
import os

# Directory for storing RMSE and accuracy results
results_summary = {}
scaler_dict = {}  # Store scalers for each equipment


start_time = time.time()
print("Starting model training and evaluation...")

# Check or create directories
os.makedirs('test', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

# First pass: Train models and save them
for filename in os.listdir('results'):
    if filename.endswith(".csv"):
        file_path = os.path.join('results', filename)
        equipment_df = pd.read_csv(file_path)
        equipment_name = filename.split('.csv')[0]

        print(f"\nProcessing {equipment_name}...")

        # Get equipment parameters
        TFO = tfo_parameters_df[tfo_parameters_df['equipmentId'] == int(equipment_name)]
        if TFO.empty:
            print(f"No parameters found for {equipment_name}, skipping...")
            continue

        dt_o = float(TFO['deltaTopOil'].iloc[0])
        dt_h = float(TFO['deltaHotspot'].iloc[0])
        x_param = float(TFO['x'].iloc[0])
        y_param = float(TFO['y'].iloc[0])
        R = float(TFO['copperLosses'].iloc[0])/float(TFO['noLoadLosses'].iloc[0])

        # Store physics parameters for later use
        physics_params = {
            'dt_o': dt_o,
            'dt_h': dt_h,
            'x_param': x_param,
            'y_param': y_param,
            'R': R
        }

        # Prepare data with more features
        X_data = pd.DataFrame({
            'K': equipment_df['K'],
            'ambient_temp': equipment_df['temperature'],
            'time_of_day': pd.to_datetime(equipment_df['dateTime']).dt.hour / 24.0,
            'day_of_week': pd.to_datetime(equipment_df['dateTime']).dt.dayofweek / 7.0,
            'load_derivative': equipment_df['K'].diff().fillna(0)
        })

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_data)
        scaler_dict[equipment_name] = scaler

        y_true = equipment_df['temperature'].values.reshape(-1, 1)
        y_whitebox = equipment_df['Temp Pred'].values.reshape(-1, 1)

        # Split data for training
        equipment_df['dateTime'] = pd.to_datetime(equipment_df['dateTime'])
        split_datetime = pd.Timestamp('2025-01-01T00:00:00')

        # Create train and test masks
        train_mask = equipment_df['dateTime'] < split_datetime

        # Ensure we have at least some training data
        if sum(train_mask) == 0:
            print(f"No training data before {split_datetime} for {equipment_name}, skipping...")
            continue

        # Get train and test indices
        train_indices = np.where(train_mask)[0]

        # Split train data into train and validation
        train_size = int(0.8 * len(train_indices))

        # Create final indices
        final_train_indices = train_indices[:train_size]
        final_val_indices = train_indices[train_size:]

        # Create tensors
        X_train = torch.tensor(X_scaled[final_train_indices], dtype=torch.float32).to(device)
        y_train = torch.tensor(y_true[final_train_indices], dtype=torch.float32).to(device)
        y_whitebox_train = torch.tensor(y_whitebox[final_train_indices], dtype=torch.float32).to(device)
        k_train = torch.tensor(equipment_df['K'].values[final_train_indices], dtype=torch.float32).view(-1, 1).to(device)

        X_val = torch.tensor(X_scaled[final_val_indices], dtype=torch.float32).to(device)
        y_val = torch.tensor(y_true[final_val_indices], dtype=torch.float32).to(device)

        # Create DataLoaders
        train_dataset = TensorDataset(X_train, y_train, y_whitebox_train, k_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Initialize model and move to GPU
        model = ImprovedPINN(input_dim=X_scaled.shape[1]).to(device)

        # Initialize optimizer with learning rate scheduler
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, verbose=True
        )

        # Loss functions
        mse_loss = nn.MSELoss()

        # Training loop
        epochs = 2000
        best_val_loss = float('inf')
        patience = 50
        counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0

            for batch_X, batch_y, batch_whitebox, batch_k in train_loader:
                optimizer.zero_grad()

                # Forward pass
                outputs = model(batch_X)

                # Calculate losses
                data_loss = mse_loss(outputs, batch_y)
                physics_constraint = physics_loss(
                    model, batch_X, batch_k, dt_o, dt_h,
                    x_param, y_param, R, batch_X[:, 1]  # ambient_temp is at index 1
                )

                # Combine losses - balance between data and physics
                alpha = 0.7  # Weight for data loss
                beta = 0.3   # Weight for physics loss
                loss = alpha * data_loss + beta * physics_constraint

                # Backward pass and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = mse_loss(val_outputs, y_val)
                val_losses.append(val_loss.item())

                # Print progress every 100 epochs
                if (epoch + 1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}')

                # Learning rate scheduling
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    # Save best model
                    torch.save(model.state_dict(), f'best_model_{equipment_name}.pth')
                else:
                    counter += 1
                    if counter >= patience:
                        print(f'Early stopping at epoch {epoch+1}')
                        break

        # Load best model for evaluation
        model.load_state_dict(torch.load(f'best_model_{equipment_name}.pth'))
        model.eval()

        # Save model with all necessary components for prediction
        save_model_for_prediction(model, scaler, equipment_name, physics_params)

        # Plot training curves
        plot_training_curves(train_losses, val_losses, equipment_name)

# Second pass: Generate predictions and prepare for API upload
print("\n\nGenerating predictions for API upload...")
for filename in os.listdir('results'):
    if filename.endswith(".csv"):
        file_path = os.path.join('results', filename)
        equipment_df = pd.read_csv(file_path)

        equipment_name = filename.split('.csv')[0]
        print('\n' + equipment_name + '\n')

        equipment_df['dateTime'] = pd.to_datetime(equipment_df['dateTime'])

        # Define the test data based on the datetime condition
        split_datetime = pd.Timestamp('2025-01-01T00:00:00')
        equipment_df_test = equipment_df[equipment_df['dateTime'] >= split_datetime]

        # Skip if no test data
        if len(equipment_df_test) == 0:
            print(f"No test data found for {equipment_name} after {split_datetime}")
            continue

        # Prepare features for prediction using the PINN model
        X_test_data = pd.DataFrame({
            'K': equipment_df_test['K'],
            'ambient_temp': equipment_df_test['temperature'],
            'time_of_day': pd.to_datetime(equipment_df_test['dateTime']).dt.hour / 24.0,
            'day_of_week': pd.to_datetime(equipment_df_test['dateTime']).dt.dayofweek / 7.0,
            'load_derivative': equipment_df_test['K'].diff().fillna(0)
        })

        # Load model and make predictions
        try:
            # Get PINN predictions for test data
            pinn_predictions = load_model_and_predict(equipment_name, X_test_data)

            # Create a copy of original dataframe before dateTime format changes
            plot_df = equipment_df.copy()

            # Add a filter for test period data
            test_mask = plot_df['dateTime'] >= split_datetime

            # Create plot comparing predictions
            plt.figure(figsize=(12, 6))

            # Plot actual temperatures for the entire dataset
            plt.plot(plot_df['dateTime'], plot_df['temperature'], label='Actual', color='black')

            # Plot white box predictions
            plt.plot(plot_df['dateTime'], plot_df['Temp Pred'], label='White Box', color='blue', linestyle='--')

            # Create full array of predictions (NaN for training period, PINN for test period)
            full_pinn_preds = np.full(len(plot_df), np.nan)
            full_pinn_preds[test_mask] = pinn_predictions.flatten()

            # Plot PINN predictions (only for test period)
            plt.plot(plot_df['dateTime'], full_pinn_preds, label='PINN', color='red')

            # Add vertical line at train/test split
            plt.axvline(x=split_datetime, color='green', linestyle='-', alpha=0.5, label='Train/Test Split')

            # Calculate metrics for test period only
            if sum(test_mask) > 0:
                test_actual = plot_df.loc[test_mask, 'temperature'].values
                test_whitebox = plot_df.loc[test_mask, 'Temp Pred'].values

                whitebox_rmse = np.sqrt(mean_squared_error(test_actual, test_whitebox))
                pinn_rmse = np.sqrt(mean_squared_error(test_actual, pinn_predictions.flatten()))

                whitebox_r2 = r2_score(test_actual, test_whitebox)
                pinn_r2 = r2_score(test_actual, pinn_predictions.flatten())

                # Add metrics to plot
                plt.annotate(f'Test Period Metrics:\nWhite Box - RMSE: {whitebox_rmse:.4f}, R²: {whitebox_r2:.4f}\n'
                            f'PINN - RMSE: {pinn_rmse:.4f}, R²: {pinn_r2:.4f}',
                            xy=(0.02, 0.02), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

            plt.title(f'Temperature Prediction Comparison - {equipment_name}')
            plt.xlabel('Date')
            plt.ylabel('Temperature')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            # Save the plot
            plt.savefig(f'prediction_plot_{equipment_name}.png')
            plt.close()

            # Use PINN predictions instead of whitebox for export
            equipment_df_test['value'] = pinn_predictions.flatten()
            print(f"Using PINN predictions for {equipment_name}")

        except Exception as e:
            # Fallback to whitebox predictions if PINN model fails
            print(f"Failed to load PINN model for {equipment_name}, using whitebox predictions instead: {e}")
            equipment_df_test['value'] = equipment_df_test['Temp Pred']

            # Still create a plot but only with actual and whitebox
            plot_df = equipment_df.copy()
            test_mask = plot_df['dateTime'] >= split_datetime

            plt.figure(figsize=(12, 6))
            plt.plot(plot_df['dateTime'], plot_df['temperature'], label='Actual', color='black')
            plt.plot(plot_df['dateTime'], plot_df['Temp Pred'], label='White Box', color='blue', linestyle='--')
            plt.axvline(x=split_datetime, color='green', linestyle='-', alpha=0.5, label='Train/Test Split')

            if sum(test_mask) > 0:
                test_actual = plot_df.loc[test_mask, 'temperature'].values
                test_whitebox = plot_df.loc[test_mask, 'Temp Pred'].values
                whitebox_rmse = np.sqrt(mean_squared_error(test_actual, test_whitebox))
                whitebox_r2 = r2_score(test_actual, test_whitebox)
                plt.annotate(f'Test Period Metrics:\nWhite Box - RMSE: {whitebox_rmse:.4f}, R²: {whitebox_r2:.4f}',
                            xy=(0.02, 0.02), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

            plt.title(f'Temperature Prediction Comparison - {equipment_name} (PINN model unavailable)')
            plt.xlabel('Date')
            plt.ylabel('Temperature')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'prediction_plot_{equipment_name}.png')
            plt.close()

        # Prepare for export
        equipment_df_test['equipmentId'] = int(equipment_name)
        equipment_df_test['dateTime'] = equipment_df_test['dateTime'].dt.strftime('%Y-%m-%dT%H:%M:%S.0000000')
        equipment_df_test = equipment_df_test[['equipmentId', 'dateTime','value']]

        # Save to CSV
        os.makedirs('test', exist_ok=True)
        equipment_df_test.to_csv('test/' + str(equipment_name) + '.csv', index=False)
        print(f"Saved predictions to test/{equipment_name}.csv")
        # Data is separated into smaller chunks to not exceed the limit of uploading to the API
        chunk_size = 2950
        chunks = split_dataframe(equipment_df_test, chunk_size)

        # Comment out the API upload or replace with your actual endpoint
        upload_path = 'prediction/Temperature'
        api_url = f"https://your-api-endpoint.com/{upload_path}"  # Replace with your actual API endpoint

        print(f"Uploading {len(chunks)} chunks for {equipment_name}...")

        for i, chunk in enumerate(chunks):
            dict_data = chunk.to_dict(orient='records')
            json_data = json.dumps(dict_data, indent=4)

            # Save the JSON for inspection (optional)
            with open(f'output_{equipment_name}_chunk{i}.json', 'w') as json_file:
                json_file.write(json_data)

            # Uncomment to enable actual API upload
            # success = post_data(json_data, api_url)
            # if success:
            #     print(f"Successfully uploaded chunk {i+1} of {len(chunks)} for {equipment_name}")
            # else:
            #     print(f"Failed to upload chunk {i+1} of {len(chunks)} for {equipment_name}")

            # Simulate successful upload for now
            print(f"Simulated upload of chunk {i+1} of {len(chunks)} for {equipment_name}")

print("\nPrediction generation and export completed.")
end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")