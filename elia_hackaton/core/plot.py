from elia_hackaton.config import IMAGES_DIR
import matplotlib.pyplot as plt


def plot_training_curves(train_losses, val_losses, equipment_name):
    """
    Plot and save the training and validation loss curves.

    This function generates a plot of the training and validation loss curves over epochs
    and saves the plot as a PNG file.

    Parameters:
    train_losses (list of float): List of training loss values for each epoch.
    val_losses (list of float): List of validation loss values for each epoch.
    equipment_name (str): The name of the equipment being trained.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {equipment_name}')
    plt.legend()
    plt.savefig(IMAGES_DIR / f'training_curves_{equipment_name}.png')
    plt.close()


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_load_vs_time(csv_file, period='day'):
    # Load the CSV file
    df = pd.read_csv(csv_file, parse_dates=['dateTime'])

    # Extract the selected period
    if period == 'day':
        df['period'] = df['dateTime'].dt.date.astype(str)
        time_format = '%H:%M'
    elif period == 'month':
        df['period'] = df['dateTime'].dt.to_period('M').astype(str)
        time_format = '%d'
    elif period == 'year':
        df['period'] = df['dateTime'].dt.to_period('Y').astype(str)
        time_format = '%m'
    else:
        raise ValueError("Period must be 'day', 'month', or 'year'")

    # Group by the selected period
    grouped = df.groupby('period')

    plt.figure(figsize=(12, 6))
    all_times = []
    all_loads = []

    for period, group in grouped:
        # Normalize time for x-axis
        group = group.sort_values(by='dateTime')
        group['time'] = group['dateTime'].dt.strftime(time_format)

        all_times.append(group['time'].values)
        all_loads.append(group['load'].values)
        plt.plot(group['time'], group['load'], color='blue', alpha=0.3)

    # Compute the average load per time unit within the period
    df['time'] = df['dateTime'].dt.strftime(time_format)
    avg_load = df.groupby('time')['load'].mean()

    plt.plot(avg_load.index, avg_load.values, color='red', linewidth=2, label='Average Load')
    plt.xlabel('Time')
    plt.ylabel('Load')
    plt.title(f'Load vs Time ({period})')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.show()

    return pd.DataFrame(avg_load).reset_index()


import pandas as pd


def add_time_increment_column(df, period='day'):
    """
    Add a time increment column to the DataFrame.

    This function calculates the time increment by subtracting the first two time values
    and adds it as a new column to the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the time column.
    time_column (str): The name of the time column.

    Returns:
    pd.DataFrame: The DataFrame with the added time increment column.
    """
    if period == 'day':
        increment = pd.Timedelta('15 minutes')
    elif period == 'month':
        increment = pd.Timedelta('1 day')
    elif period == 'year':
        increment = pd.Timedelta('1 month')

    # Calculate the time increment in minutes
    df['time_increment'] = increment
    df['increment_times_line'] = df.index * increment

    return df, increment


def curve_above_constant(df, y_column, constant):
    # Interpolate to get y value at point x
    # Extract x and y values
    df[y_column] = df[y_column].apply(lambda y: y - constant if y > constant else 0)
    return df


def calculate_area(df, y_column, increment):
    return df[y_column].sum() * increment.total_seconds() / 3600
