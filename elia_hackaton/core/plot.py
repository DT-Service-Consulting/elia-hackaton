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
