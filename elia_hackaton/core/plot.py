from elia_hackaton.config import IMAGES_DIR

# Training visualization
def plot_training_curves(train_losses, val_losses, equipment_name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {equipment_name}')
    plt.legend()
    plt.savefig(f'training_curves_{equipment_name}.png')
    plt.close()
