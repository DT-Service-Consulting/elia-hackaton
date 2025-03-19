import  torch

# Compute RMSE for the white-box model
def white_box_model(x, R=0.1, K=0.05):
    theta_a = x[:, 3]  # ambient temperature
    theta_or = x[:, 4]  # delta top oil
    theta_hr = x[:, -2]  # heat run test y (assumption)
    x_param = x[:, -3]  # heat run test x (assumption)
    y_param = x[:, -1]  # heat run test gradient (assumption)

    white_box_pred = ((1 + R * K**2) / (1 + R)) ** x_param * \
        (theta_or - theta_a) + K**y_param * (theta_hr - theta_or)
    return white_box_pred


def setup_gpu():
    if torch.cuda.is_available():
        # Get GPU information
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPU(s):")

#        for i in range(gpu_count):
#            gpu_name = torch.cuda.get_device_name(i)
#            print(f"  GPU {i}: {gpu_name}")

        # Set device to the first available GPU
        device = torch.device("cuda:0")

        # Print CUDA version
        print(f"CUDA Version: {torch.version.cuda}")

        # Configure for mixed precision training if available
        if hasattr(torch.cuda, 'amp') and torch.cuda.is_available():
            print("Mixed precision training available")
        else:
            print("Mixed precision training not available")

        # Ensure cuDNN is enabled for better performance
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            print("cuDNN enabled for faster training")

    elif torch.backends.mps.is_available():
        print("MPS available")
        device = torch.device("mps")  # Use Apple Metal (macOS)
    else:
        print("No GPU available, using CPU")
        device = torch.device("cpu")  # Default to CPU

    print(f"Using device: {device}")
    return device
