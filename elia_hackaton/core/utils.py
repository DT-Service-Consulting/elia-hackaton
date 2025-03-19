import torch


def setup_gpu():
    """
    Set up the GPU or CPU device for PyTorch.

    This function checks for the availability of CUDA-enabled GPUs or Apple Metal (MPS) on macOS.
    It configures the device for mixed precision training if available and enables cuDNN for better performance.

    Returns:
    torch.device: The device to be used for PyTorch operations.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

        if hasattr(torch.cuda, 'amp'):
            print("Mixed precision training available")
        else:
            print("Mixed precision training not available")

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            print("cuDNN enabled for faster training")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal (MPS)")

    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")

    print(f"Using device: {device}")
    return device
