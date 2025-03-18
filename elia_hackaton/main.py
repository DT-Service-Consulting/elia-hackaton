import numpy as np
import torch
import torch.nn as nn
import requests
from scipy.optimize import linprog

from elia_hackaton.core import fetch_data

#fetch_data()
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")  # Use NVIDIA GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Use Apple Metal (macOS)
else:
    device = torch.device("cpu")  # Default to CPU

print(f"Using device: {device}")

