import numpy as np
import pandas as pd
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

from elia_hackaton.config import DATA_DIR
print(DATA_DIR)
equipment = pd.read_csv(DATA_DIR / 'Equipment.csv', index_col=0)
print(equipment)