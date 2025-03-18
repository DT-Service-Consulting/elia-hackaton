import torch.nn as nn
import torch

class basic_model(nn.Module):
    def __init__(self):
        """
        Initialize the basic_model class.

        This model consists of three hidden layers and one output layer.
        The hidden layers use ReLU activation functions.

        Layers:
        - hidden1: Linear layer with 5 input features and 64 output features.
        - hidden2: Linear layer with 64 input features and 64 output features.
        - hidden3: Linear layer with 64 input features and 32 output features.
        - output: Linear layer with 32 input features and 1 output feature.
        """
        super(basic_model, self).__init__()
        self.hidden1 = nn.Linear(5, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.hidden3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        """
        Forward pass of the basic_model.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after passing through the network.
        """
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        return self.output(x)


class PINN(nn.Module):
    def __init__(self, input_size, hidden_layers, neurons):
        """
        Initialize the PINN class.

        This model consists of a configurable number of hidden layers and neurons per layer.
        Each hidden layer uses a ReLU activation function.

        Parameters:
        input_size (int): The number of input features.
        hidden_layers (int): The number of hidden layers.
        neurons (int): The number of neurons in each hidden layer.
        """
        super(PINN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, neurons))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(neurons, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the PINN.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after passing through the network.
        """
        return self.model(x)

class Transformer:
    def __init__(self, load, latitude, longitude, nn_params=None):
        """
        Initialize the Transformer class.

        Parameters:
        load (float): The load of the transformer.
        latitude (float): The latitude of the transformer's location.
        longitude (float): The longitude of the transformer's location.
        nn_params (dict): Dictionary containing neural network parameters.
        """
        self.load = load
        self.latitude = latitude
        self.longitude = longitude
        self.nn_params = nn_params if nn_params is not None else {}

    def get_location(self):
        """
        Get the location of the transformer.

        Returns:
        tuple: A tuple containing the latitude and longitude.
        """
        return self.latitude, self.longitude

    def get_nn_params(self):
        """
        Get the neural network parameters.

        Returns:
        dict: The neural network parameters.
        """
        return self.nn_params