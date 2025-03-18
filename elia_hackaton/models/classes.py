import torch.nn as nn
import torch


class basic_model(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden1 = nn.Linear(5, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.hidden3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        return self.output(x)


# Define PINN model
class PINN(nn.Module):
    def __init__(self, input_size, hidden_layers, neurons):
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
        return self.model(x)
