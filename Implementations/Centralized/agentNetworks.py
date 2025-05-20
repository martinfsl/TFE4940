import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

import copy

from constants import *
from fh_pattern import FH_Pattern

class ActorNetwork(nn.Module):
    def __init__(self, device = "cpu"):
        super(ActorNetwork, self).__init__()

        self.input_size = PPO_NETWORK_INPUT_SIZE
        self.hidden_size1 = 128
        self.hidden_size2 = 64
        self.output_size = PPO_NETWORK_OUTPUT_SIZE

        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(self.hidden_size2, self.output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
    
class CriticNetwork(nn.Module):
    def __init__(self, device = "cpu"):
        super(CriticNetwork, self).__init__()

        self.input_size = PPO_NETWORK_INPUT_SIZE
        self.hidden_size1 = 128
        self.hidden_size2 = 64
        self.output_size = 1

        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(self.hidden_size2, self.output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
    
class Network(nn.Module):
    def __init__(self, device="cpu"):
        super(Network, self).__init__()

        self.actor = ActorNetwork()
        self.critic = CriticNetwork()

        self.device = device

    def forward(self, x):
        actor = self.actor(x)
        value = self.critic(x)

        return actor, value