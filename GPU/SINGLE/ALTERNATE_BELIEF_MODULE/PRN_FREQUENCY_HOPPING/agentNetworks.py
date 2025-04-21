import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

import copy

from constants import *
from fh_pattern import FH_Pattern

#################################################################################
### Defining classes for the Belief Module
#################################################################################

# Create a class for the neural network that predicts the other agent's action
# The input to the neural network is the history of the other agent's actions
# The neural network is made up off 2 fully connected layers with ReLU activation functions
# The neural network has a dropout rate of 30%
class PredictionNN(nn.Module):
    def __init__(self):
        super(PredictionNN, self).__init__()

        self.input_size = PREDICTION_NETWORK_INPUT_SIZE
        self.hidden_size1 = 128
        self.hidden_size2 = 64
        # self.hidden_size = 64
        self.output_size = PREDICTION_NETWORK_OUTPUT_SIZE

        # Defining the fully connected layers
        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(self.hidden_size2, self.output_size)
        # self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        # self.dropout1 = nn.Dropout(0.3)
        # self.fc3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
    
# Create a class for the agent that uses the neural network to predict the other agent's action
# The agent has a memory to store experiences
# The agent uses the Adam optimizer to train the neural network
# The agent uses the cross entropy loss function to train the neural network
# The output of the neural network is the predicted Rx's action using a softmax function
class PredictionNNAgent:
    def __init__(self, device = "cpu"):
        self.learning_rate = BELIEF_MODULE_LEARNING_RATE
        # self.learning_rate = 0.005

        # Parameters for the neural network
        self.batch_size = BELIEF_MODULE_BATCH_SIZE
        self.maximum_memory_size = BELIEF_MODULE_MEMORY_SIZE
        self.epochs = BELIEF_MODULE_EPOCHS

        self.device = device

        self.memory_state = torch.empty((0, PREDICTION_NETWORK_INPUT_SIZE), device=self.device)
        self.memory_action = torch.empty((0, 1), device=self.device)

        self.losses = torch.tensor([], device=self.device)

        self.pred_network = PredictionNN()
        self.pred_network.to(self.device)
        self.optimizer = optim.Adam(self.pred_network.parameters(), lr=self.learning_rate)

    def store_in_memory(self, state, action):
        if self.memory_state.size(0) >= self.maximum_memory_size:
            self.memory_state = self.memory_state[1:]
            self.memory_action = self.memory_action[1:]

        self.memory_state = torch.cat((self.memory_state, state.unsqueeze(0)), dim=0)
        self.memory_action = torch.cat((self.memory_action, action.unsqueeze(0)), dim=0)

    def train(self):
        if self.memory_state.size(0) >= self.batch_size:
            for _ in range(self.epochs):
                indices = random.sample(range(self.memory_state.size(0)), self.batch_size)

                batch_state = self.memory_state[indices]
                batch_action = self.memory_action[indices]

                pred = self.pred_network(batch_state)
                loss = nn.CrossEntropyLoss()(pred, batch_action.long().squeeze())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.losses = torch.cat((self.losses, loss.unsqueeze(0)))

    def predict_action(self, observation):
        with torch.no_grad():
            pred = self.pred_network(observation)
        return nn.Softmax(dim=0)(pred)
    
#################################################################################
### Alternate Model 1
#################################################################################

# class ActorNetwork(nn.Module):
#     def __init__(self):
#         super(ActorNetwork, self).__init__()

#         self.input_size = PPO_NETWORK_INPUT_SIZE
#         self.hidden_size1 = 128
#         self.hidden_size2 = 64
#         self.output_size = PPO_NETWORK_OUTPUT_SIZE

#         # Defining the fully connected layers
#         if USE_PREDICTION:
#             self.fc1 = nn.Linear(self.input_size + PREDICTION_NETWORK_OUTPUT_SIZE, self.hidden_size1)
#         else:
#             self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
#         self.dropout1 = nn.Dropout(0.3)
#         self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
#         self.dropout2 = nn.Dropout(0.3)
#         self.fc3 = nn.Linear(self.hidden_size2, self.output_size)

#     def forward(self, obs, encoded_belief, dimension=0):
#         if USE_PREDICTION:
#             fused = torch.cat([obs, encoded_belief], dim=dimension)
#         else:
#             fused = obs
#         x = torch.relu(self.fc1(fused))
#         x = self.dropout1(x)
#         x = torch.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = self.fc3(x)

#         return x
    
# class CriticNetwork(nn.Module):
#     def __init__(self):
#         super(CriticNetwork, self).__init__()

#         self.input_size = PPO_NETWORK_INPUT_SIZE
#         self.hidden_size1 = 128
#         self.hidden_size2 = 64
#         self.output_size = 1

#         # Defining the fully connected layers
#         if USE_PREDICTION:
#             self.fc1 = nn.Linear(self.input_size + PREDICTION_NETWORK_OUTPUT_SIZE, self.hidden_size1)
#         else:
#             self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
#         self.dropout1 = nn.Dropout(0.3)
#         self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
#         self.dropout2 = nn.Dropout(0.3)
#         self.fc3 = nn.Linear(self.hidden_size2, self.output_size)

#     def forward(self, obs, encoded_belief, dimension=0):
#         if USE_PREDICTION:
#             fused = torch.cat([obs, encoded_belief], dim=dimension)
#         else:
#             fused = obs
#         x = torch.relu(self.fc1(fused))
#         x = self.dropout1(x)
#         x = torch.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = self.fc3(x)

#         return x

#################################################################################
#################################################################################
#################################################################################

#################################################################################
### Alternate Model 2
#################################################################################

OBSERVATION_ENCODER_OUTPUT_SIZE = 50

class ObservationEncoder(nn.Module):
    def __init__(self):
        super(ObservationEncoder, self).__init__()

        self.input_size = PPO_NETWORK_INPUT_SIZE
        self.hidden_size = 128
        self.output_size = OBSERVATION_ENCODER_OUTPUT_SIZE

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)

        return x
    
class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()

        self.obs_encoder = ObservationEncoder()

        self.input_size = OBSERVATION_ENCODER_OUTPUT_SIZE
        self.hidden_size1 = 128
        self.hidden_size2 = 64
        self.output_size = PPO_NETWORK_OUTPUT_SIZE

        # Defining the fully connected layers
        if USE_PREDICTION:
            self.fc1 = nn.Linear(self.input_size + PREDICTION_NETWORK_OUTPUT_SIZE, self.hidden_size1)
        else:
            self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(self.hidden_size2, self.output_size)

    def forward(self, obs, encoded_belief, dimension=0):
        encoded_obs = self.obs_encoder(obs)
        if USE_PREDICTION:
            fused = torch.cat([encoded_obs, encoded_belief], dim=dimension)
        else:
            fused = encoded_obs
        x = torch.relu(self.fc1(fused))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
    
class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()

        self.obs_encoder = ObservationEncoder()

        self.input_size = OBSERVATION_ENCODER_OUTPUT_SIZE
        self.hidden_size1 = 128
        self.hidden_size2 = 64
        self.output_size = 1

        # Defining the fully connected layers
        if USE_PREDICTION:
            self.fc1 = nn.Linear(self.input_size + PREDICTION_NETWORK_OUTPUT_SIZE, self.hidden_size1)
        else:
            self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(self.hidden_size2, self.output_size)

    def forward(self, obs, encoded_belief, dimension=0):
        encoded_obs = self.obs_encoder(obs)
        if USE_PREDICTION:
            fused = torch.cat([encoded_obs, encoded_belief], dim=dimension)
        else:
            fused = encoded_obs
        x = torch.relu(self.fc1(fused))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
    
#################################################################################
#################################################################################
#################################################################################