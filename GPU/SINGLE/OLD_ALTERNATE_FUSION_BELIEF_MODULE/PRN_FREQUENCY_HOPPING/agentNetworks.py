import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

import copy

from constants import *
from fh_pattern import FH_Pattern

##################################################################
class ObservationEncoder(nn.Module):
    def __init__(self):
        super(ObservationEncoder, self).__init__()

        self.input_size = PPO_NETWORK_INPUT_SIZE
        self.hidden_size = 128
        self.output_size = PPO_NETWORK_OUTPUT_SIZE

        # Define the layers
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        # Pass the input through the layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return x
    
class BeliefModule(nn.Module):
    def __init__(self):
        super(BeliefModule, self).__init__()

        self.input_size = PREDICTION_NETWORK_INPUT_SIZE
        self.hidden_size1 = 128
        self.hidden_size2 = 64
        self.output_size = PREDICTION_NETWORK_OUTPUT_SIZE

        # Define the layers
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
    
class BeliefModuleAgent(nn.Module):
    def __init__(self, device = "cpu"):
        super(BeliefModuleAgent, self).__init__()

        self.learning_rate = 0.005

        self.batch_size = 4
        self.maximum_memory_size = 25
        self.epochs = 5

        self.device = device

        self.memory_state = torch.empty((0, PREDICTION_NETWORK_INPUT_SIZE), device=self.device)
        self.memory_action = torch.empty((0, 1), device=self.device)

        self.losses = torch.tensor([], device=self.device)

        self.belief_network = BeliefModule()
        self.belief_network.to(self.device)
        self.optimizer = optim.Adam(self.belief_network.parameters(), lr=self.learning_rate)
    
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

                # Forward pass
                predictions = self.belief_network(batch_state)

                # Calculate loss
                loss = nn.CrossEntropyLoss()(predictions, batch_action.long().squeeze())

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Store the loss
            self.losses = torch.cat((self.losses, loss.unsqueeze(0)))

    def predict_action(self, observation):
        with torch.no_grad():
            pred = self.belief_network(observation)
         
        return nn.Softmax(dim=0)(pred)

class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()

        self.obs_encoder = ObservationEncoder()

        self.hidden_size = 128

        if USE_PREDICTION:
            self.fusion_layer = nn.Linear(PREDICTION_NETWORK_OUTPUT_SIZE + PPO_NETWORK_OUTPUT_SIZE, self.hidden_size)
        else:
            self.fusion_layer = nn.Linear(PPO_NETWORK_OUTPUT_SIZE, self.hidden_size)

        self.actor_head = nn.Linear(self.hidden_size, PPO_NETWORK_OUTPUT_SIZE)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, obs, encoded_belief, dimension=0):
        # Encode the observation and belief
        obs_encoded = self.obs_encoder(obs)

        fused = torch.cat([obs_encoded, encoded_belief], dim=dimension)
        fused = torch.relu(self.fusion_layer(fused))

        fused = self.dropout(fused)

        output = self.actor_head(fused)

        return output
    
class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()

        self.obs_encoder = ObservationEncoder()

        self.hidden_size = 128

        if USE_PREDICTION:
            self.fusion_layer = nn.Linear(PREDICTION_NETWORK_OUTPUT_SIZE + PPO_NETWORK_OUTPUT_SIZE, self.hidden_size)
        else:
            self.fusion_layer = nn.Linear(PPO_NETWORK_OUTPUT_SIZE, self.hidden_size)

        self.critic_head = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, obs, encoded_belief):
        # Encode the observation and belief
        obs_encoded = self.obs_encoder(obs)

        fused = torch.cat([obs_encoded, encoded_belief])
        fused = torch.relu(self.fusion_layer(fused))

        fused = self.dropout(fused)

        output = self.critic_head(fused)

        return output
##################################################################