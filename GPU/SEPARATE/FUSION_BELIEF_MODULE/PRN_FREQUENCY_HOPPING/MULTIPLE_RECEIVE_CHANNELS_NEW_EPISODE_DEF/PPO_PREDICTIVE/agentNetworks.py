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
    
class BeliefEncoder(nn.Module):
    def __init__(self):
        super(BeliefEncoder, self).__init__()

        self.input_size = PREDICTION_NETWORK_INPUT_SIZE
        self.hidden_size = 128
        self.output_size = PREDICTION_NETWORK_OUTPUT_SIZE
        
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
    
class ActorCriticNetwork_(nn.Module):
    def __init__(self):
        super(ActorCriticNetwork_, self).__init__()

        self.obs_encoder = ObservationEncoder()
        self.belief_encoder = BeliefEncoder()

        self.hidden_size = 128

        if USE_PREDICTION:
            self.fusion_layer = nn.Linear(PREDICTION_NETWORK_OUTPUT_SIZE + PPO_NETWORK_OUTPUT_SIZE, self.hidden_size)
        else:
            self.fusion_layer = nn.Linear(PPO_NETWORK_OUTPUT_SIZE, self.hidden_size)

        self.actor_head = nn.Linear(self.hidden_size, PPO_NETWORK_OUTPUT_SIZE)
        self.critic_head = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, obs, belief, dimension = 0):
        # Encode the observation and belief
        obs_encoded = self.obs_encoder(obs)

        if USE_PREDICTION:
            belief_encoded = self.belief_encoder(belief)
        else:
            belief_encoded = belief

        fused = torch.cat([obs_encoded, belief_encoded], dim=dimension)
        fused = torch.relu(self.fusion_layer(fused))

        action_probs = self.actor_head(fused)
        state_value = self.critic_head(fused)

        return action_probs, state_value
##################################################################

##################################################################
class PredictionNetwork(nn.Module):
    def __init__(self):
        super(PredictionNetwork, self).__init__()

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
    
class PredictionNetworkAgent:
    def __init__(self, device = "cpu"):
        # Parameters for the neural network
        self.learning_rate = 0.005
        self.batch_size = 4
        self.maximum_memory_size = 25

        self.device = device

        self.memory_state = torch.empty((0, PREDICTION_NETWORK_INPUT_SIZE), device=self.device)
        self.memory_action = torch.empty((0, 1), device=self.device)

        self.pred_network = PredictionNetwork()
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
            indices = random.sample(range(self.memory_state.size(0)), self.batch_size)

            batch_state = self.memory_state[indices]
            batch_action = self.memory_action[indices]

            pred = self.pred_network(batch_state)
            loss = nn.CrossEntropyLoss()(pred, batch_action.long().squeeze())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict_action(self, observation):
        with torch.no_grad():
            pred = self.pred_network(observation)
        return nn.Softmax(dim=0)(pred)
    
class PPOActor(nn.Module):
    def __init__(self):
        super(PPOActor, self).__init__()

        self.input_size = PPO_NETWORK_INPUT_SIZE
        self.hidden_size1 = 128
        self.hidden_size2 = 64
        self.output_size = PPO_NETWORK_OUTPUT_SIZE

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
    
class PPOCritic(nn.Module):
    def __init__(self):
        super(PPOCritic, self).__init__()

        self.input_size = PPO_NETWORK_INPUT_SIZE
        self.hidden_size1 = 128
        self.hidden_size2 = 64
        self.output_size = 1

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
##################################################################