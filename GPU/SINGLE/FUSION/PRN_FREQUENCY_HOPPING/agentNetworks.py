import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

import copy

from constants import *
from fh_pattern import FH_Pattern
    
#################################################################################
### Fusion Model
#################################################################################

# class ActorNetwork(nn.Module):
#     def __init__(self):
#         super(ActorNetwork, self).__init__()

#         self.input_size = PPO_NETWORK_INPUT_SIZE
#         self.hidden_size1 = 128
#         self.hidden_size2 = 64
#         self.output_size = PPO_NETWORK_OUTPUT_SIZE

#         # Defining the fully connected layers
#         if USE_PREDICTION and USE_STANDALONE_BELIEF_MODULE:
#             self.fc1 = nn.Linear(self.input_size + PREDICTION_NETWORK_OUTPUT_SIZE, self.hidden_size1)
#         elif USE_PREDICTION and not USE_STANDALONE_BELIEF_MODULE:
#             self.fc1 = nn.Linear(self.input_size + P, self.hidden_size1)
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
#         if USE_PREDICTION and USE_STANDALONE_BELIEF_MODULE:
#             self.fc1 = nn.Linear(self.input_size + PREDICTION_NETWORK_OUTPUT_SIZE, self.hidden_size1)
#         elif USE_PREDICTION and not USE_STANDALONE_BELIEF_MODULE:
#             self.fc1 = nn.Linear(self.input_size + P, self.hidden_size1)
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
### Gated Fusion
#################################################################################

# class GatedFusion(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.obs_proj    = nn.Linear(obs_dim, 64)
#         self.belief_proj = nn.Linear(belief_dim, 64)
#         self.gate        = nn.Linear(obs_dim + belief_dim,  64)  # gate input = concatenated raw inputs
    
#     def forward(self, obs, belief, dimension):
#         fo = torch.relu(self.obs_proj(obs))
#         fb = torch.relu(self.belief_proj(belief))
#         # compute gate g in (0,1)
#         g = torch.sigmoid(self.gate(torch.cat([obs, belief],dim=dimension)))
#         return g*fb + (1-g)*fo
    
class ObservationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_proj = nn.Linear(PPO_NETWORK_INPUT_SIZE, 64)

    def forward(self, obs):
        return torch.relu(self.obs_proj(obs))

class BeliefEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.belief_proj = nn.Linear(PREDICTION_NETWORK_INPUT_SIZE, PREDICTION_NETWORK_OUTPUT_SIZE)

    def forward(self, belief, dimension=0):
        # return torch.softmax(self.belief_proj(belief), dim=dimension)
        return torch.relu(self.belief_proj(belief))

class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()

        self.obs_encoder = ObservationEncoder()
        # self.gate = GatedFusion()


        # self.fusion_size = PPO_NETWORK_INPUT_SIZE + PREDICTION_NETWORK_OUTPUT_SIZE
        self.fusion_size = 64 + PREDICTION_NETWORK_OUTPUT_SIZE
        self.hidden_size = 64
        self.output_size = PPO_NETWORK_OUTPUT_SIZE

        # Defining the fully connected layers
        if USE_PREDICTION:
            self.fc1 = nn.Linear(self.fusion_size, self.hidden_size)
        else:
            self.fc1 = nn.Linear(64, self.hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, obs, encoded_belief, dimension=0):
        encoded_obs = self.obs_encoder(obs)
        if USE_PREDICTION:
            # fused = torch.cat([obs, encoded_belief], dim=dimension)
            fused = torch.cat([encoded_obs, encoded_belief], dim=dimension)
            # fused = self.gate(encoded_obs, encoded_belief, dimension)
        else:
            fused = encoded_obs
        x = torch.relu(self.fc1(fused))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x
    
class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()

        self.obs_encoder = ObservationEncoder()
        # self.gate = GatedFusion()


        # self.fusion_size = PPO_NETWORK_INPUT_SIZE + PREDICTION_NETWORK_OUTPUT_SIZE
        self.fusion_size = 64 + PREDICTION_NETWORK_OUTPUT_SIZE
        self.hidden_size = 64
        self.output_size = 1

        # Defining the fully connected layers
        if USE_PREDICTION:
            self.fc1 = nn.Linear(self.fusion_size, self.hidden_size)
        else:
            self.fc1 = nn.Linear(64, self.hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, obs, encoded_belief, dimension=0):
        encoded_obs = self.obs_encoder(obs)
        if USE_PREDICTION:
            # fused = torch.cat([obs, encoded_belief], dim=dimension)
            fused = torch.cat([encoded_obs, encoded_belief], dim=dimension)
            # fused = self.gate(encoded_obs, encoded_belief, dimension)
        else:
            fused = encoded_obs
        x = torch.relu(self.fc1(fused))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x
    
class Network(nn.Module):
    def __init__(self, device="cpu"):
        super(Network, self).__init__()

        self.actor = ActorNetwork()
        self.critic = CriticNetwork()
        self.belief_encoder = BeliefEncoder()

        self.device = device

    def forward(self, obs, belief, dimension=0):
        encoded_belief = self.belief_encoder(belief)
        actor = self.actor(obs, encoded_belief, dimension)
        value = self.critic(obs, encoded_belief, dimension)

        if USE_PREDICTION:
            pred_action = torch.argmax(encoded_belief).unsqueeze(0)
        else:
            pred_action = torch.tensor([-1], device=self.device)

        return actor, value, pred_action

#################################################################################
#################################################################################
#################################################################################