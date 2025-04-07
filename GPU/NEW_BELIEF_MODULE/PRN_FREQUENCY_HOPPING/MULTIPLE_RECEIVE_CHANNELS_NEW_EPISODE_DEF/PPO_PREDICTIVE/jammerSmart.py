import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

import copy

from constants import *

#################################################################################
### Defining class for the PPO for the smart jammer
#################################################################################

class jammerPPOActor(nn.Module):
    def __init__(self, device = "cpu"):
        super(jammerPPOActor, self).__init__()

        self.input_size = NUM_JAMMER_SENSE_CHANNELS + 1
        # self.input_size = NUM_CHANNELS + 1
        self.hidden_size1 = 128
        self.hidden_size2 = 64
        self.output_size = NUM_CHANNELS

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
    
class jammerPPOCritic(nn.Module):
    def __init__(self, device = "cpu"):
        super(jammerPPOCritic, self).__init__()

        self.input_size = NUM_JAMMER_SENSE_CHANNELS + 1
        # self.input_size = NUM_CHANNELS + 1
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
    
class jammerPPOAgent:
    def __init__(self, gamma = GAMMA, learning_rate = LEARNING_RATE,
                lambda_param = LAMBDA, epsilon_clip = EPSILON_CLIP,
                k = K, m = M, c1 = C1, c2 = C2, device = "cpu"):
        self.gamma = gamma
        # self.learning_rate = 0.001
        self.learning_rate = 0.0001

        # self.LAMBDA = lambda_param
        self.LAMBDA = 0.80
        self.epsilon_clip = epsilon_clip

        self.k = k
        self.m = m
        self.c1 = 0.5
        # self.c2 = 0.01
        self.c2 = 0.1

        self.device = device

        self.actor_network = jammerPPOActor(device=device)
        self.actor_network.to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.learning_rate)

        self.actor_network_old = copy.deepcopy(self.actor_network).to(self.device)
        
        self.critic_network = jammerPPOCritic(device=device)
        self.critic_network.to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.learning_rate)

        self.memory_state = torch.empty((0, NUM_JAMMER_SENSE_CHANNELS+1), device=self.device)
        # self.memory_state = torch.empty((0, NUM_CHANNELS+1), device=self.device)
        self.memory_action = torch.empty((0, 1), device=self.device)
        self.memory_logprob = torch.empty((0, 1), device=self.device)
        self.memory_reward = torch.empty((0, 1), device=self.device)
        self.memory_value = torch.empty((0, 1), device=self.device)

    def clear_memory(self):
        self.memory_state = torch.empty((0, NUM_JAMMER_SENSE_CHANNELS+1), device=self.device)
        # self.memory_state = torch.empty((0, NUM_CHANNELS+1), device=self.device)
        self.memory_action = torch.empty((0, 1), device=self.device)
        self.memory_logprob = torch.empty((0, 1), device=self.device)
        self.memory_reward = torch.empty((0, 1), device=self.device)
        self.memory_value = torch.empty((0, 1), device=self.device)

    def store_in_memory(self, state, action, logprob, reward, value):
        self.memory_state = torch.cat((self.memory_state, state.unsqueeze(0)), dim=0)
        self.memory_action = torch.cat((self.memory_action, action.unsqueeze(0)), dim=0)
        self.memory_logprob = torch.cat((self.memory_logprob, logprob.unsqueeze(0).unsqueeze(0)), dim=0)
        self.memory_reward = torch.cat((self.memory_reward, reward.unsqueeze(0)), dim=0)
        self.memory_value = torch.cat((self.memory_value, value.unsqueeze(0).unsqueeze(0)), dim=0)

    def choose_action(self, observation):
        with torch.no_grad():
            policy = nn.Softmax(dim=0)(self.actor_network(observation))
            value = self.critic_network(observation)

        action = torch.argmax(policy)
        logprob = torch.log(policy[action])

        return action, logprob, value
    
    def compute_returns(self):
        returns = torch.empty((0, 1), device=self.device)
        R = 0
        for r in reversed(self.memory_reward):
            R = r + self.gamma * R
            returns = torch.cat((R.unsqueeze(0), returns), dim=0)
        return returns

    def compute_advantages(self):
        # Generalized Advantage Estimation (GAE):
        advantages = torch.empty((0, 1), device=self.device)
        gae = 0
        for i in reversed(range(len(self.memory_reward))):
            if i == len(self.memory_reward) - 1:
                delta = self.memory_reward[i] - self.memory_value[i]
            else:
                delta = self.memory_reward[i] + self.gamma*self.memory_value[i+1] - self.memory_value[i]
            gae = delta + self.gamma*self.LAMBDA*gae
            advantages = torch.cat((gae.unsqueeze(0), advantages), dim=0)
        
        # Normalize the advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        return advantages

    def update(self):
        returns = self.compute_returns()
        advantages = self.compute_advantages()

        values = self.critic_network(self.memory_state)

        old_logits = self.actor_network_old(self.memory_state).detach()

        data_size = self.memory_state.size(0)

        for epoch in range(self.k):
            batch_indices = torch.randperm(data_size)[:self.m]

            batch_state = self.memory_state[batch_indices]
            batch_action = self.memory_action[batch_indices]
            batch_logprob = self.memory_logprob[batch_indices]
            batch_return = returns[batch_indices].detach()
            batch_advantage = advantages[batch_indices].detach()

            new_logits = self.actor_network(batch_state)
            new_policy = nn.Softmax(dim=1)(new_logits)
            new_logprobs = torch.log(torch.gather(new_policy, 1, batch_action.long()))

            ratio = torch.exp(new_logprobs - batch_logprob)

            surr1 = ratio*batch_advantage
            surr2 = torch.clamp(ratio, 1-self.epsilon_clip, 1+self.epsilon_clip)*batch_advantage

            entropy = -(new_policy*torch.log(new_policy)).sum(dim=1).mean()

            actor_loss = -torch.min(surr1, surr2).mean() - self.c2*entropy

            batch_value = self.critic_network(batch_state)
            critic_loss = nn.MSELoss()(batch_value, batch_return)

            total_loss = actor_loss + self.c1*critic_loss

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.actor_network_old.load_state_dict(self.actor_network.state_dict())

        self.clear_memory()

#################################################################################
### Defining class for a smart jammer
#################################################################################

# Class that represents a smart jammer in the environment
# The jammer uses a DQN to learn the optimal policy, which is used to predict the frequency band chosen for the Tx-Rx link
class SmartJammer:
    def __init__(self, type = "PPO", device = "cpu"):
        self.power = JAMMER_TRANSMIT_POWER
        self.h_jt_variance = H_JT_VARIANCE # Variance of the Rayleigh distribution for the Rayleigh fading from jammer to transmitter
        self.h_jr_variance = H_JR_VARIANCE # Variance of the Rayleigh distribution for the Rayleigh fading from jammer to receiver

        self.type = type

        self.agent = jammerPPOAgent(device=device)

    def choose_action(self, observation):
        if self.type == "PPO":
            action, logprob, value = self.agent.choose_action(observation)
            return action.unsqueeze(0), logprob.unsqueeze(0), value
        else:
            return self.agent.choose_action(observation)
    
    # Used for the PPO smart jammer
    def store_in_memory(self, state, action, logprob, reward, value):
        self.agent.store_in_memory(state, action, logprob, reward, value)

    # Used for the PPO smart jammer
    def clear_memory(self):
        self.agent.clear_memory()

    def update(self):
        self.agent.update()