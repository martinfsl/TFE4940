import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

from constants import *

#################################################################################
### Defining class for the RNN for the smart jammer
#################################################################################

class jammerRNNQN(nn.Module):
    def __init__(self):
        super(jammerRNNQN, self).__init__()

        # self.input_size = NUM_JAMMER_SENSE_CHANNELS + 1
        self.input_size = NUM_CHANNELS + 1
        self.hidden_size = 128
        self.num_layers = 2
        self.num_channels = NUM_CHANNELS

        # Defining the RNN layer (using GRU)
        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)

        # Fully connected layer to output Q-values for each action
        self.fc = nn.Linear(self.hidden_size, self.num_channels)

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    
    def forward(self, x, hidden=None):
        # Pass the input through the RNN
        out, hidden = self.rnn(x.unsqueeze(0), hidden)

        # Use the last hidden state
        out = out[:, -1, :]

        # Pass the last hidden state through the fully connected layer
        q_values = self.fc(out)

        return q_values, hidden
    
class jammerRNNQNAgent:
    def __init__(self, gamma = GAMMA, epsilon = EPSILON):
        self.gamma = 0.99
        self.epsilon = 0.4

        self.learning_rate = 0.0075
        
        # Parameters for the RNN network
        self.batch_size = DQN_BATCH_SIZE
        self.memory = []

        self.q_network = jammerRNNQN()
        self.target_network = jammerRNNQN()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def update_target_q_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def store_experience_in(self, state, action, reward, next_state):
        if len(self.memory) >= MAXIMUM_MEMORY_SIZE:
            self.memory.pop(0)

        self.memory.append((state, action, reward, next_state))
    
    def replay(self):
        if len(self.memory) < MEMORY_SIZE_BEFORE_TRAINING:
            return

        # Selecting a random index and getting the batch from that index onwards
        index = random.randint(0, len(self.memory)-self.batch_size)
        batch = self.memory[index:index+self.batch_size]

        total_loss = 0
        for state, action, reward, next_state in batch:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            action = torch.tensor(action, dtype=torch.long).unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float).unsqueeze(0)
            next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)

            hidden = self.q_network.init_hidden(1)
            q_values, _ = self.q_network(state, hidden)
            q_value = q_values[0][action]

            hidden_next = self.q_network.init_hidden(1)
            q_values_next, _ = self.q_network(next_state, hidden_next)
            a_argmax = torch.argmax(q_values_next).item()

            target = reward + self.gamma*self.target_network(next_state, hidden_next)[0].detach().numpy()[0][a_argmax]

            loss = nn.MSELoss()(q_value, target)
            total_loss += loss

#################################################################################
### Defining class for a smart jammer
#################################################################################

# Class that represents a smart jammer in the environment
# The jammer uses a DQN to learn the optimal policy, which is used to predict the frequency band chosen for the Tx-Rx link
class SmartJammer:
    def __init__(self):
        self.power = JAMMER_TRANSMIT_POWER
        self.h_jt_variance = H_JT_VARIANCE # Variance of the Rayleigh distribution for the Rayleigh fading from jammer to transmitter
        self.h_jr_variance = H_JR_VARIANCE # Variance of the Rayleigh distribution for the Rayleigh fading from jammer to receiver

        self.agent = jammerRNNQNAgent()

    def choose_action(self, observation):
        if random.uniform(0, 1) < self.agent.epsilon:
            if self.agent.epsilon > EPSILON_MIN_JAMMER:
                self.agent.epsilon *= EPSILON_REDUCTION_JAMMER
            return random.choice(range(NUM_CHANNELS))
        else:
            observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0)
            hidden = self.agent.q_network.init_hidden(1)
            q_values, _ = self.agent.q_network(observation, hidden)
            return torch.argmax(q_values).item()