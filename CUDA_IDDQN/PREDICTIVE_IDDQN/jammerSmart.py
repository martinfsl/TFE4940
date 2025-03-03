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
    def __init__(self, gamma = GAMMA, epsilon = EPSILON, device = "cpu"):
        self.gamma = 0.99
        self.epsilon = 0.4

        self.learning_rate = 0.0075
        
        # Parameters for the RNN network
        self.batch_size = DQN_BATCH_SIZE
        self.memory = []

        self.device = device

        self.q_network = jammerRNNQN()
        self.q_network.to(self.device)
        self.target_network = jammerRNNQN()
        self.target_network.to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def choose_action(self, observation):
        if random.uniform(0, 1) < self.epsilon:
            if self.epsilon > EPSILON_MIN_JAMMER:
                self.epsilon *= EPSILON_REDUCTION_JAMMER
            return random.choice(range(NUM_CHANNELS))
        else:
            observation = observation.clone().to(self.device).unsqueeze(0)
            hidden = self.q_network.init_hidden(1).to(self.device)
            q_values, _ = self.q_network(observation, hidden)
            return torch.argmax(q_values).item()

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
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            next_state = next_state.unsqueeze(0)

            hidden = self.q_network.init_hidden(1).to(self.device)
            q_values, _ = self.q_network(state, hidden)
            q_value = q_values[0][action]

            hidden_next = self.q_network.init_hidden(1).to(self.device)
            q_values_next, _ = self.q_network(next_state, hidden_next)
            a_argmax = torch.argmax(q_values_next).item()

            target = reward + self.gamma*self.target_network(next_state, hidden_next)[0][0][a_argmax].detach()

            loss = nn.MSELoss()(q_value, target)
            total_loss += loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

#################################################################################
### Defining class for the DDQN for the smart jammer
#################################################################################

class jammerFNNDDQN(nn.Module):
    def __init__(self):
        super(jammerFNNDDQN, self).__init__()

        self.input_size = NUM_CHANNELS + 1
        self.hidden_size = 128
        self.num_channels = NUM_CHANNELS

        # Defining the fully connected layers
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(self.hidden_size, self.num_channels)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
    
class jammerFNNDDQNAgent:
    def __init__(self, device = "cpu"):
        self.gamma = 0.35
        self.epsilon = 0.4

        self.learning_rate = 0.0075

        # Parameters for the FNN network
        self.batch_size = DQN_BATCH_SIZE
        self.memory = []

        self.device = device

        self.q_network = jammerFNNDDQN()
        self.q_network.to(self.device)
        self.target_network = jammerFNNDDQN()
        self.target_network.to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def choose_action(self, observation):
        if random.uniform(0, 1) < self.epsilon:
            if self.epsilon > EPSILON_MIN_JAMMER:
                self.epsilon *= EPSILON_REDUCTION_JAMMER
            return torch.tensor(random.choice(range(NUM_CHANNELS)), device=self.device)
            # return torch.tensor(random.choice(range(NUM_CHANNELS)), device=self.device)
        else:
            # observation = torch.tensor(observation, dtype=torch.float, device=self.device).unsqueeze(0)
            q_values = self.q_network(observation)
            return torch.argmax(q_values).item()

    def update_target_q_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def store_experience_in(self, state, action, reward, next_state):
        if len(self.memory) >= MAXIMUM_MEMORY_SIZE:
            self.memory.pop(0)

        self.memory.append((state, action, reward, next_state))

    def replay(self):
        if len(self.memory) >= MEMORY_SIZE_BEFORE_TRAINING:
            batch = random.sample(self.memory, self.batch_size)

            total_loss = 0
            for state, action, reward, next_state in batch:
                state = state.unsqueeze(0)
                action = action.unsqueeze(0)
                reward = reward.unsqueeze(0)
                next_state = next_state.unsqueeze(0)

                q_values = self.q_network(state)
                q_value = q_values[0][action]

                q_values_next = self.q_network(next_state)
                a_argmax = torch.argmax(q_values_next).item()

                target = reward + self.gamma*self.target_network(next_state)[0][0][a_argmax].detach()

                loss = nn.MSELoss()(q_value, target)
                total_loss += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

#################################################################################
### Defining class for a smart jammer
#################################################################################

# Class that represents a smart jammer in the environment
# The jammer uses a DQN to learn the optimal policy, which is used to predict the frequency band chosen for the Tx-Rx link
class SmartJammer:
    def __init__(self, type = "RNN", device = "cpu"):
        self.power = JAMMER_TRANSMIT_POWER
        self.h_jt_variance = H_JT_VARIANCE # Variance of the Rayleigh distribution for the Rayleigh fading from jammer to transmitter
        self.h_jr_variance = H_JR_VARIANCE # Variance of the Rayleigh distribution for the Rayleigh fading from jammer to receiver

        if type == "RNN":
            self.agent = jammerRNNQNAgent(device=device)
        elif type == "FNN":
            self.agent = jammerFNNDDQNAgent(device=device)

    def choose_action(self, observation):
        return self.agent.choose_action(observation)
        
    def update_target_q_network(self):
        self.agent.update_target_q_network()
    
    def store_experience_in(self, state, action, reward, next_state):
        self.agent.store_experience_in(state, action, reward, next_state)

    def replay(self):
        self.agent.replay()