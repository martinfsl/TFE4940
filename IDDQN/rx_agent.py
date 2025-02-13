import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

from constants import *

#################################################################################
### Defining classes RNNQN and RNNQN-agent
#################################################################################

class rxRNNQN(nn.Module):
    def __init__(self):
        super(rxRNNQN, self).__init__()

        self.input_size = NUM_SENSE_CHANNELS + 1
        self.hidden_size = 128
        self.num_layers = 2
        self.num_channels = NUM_CHANNELS

        # Defining the RNN layer (using LSTM)
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

class rxRNNQNAgent:
    def __init__(self, gamma = GAMMA, epsilon = EPSILON):
        self.gamma = gamma
        self.epsilon = epsilon

        self.learning_rate = LEARNING_RATE

        # Power
        self.power = RX_USER_TRANSMIT_POWER
        self.h_rt_variance = H_TR_VARIANCE # Variance of the Rayleigh distribution for the Rayleigh fading from transmitter to receiver
        self.h_rj_variance = H_JR_VARIANCE # Variance of the Rayleigh distribution for the Rayleigh fading from transmitter to jammer

        # Parameters for the RNN network A
        self.batch_size = DQN_BATCH_SIZE
        self.memory = []

        self.q_network = rxRNNQN()
        self.target_network = rxRNNQN()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr = self.learning_rate)

    def get_transmit_power(self, direction):
        if CONSIDER_FADING:
            if direction == "transmitter":
                h = np.abs(np.random.normal(0, self.h_rt_variance, 1) + 1j*np.random.normal(0, self.h_rt_variance, 1))
            elif direction == "jammer":
                h = np.abs(np.random.normal(0, self.h_rj_variance, 1) + 1j*np.random.normal(0, self.h_rj_variance, 1))
            received_power = (h*self.power)[0]
        else:
            received_power = self.power

        return received_power

    def get_observation(self, state, action):
        # Create an array of zeros for the observation that is the length of NUM_SENSE_CHANNELS + 1
        # The first elements are the state values centered around the action channel and the last element is the action channel
        observation = np.zeros(NUM_SENSE_CHANNELS + 1)
        half_sense_channels = NUM_SENSE_CHANNELS // 2

        for i in range(-half_sense_channels, half_sense_channels + 1):
            index = (action + i) % len(state)
            observation[i + half_sense_channels] = state[index]

        observation[-1] = action
        return observation

    def choose_action(self, observation):
        if random.uniform(0, 1) < self.epsilon:
            if self.epsilon > EPSILON_MIN:
                self.epsilon *= EPSILON_REDUCTION
            return random.choice(range(NUM_CHANNELS))
        else:
            observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0)
            hidden = self.q_network.init_hidden(1)
            q_values, _ = self.q_network(observation, hidden)
            return torch.argmax(q_values).item()

    # Functions for the RNN network A
    def update_target_q_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def store_experience_in(self, state, action, reward, next_state):
        if len(self.memory) >= MAXIMUM_MEMORY_SIZE:
            self.memory.pop(0)

        self.memory.append((state, action, reward, next_state))

    def replay(self):
        if len(self.memory) >= MEMORY_SIZE_BEFORE_TRAINING:
            # batch = random.sample(self.memory, self.batch_size)

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

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()