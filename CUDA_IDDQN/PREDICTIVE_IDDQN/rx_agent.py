import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

import copy

from constants import *

#################################################################################
### Defining classes for the model predicting Tx's action at the Rx
#################################################################################

# Create a class for the neural network that predicts the Tx's action at the Rx
# The input to the neural network is the current observation at the Rx and the output is the predicted Tx's action
# The neural network is made up off 2 fully connected layers with ReLU activation functions
# The neural network has a dropout rate of 30% to prevent overfitting
class rxPredNN(nn.Module):
    def __init__(self):
        super(rxPredNN, self).__init__()

        self.input_size = NUM_SENSE_CHANNELS + 1
        self.hidden_size1 = 128
        self.hidden_size2 = 64
        self.output_size = NUM_CHANNELS

        # Defining the fully connected layers
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
    
# Create a class for the agent that uses the neural network to predict the Tx's action at the Rx
# The agent has a memory to store experiences
# The agent uses the Adam optimizer to train the neural network
# The agent uses the cross entropy loss function to train the neural network
# The output of the neural network is the predicted Tx's action using a softmax function
class rxPredAgent:
    def __init__(self, device = "cpu"):
        self.learning_rate = 0.01

        # Parameters for the neural network
        self.batch_size = 16
        self.memory = []
        self.maximum_memory_size = 100

        self.device = device

        self.pred_network = rxPredNN()
        self.pred_network.to(self.device)
        self.optimizer = optim.Adam(self.pred_network.parameters(), lr = self.learning_rate)

    # Function to store experiences in the memory
    def store_experience_in(self, state, action):
        if len(self.memory) >= self.maximum_memory_size:
            self.memory.pop(0)

        self.memory.append((state, action))

    # Function to train the neural network
    def train(self):
        if len(self.memory) >= self.batch_size:
            batch = random.sample(self.memory, self.batch_size)

            total_loss = 0
            for state, action in batch:
                state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
                action = torch.tensor(action, dtype=torch.long, device=self.device).unsqueeze(0)

                pred = self.pred_network(state)
                loss = nn.CrossEntropyLoss()(pred, action)
                total_loss += loss
                
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

    # Function to predict the Tx's action at the Rx
    def predict_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float, device=self.device).unsqueeze(0)
        pred = self.pred_network(observation)
        # return torch.argmax(pred).item()
        # return pred
        return nn.Softmax(dim=1)(pred)

#################################################################################
### Defining classes RNNQN and RNNQN-agent
#################################################################################

class rxRNNQN(nn.Module):
    def __init__(self):
        super(rxRNNQN, self).__init__()

        # self.input_size = NUM_SENSE_CHANNELS + 1
        self.input_size = NUM_SENSE_CHANNELS + 1 + NUM_CHANNELS
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
    def __init__(self, gamma = GAMMA, epsilon = EPSILON, device = "cpu"):
        self.gamma = gamma
        self.epsilon = epsilon

        self.learning_rate = LEARNING_RATE

        # Power
        self.power = RX_USER_TRANSMIT_POWER
        self.h_rt_variance = H_TR_VARIANCE # Variance of the Rayleigh distribution for the Rayleigh fading from transmitter to receiver
        self.h_rj_variance = H_JR_VARIANCE # Variance of the Rayleigh distribution for the Rayleigh fading from transmitter to jammer

        # For CUDA
        self.device = device

        # Parameters for the RNN network A
        self.batch_size = DQN_BATCH_SIZE
        self.memory = []

        self.q_network = rxRNNQN()
        self.q_network.to(self.device) # Moving Q-network to device (CUDA or CPU)
        self.target_network = rxRNNQN()
        self.target_network.to(self.device) # Moving target network to device (CUDA or CPU)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr = self.learning_rate)

        # Predictive network
        self.pred_agent = rxPredAgent(device=self.device)

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
        if NUM_SENSE_CHANNELS < NUM_CHANNELS:
            # Create an array of zeros for the observation that is the length of NUM_SENSE_CHANNELS + 1
            # The first elements are the state values centered around the action channel and the last element is the action channel
            observation = np.zeros(NUM_SENSE_CHANNELS + 1)
            half_sense_channels = NUM_SENSE_CHANNELS // 2

            for i in range(-half_sense_channels, half_sense_channels + 1):
                index = (action + i) % len(state)
                observation[i + half_sense_channels] = state[index]

            observation[-1] = action
        else:
            observation = copy.deepcopy(state)
            observation.append(action)
        return observation

    def choose_action(self, observation):
        if random.uniform(0, 1) < self.epsilon:
            if self.epsilon > EPSILON_MIN:
                self.epsilon *= EPSILON_REDUCTION
            # return random.choice(range(NUM_CHANNELS))
            # Extract one main action and NUM_EXTRA_ACTIONS extra actions
            # The extra actions should all be unique and not the same as the main action
            main_action = random.choice(range(NUM_CHANNELS))
            extra_actions = random.sample(range(NUM_CHANNELS), NUM_EXTRA_ACTIONS)
            while main_action in extra_actions:
                extra_actions = random.sample(range(NUM_CHANNELS), NUM_EXTRA_ACTIONS)
            return [main_action, extra_actions]
        else:
            # Add the predicted action of Tx to the observation
            pred_action = self.pred_agent.predict_action(observation).to("cpu").detach().numpy()[0]
            observation = np.append(observation, pred_action)

            observation = torch.tensor(observation, dtype=torch.float, device=self.device).unsqueeze(0)
            hidden = self.q_network.init_hidden(1).to(self.device)
            q_values, _ = self.q_network(observation, hidden)

            q_values = q_values.to("cpu") # Add Q-values to CPU to use numpy arrays
            q_values = q_values.detach().numpy()[0]
            main_action = np.argmax(q_values)
            extra_actions = np.argsort(q_values)[-NUM_EXTRA_ACTIONS-1:-1]
            while main_action in extra_actions:
                extra_actions = np.argsort(q_values)[-NUM_EXTRA_ACTIONS-1:-1]
            return [main_action, extra_actions.tolist()]

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
                pred_action = self.pred_agent.predict_action(state).to("cpu").detach().numpy()[0]
                state = np.append(state, pred_action)
                state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)

                action = torch.tensor(action, dtype=torch.long, device=self.device).unsqueeze(0)
                reward = torch.tensor(reward, dtype=torch.float, device=self.device).unsqueeze(0)

                pred_next_action = self.pred_agent.predict_action(next_state).to("cpu").detach().numpy()[0]
                next_state = np.append(next_state, pred_next_action)
                next_state = torch.tensor(next_state, dtype=torch.float, device=self.device).unsqueeze(0)

                hidden = self.q_network.init_hidden(1).to(self.device)
                q_values, _ = self.q_network(state, hidden)
                q_value = q_values[0][action]

                hidden_next = self.q_network.init_hidden(1).to(self.device)
                q_values_next, _ = self.q_network(next_state, hidden_next)
                a_argmax = torch.argmax(q_values_next).item()

                target = reward + self.gamma*self.target_network(next_state, hidden_next)[0].to("cpu").detach().numpy()[0][a_argmax]

                loss = nn.MSELoss()(q_value, target)
                total_loss += loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()