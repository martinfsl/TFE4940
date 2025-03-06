import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

import copy

from constants import *

#################################################################################
### Defining classes for the model predicting Rx's action at the Tx
#################################################################################

# Create a class for the neural network that predicts the Rx's action at the Tx
# The input to the neural network is the current observation at the Tx and the output is the predicted Rx's action
# The neural network is made up off 2 fully connected layers with ReLU activation functions
# The neural network has a dropout rate of 30%
class txPredNN(nn.Module):
    def __init__(self):
        super(txPredNN, self).__init__()

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
    
# Create a class for the agent that uses the neural network to predict the Rx's action at the Tx
# The agent has a memory to store experiences
# The agent uses the Adam optimizer to train the neural network
# The agent uses the cross entropy loss function to train the neural network
# The output of the neural network is the predicted Rx's action using a softmax function
class txPredNNAgent:
    def __init__(self, device = "cpu"):
        self.learning_rate = 0.01

        # Parameters for the neural network
        self.batch_size = 16
        self.maximum_memory_size = 100

        self.device = device

        self.memory_state = torch.empty((0, NUM_SENSE_CHANNELS+1), device=self.device)
        self.memory_action = torch.empty((0, 1), device=self.device)

        self.pred_network = txPredNN()
        self.pred_network.to(self.device)
        self.optimizer = optim.Adam(self.pred_network.parameters(), lr=self.learning_rate)

    def store_experience_in(self, state, action):
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

#################################################################################
### Defining classes RNNQN and RNNQN-agent
#################################################################################

class txPPOActor(nn.Module):
    def __init__(self, device = "cpu"):
        super(txPPOActor, self).__init__()

        # self.input_size = NUM_SENSE_CHANNELS + 1 + NUM_CHANNELS
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
    
class txPPOCritic(nn.Module):
    def __init__(self, device = "cpu"):
        super(txPPOCritic, self).__init__()

        # self.input_size = NUM_SENSE_CHANNELS + 1 + NUM_CHANNELS
        self.input_size = NUM_SENSE_CHANNELS + 1
        self.hidden_size1 = 128
        self.hidden_size2 = 64
        self.output_size = 1

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

class txPPOAgent:
    def __init__(self, gamma=GAMMA, learning_rate = LEARNING_RATE, device = "cpu"):
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.LAMBDA = 0.95
        self.epsilon_clip = 0.2

        self.power = TX_USER_TRANSMIT_POWER
        self.h_tr_variance = H_TR_VARIANCE
        self.h_tj_variance = H_JT_VARIANCE

        self.device = device

        # PPO on-policy storage (use lists to store one episode/trajectory)
        self.memory_state = torch.empty((0, NUM_SENSE_CHANNELS+1), device=self.device)
        self.memory_action = torch.empty((0, 1), device=self.device)
        self.memory_logprob = torch.empty((0, 1), device=self.device)
        self.memory_reward = torch.empty((0, 1), device=self.device)
        self.memory_value = torch.empty((0, 1), device=self.device)

        # Policy network (Actor network)
        self.actor_network = txPPOActor()
        self.actor_network.to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.learning_rate)

        self.actor_network_old = copy.deepcopy(self.actor_network).to(self.device)
        # self.actor_network_old.load_state_dict(self.actor_network.state_dict()) # To update the weights of the old network

        # Value network (Critic network)
        self.critic_network = txPPOCritic()
        self.critic_network.to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.learning_rate)

        # Predictive network remains unchanged.
        self.pred_agent = txPredNNAgent(device=self.device)

    def clear_memory(self):
        self.memory_state = torch.empty((0, NUM_SENSE_CHANNELS+1), device=self.device)
        self.memory_action = torch.empty((0, 1), device=self.device)
        self.memory_logprob = torch.empty((0, 1), device=self.device)
        self.memory_reward = torch.empty((0, 1), device=self.device)
        self.memory_value = torch.empty((0, 1), device=self.device)

    def store_in_memory(self, state, action, logprob, reward, value):
        self.memory_state = torch.cat((self.memory_state, state.unsqueeze(0)), dim=0)
        self.memory_action = torch.cat((self.memory_action, action.unsqueeze(0)), dim=0)
        self.memory_logprob = torch.cat((self.memory_logprob, logprob.unsqueeze(0).unsqueeze(0)), dim=0)
        self.memory_reward = torch.cat((self.memory_reward, reward.unsqueeze(0)), dim=0)
        self.memory_value = torch.cat((self.memory_value, value.unsqueeze(0)), dim=0)

    def get_transmit_power(self, direction):
        if CONSIDER_FADING:
            if direction == "receiver":
                h_real = torch.normal(mean=0.0, std=self.h_tr_variance, size=(1,), device=self.device)
                h_imag = torch.normal(mean=0.0, std=self.h_tr_variance, size=(1,), device=self.device)
                h = torch.abs(torch.complex(h_real, h_imag))
            elif direction == "jammer":
                h_real = torch.normal(mean=0.0, std=self.h_tj_variance, size=(1,), device=self.device)
                h_imag = torch.normal(mean=0.0, std=self.h_tj_variance, size=(1,), device=self.device)
                h = torch.abs(torch.complex(h_real, h_imag))
            received_power = (h*self.power)[0]
        else:
            received_power = self.power

        return torch.tensor(received_power, device=self.device)

    def get_observation(self, state, action):
        # Same as before: create observation by concatenating state and action data.
        if NUM_SENSE_CHANNELS < NUM_CHANNELS:
            observation = torch.zeros(NUM_SENSE_CHANNELS + 1, device=self.device)
            half_sense_channels = NUM_SENSE_CHANNELS // 2
            for i in range(-half_sense_channels, half_sense_channels + 1):
                index = (action + i) % len(state)
                observation[i + half_sense_channels] = state[index]
            observation[-1] = action
        else:
            observation = torch.cat((state, action), dim=0)
        return observation

    def get_value(self, observation):
        return self.critic_network(observation)

    def choose_action(self, observation):
        # # Get the predicted action from the predictive network
        # with torch.no_grad():
        #     pred_action = self.pred_agent.predict_action(observation)
        # # Concatenate original observation and predicted action.
        # observation = torch.concat((observation, pred_action)).unsqueeze(0)

        with torch.no_grad():
            policy = nn.Softmax(dim=0)(self.actor_network(observation))
            values = self.get_value(observation)

        # Extract the most probable action as main action and NUM_EXTRA_ACTIONS additional actions which are the next most probable actions
        main_action = torch.argmax(policy)
        additional_actions = torch.argsort(policy, descending=True)[1:NUM_EXTRA_ACTIONS+1]

        actions = torch.cat((torch.tensor([main_action], device=self.device), additional_actions))

        return actions, policy[main_action], values
    
    # Compute the returns for each time step in the trajectory
    def compute_returns(self):
        returns = torch.empty((0, 1), device=self.device)
        R = 0
        for r in reversed(self.memory_reward):
            R = r + self.gamma * R
            returns = torch.cat((R.unsqueeze(0), returns), dim=0)
        return returns
    
    # Compute the advantages for each time step in the trajectory
    def compute_advantages(self, returns, values):
        # # Simple advantage:
        # advantages = returns - values

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
        
        return advantages

    def update(self):
        returns = self.compute_returns()
        advantages = self.compute_advantages(returns, self.memory_value)

        values = self.get_value(self.memory_state)

        old_logits = self.actor_network_old(self.memory_state).detach()
        old_logprobs = torch.log(torch.gather(nn.Softmax(dim=1)(old_logits), 1, self.memory_action.long()))

        new_logits = self.actor_network(self.memory_state)
        new_logprobs = torch.log(torch.gather(nn.Softmax(dim=1)(new_logits), 1, self.memory_action.long()))

        ratio = torch.exp(new_logprobs - old_logprobs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.epsilon_clip, 1+self.epsilon_clip)*advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(values, returns)

        self.actor_network_old.load_state_dict(self.actor_network.state_dict()) # Update the weights of the old network to the current network after each update

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self.clear_memory()