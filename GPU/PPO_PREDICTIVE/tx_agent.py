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

class txPPO(nn.Module):
    def __init__(self, device="cpu"):
        super(txPPO, self).__init__()
        self.device = device

        # Input includes original observation + predicted action
        self.input_size = NUM_SENSE_CHANNELS + 1 + NUM_CHANNELS
        self.hidden_size = 128
        self.num_layers = 2
        self.num_channels = NUM_CHANNELS

        # Using an RNN (GRU) to process the observation sequence.
        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, device=self.device)
        
        # Two output heads: one for policy logits and one for value prediction.
        self.fc_policy = nn.Linear(self.hidden_size, self.num_channels, device=self.device)
        self.fc_value = nn.Linear(self.hidden_size, 1, device=self.device)

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)

    def forward(self, x, hidden=None):
        # x is assumed to be of shape [batch, input_size]
        out, hidden = self.rnn(x.unsqueeze(0), hidden)
        out = out[:, -1, :]  # Use final output
        policy_logits = self.fc_policy(out)
        value = self.fc_value(out)
        return policy_logits, value, hidden

class txPPOAgent:
    def __init__(self, gamma=GAMMA, device="cpu"):
        self.gamma = gamma
        self.learning_rate = LEARNING_RATE

        self.device = device

        # PPO on-policy storage (use lists to store one episode/trajectory)
        self.memory_state = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_logprob = []
        self.memory_value = []

        # Policy network (actor-critic). No target network is needed.
        self.policy_network = txPPO(device=self.device)
        self.policy_network.to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

        # Predictive network remains unchanged.
        self.pred_agent = txPredNNAgent(device=self.device)

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

    def choose_action(self, observation):
        # Append the predicted action from the predictive network.
        with torch.no_grad():
            pred_action = self.pred_agent.predict_action(observation)
        # Concatenate original observation and predicted action.
        obs_input = torch.cat((observation, pred_action)).unsqueeze(0)  # shape: [1, input_size]
        hidden = self.policy_network.init_hidden(1)
        policy_logits, value, _ = self.policy_network(obs_input, hidden)
        policy_probs = nn.Softmax(dim=-1)(policy_logits)
        dist = torch.distributions.Categorical(policy_probs)
        action = dist.sample()

        # Store state, chosen action, log-probability and predicted value for later PPO update.
        self.memory_state.append(obs_input)
        self.memory_action.append(action)
        self.memory_logprob.append(dist.log_prob(action))
        self.memory_value.append(value)

        return action

    def store_reward(self, reward):
        self.memory_reward.append(reward)

    def compute_returns_and_advantages(self):
        # Compute discounted returns.
        returns = []
        R = 0
        for r in reversed(self.memory_reward):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=self.device)

        # Convert stored predicted values (critic estimates) to a tensor.
        values = torch.cat(self.memory_value).squeeze()

        # Compute advantages = (return - value) for simplicity.
        advantages = returns - values
        # Optionally normalize advantages.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        return returns, advantages

    def update_policy(self, epochs=4, clip_epsilon=0.2):
        returns, advantages = self.compute_returns_and_advantages()
        old_logprobs = torch.stack(self.memory_logprob).detach()
        # Stack states and actions.
        states = torch.cat(self.memory_state, dim=0)  # shape: [batch, input_size]
        actions = torch.tensor(self.memory_action, device=self.device)

        for _ in range(epochs):
            hidden = self.policy_network.init_hidden(states.size(0))
            policy_logits, values, _ = self.policy_network(states, hidden)
            policy_probs = nn.Softmax(dim=-1)(policy_logits)
            dist = torch.distributions.Categorical(policy_probs)
            new_logprobs = dist.log_prob(actions)
            ratio = torch.exp(new_logprobs - old_logprobs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear memory after updating.
        self.memory_state = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_logprob = []
        self.memory_value = []
