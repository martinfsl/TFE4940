import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

import copy

from constants import *
from fh_pattern import FH_Pattern

#################################################################################
### Defining classes for the model choosing which additional channels the Rx can sense
#################################################################################

class rxSenseNN(nn.Module):
    def __init__(self):
        super(rxSenseNN, self).__init__()

        # self.input_size = NUM_SENSE_CHANNELS + 1
        self.input_size = STATE_SPACE_SIZE
        # self.input_size = STATE_SPACE_SIZE + NUM_SEEDS
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
    
class rxSenseNNAgent:
    def __init__(self, device = "cpu"):
        self.learning_rate = 0.005

        # Parameters for the neural network
        self.batch_size = 16
        self.maximum_memory_size = 100

        self.device = device

        # self.memory_state = torch.empty((0, NUM_SENSE_CHANNELS + 1), device=self.device)
        self.memory_state = torch.empty((0, STATE_SPACE_SIZE), device=self.device)
        # self.memory_state = torch.empty((0, STATE_SPACE_SIZE + NUM_SEEDS), device=self.device)
        self.memory_action = torch.empty((0, 1), device=self.device)

        self.sense_network = rxSenseNN()
        self.sense_network.to(self.device)
        self.optimizer = optim.Adam(self.sense_network.parameters(), lr=self.learning_rate)

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

            pred = self.sense_network(batch_state)
            loss = nn.CrossEntropyLoss()(pred, batch_action.long().squeeze())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_subobservation(self, state, action):
        if NUM_SENSE_CHANNELS < NUM_CHANNELS:
            observation = torch.zeros(NUM_SENSE_CHANNELS + 1, device=self.device)
            half_sense_channels = NUM_SENSE_CHANNELS // 2
            for i in range(-half_sense_channels, half_sense_channels + 1):
                index = (action + i) % len(state)
                observation[i + half_sense_channels] = state[index.long()]
            observation[-1] = action
        else:
            observation = torch.cat((state, action), dim=0)

        return observation

    def choose_receive_and_sensing_channels(self, observation):
        with torch.no_grad():
            pred = self.sense_network(observation)
        
        sorted_indices = torch.argsort(pred, descending=True)
        receive_channels = sorted_indices[:NUM_EXTRA_RECEIVE]
        extra_actions = sorted_indices[NUM_EXTRA_RECEIVE:NUM_EXTRA_RECEIVE+NUM_EXTRA_ACTIONS]
        
        return receive_channels, extra_actions

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

        self.input_size = STATE_SPACE_SIZE
        # self.input_size = NUM_SENSE_CHANNELS + 1
        self.hidden_size1 = 128
        self.hidden_size2 = 64
        # self.output_size = NUM_CHANNELS
        self.output_size = NUM_SEEDS

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
class rxPredNNAgent:
    def __init__(self, device = "cpu"):
        # self.learning_rate = 0.01
        self.learning_rate = 0.001

        # Parameters for the neural network
        self.batch_size = 16
        self.maximum_memory_size = 100

        self.device = device

        self.memory_state = torch.empty((0, STATE_SPACE_SIZE), device=self.device)
        self.memory_action = torch.empty((0, 1), device=self.device)

        self.pred_network = rxPredNN()
        self.pred_network.to(self.device)
        self.optimizer = optim.Adam(self.pred_network.parameters(), lr=self.learning_rate)

    # Function to store experiences in the memory
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

    # Function to predict the Tx's action at the Rx
    def predict_action(self, observation):
        with torch.no_grad():
            pred = self.pred_network(observation)
        return nn.Softmax(dim=0)(pred)

#################################################################################
### Defining classes RNNQN and RNNQN-agent
#################################################################################

class rxPPOActor(nn.Module):
    def __init__(self, device = "cpu"):
        super(rxPPOActor, self).__init__()

        # self.input_size = STATE_SPACE_SIZE + NUM_SEEDS
        self.input_size = STATE_SPACE_SIZE
        # self.input_size = STATE_SPACE_SIZE + NUM_CHANNELS
        # self.input_size = NUM_SENSE_CHANNELS + 1 + NUM_CHANNELS
        # self.input_size = NUM_SENSE_CHANNELS + 1
        self.hidden_size1 = 128
        self.hidden_size2 = 64
        self.output_size = NUM_SEEDS
        # self.output_size = NUM_CHANNELS

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
    
class rxPPOCritic(nn.Module):
    def __init__(self, device = "cpu"):
        super(rxPPOCritic, self).__init__()

        # self.input_size = STATE_SPACE_SIZE + NUM_SEEDS
        self.input_size = STATE_SPACE_SIZE
        # self.input_size = STATE_SPACE_SIZE + NUM_CHANNELS
        # self.input_size = NUM_SENSE_CHANNELS + 1 + NUM_CHANNELS
        # self.input_size = NUM_SENSE_CHANNELS + 1
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

class rxPPOAgent:
    def __init__(self, gamma = GAMMA, learning_rate = LEARNING_RATE, 
                lambda_param = LAMBDA, epsilon_clip = EPSILON_CLIP, 
                k = K, m = M, c1 = C1, c2 = C2,
                device = "cpu"):
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.LAMBDA = lambda_param
        self.epsilon_clip = epsilon_clip

        self.k = k
        self.m = m
        self.c1 = c1
        self.c2 = c2

        # Power
        self.power = RX_USER_TRANSMIT_POWER
        self.h_rt_variance = H_TR_VARIANCE # Variance of the Rayleigh distribution for the Rayleigh fading from transmitter to receiver
        self.h_rj_variance = H_JR_VARIANCE # Variance of the Rayleigh distribution for the Rayleigh fading from transmitter to jammer

        # For CUDA
        self.device = device

        # PPO on-policy storage
        # self.memory_state = torch.empty((0, STATE_SPACE_SIZE + NUM_SEEDS), device=self.device)
        self.memory_state = torch.empty((0, STATE_SPACE_SIZE), device=self.device)
        # self.memory_state = torch.empty((0, STATE_SPACE_SIZE+NUM_CHANNELS), device=self.device)
        self.memory_action = torch.empty((0, 1), device=self.device)
        self.memory_logprob = torch.empty((0, 1), device=self.device)
        self.memory_reward = torch.empty((0, 1), device=self.device)
        self.memory_value = torch.empty((0, 1), device=self.device)

        self.previous_seeds = torch.empty((0, 1), device=self.device)

        # Policy network (Actor network)
        self.actor_network = rxPPOActor()
        self.actor_network.to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.learning_rate)

        self.actor_network_old = copy.deepcopy(self.actor_network).to(self.device)

        # Value network (Critic network)
        self.critic_network = rxPPOCritic()
        self.critic_network.to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.learning_rate)

        # Predictive network remains unchanged
        self.pred_agent = rxPredNNAgent(device=self.device)

        # Sensing network
        self.sensing_agent = rxSenseNNAgent(device=self.device)

        # Logging actor and critic losses
        self.actor_losses = torch.tensor([], device=self.device)
        self.critic_losses = torch.tensor([], device=self.device)

        # Logging the channels selected
        self.channels_selected = torch.tensor([], device=self.device)

        # FH pattern
        self.fh = FH_Pattern(device = self.device)
        self.fh_seeds_used = torch.tensor([], device=self.device)

    def add_previous_seed(self, seed):
        self.previous_seeds = torch.cat((self.previous_seeds, seed.unsqueeze(0)), dim=0)

        if self.previous_seeds.size(0) > NUM_PREV_PATTERNS:
            self.previous_seeds = self.previous_seeds[1:]

    def clear_memory(self):
        # self.memory_state = torch.empty((0, STATE_SPACE_SIZE + NUM_SEEDS), device=self.device)
        self.memory_state = torch.empty((0, STATE_SPACE_SIZE), device=self.device)
        # self.memory_state = torch.empty((0, STATE_SPACE_SIZE+NUM_CHANNELS), device=self.device)
        self.memory_action = torch.empty((0, 1), device=self.device)
        self.memory_logprob = torch.empty((0, 1), device=self.device)
        self.memory_reward = torch.empty((0, 1), device=self.device)
        self.memory_value = torch.empty((0, 1), device=self.device)

    def store_in_memory(self, state, action, logprob, reward, value):
        self.memory_state = torch.cat((self.memory_state, state.unsqueeze(0)), dim=0)
        self.memory_action = torch.cat((self.memory_action, action.unsqueeze(0)), dim=0)
        self.memory_logprob = torch.cat((self.memory_logprob, logprob.unsqueeze(0)), dim=0)
        self.memory_reward = torch.cat((self.memory_reward, reward.unsqueeze(0)), dim=0)
        self.memory_value = torch.cat((self.memory_value, value.unsqueeze(0)), dim=0)

    def get_transmit_power(self, direction):
        if CONSIDER_FADING:
            if direction == "transmitter":
                h_real = torch.normal(mean=0.0, std=self.h_rt_variance, size=(1,), device=self.device)
                h_imag = torch.normal(mean=0.0, std=self.h_rt_variance, size=(1,), device=self.device)
                h = torch.abs(torch.complex(h_real, h_imag))
            elif direction == "jammer":
                h_real = torch.normal(mean=0.0, std=self.h_rj_variance, size=(1,), device=self.device)
                h_imag = torch.normal(mean=0.0, std=self.h_rj_variance, size=(1,), device=self.device)
                h = torch.abs(torch.complex(h_real, h_imag))
            received_power = (h*self.power)[0]
        else:
            received_power = self.power

        return torch.tensor(received_power, device=self.device)

    def get_observation(self, state, action):
        observation_pattern = torch.zeros(STATE_SPACE_SIZE, device=self.device)

        for i in range(NUM_HOPS_PER_PATTERN):
            # Same as before: create observation by concatenating state and action data.
            if NUM_SENSE_CHANNELS < NUM_CHANNELS:
                observation = torch.zeros(NUM_SENSE_CHANNELS + 1, device=self.device)
                half_sense_channels = NUM_SENSE_CHANNELS // 2
                for j in range(-half_sense_channels, half_sense_channels + 1):
                    index = (action[i] + j) % len(state[i])
                    observation[j + half_sense_channels] = state[i][index.long()]
                observation[-1] = action[i]
            else:
                observation = torch.cat((state[i], action[i]), dim=0)
        
            observation_pattern[i*(NUM_SENSE_CHANNELS+1):(i+1)*(NUM_SENSE_CHANNELS+1)] = observation

        return observation_pattern
    
    def concat_predicted_action(self, observation):
        # Get the predicted action from the predictive network
        with torch.no_grad():
            pred_action = self.pred_agent.predict_action(observation)
        # Concatenate original observation and predicted action.
        observation = torch.concat((observation, pred_action))

        return observation

    def get_value(self, observation):
        return self.critic_network(observation)

    # # Returning the most probable action and the NUM_EXTRA_ACTIONS next most probable actions
    # def choose_action(self, observation):
    #     with torch.no_grad():
    #         policy = nn.Softmax(dim=0)(self.actor_network(observation))
    #         values = self.get_value(observation)

    #     # Extract the most probable action as main action and NUM_EXTRA_ACTIONS additional actions which are the next most probable actions
    #     main_action = torch.argmax(policy)
    #     additional_actions = torch.argsort(policy, descending=True)[1:NUM_EXTRA_ACTIONS+1]
    #     actions = torch.cat((torch.tensor([main_action], device=self.device), additional_actions))

    #     action_logprob = torch.log(torch.gather(policy, 0, main_action.unsqueeze(0)))
    #     self.fh_patterns_used = torch.cat((self.fh_patterns_used, main_action.unsqueeze(0)))

    #     return actions, action_logprob, values

    # Only returning the most probable action
    def choose_action(self, observation):
        with torch.no_grad():
            policy = nn.Softmax(dim=0)(self.actor_network(observation))
            values = self.get_value(observation)

        action = torch.argmax(policy)
        action_logprob = torch.log(torch.gather(policy, 0, action.unsqueeze(0)))
        self.fh_seeds_used = torch.cat((self.fh_seeds_used, action.unsqueeze(0)))

        return action.unsqueeze(0), action_logprob, values

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
        
        # Normalize the advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        return advantages
    
    def update(self):
        returns = self.compute_returns()
        advantages = self.compute_advantages(returns, self.memory_value)

        values = self.get_value(self.memory_state)

        old_logits = self.actor_network_old(self.memory_state).detach()
        old_logprobs = torch.log(torch.gather(nn.Softmax(dim=1)(old_logits), 1, self.memory_action.long()))

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

            entropy = -(new_policy*torch.log(new_policy + 1e-8)).sum(dim=1).mean()

            actor_loss = -torch.min(surr1, surr2).mean() - self.c2*entropy

            batch_value = self.get_value(batch_state)
            critic_loss = nn.MSELoss()(batch_value, batch_return)
            
            total_loss = actor_loss + self.c1*critic_loss

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # actor_loss.backward()
            # critic_loss.backward()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
        self.actor_network_old.load_state_dict(self.actor_network.state_dict()) # Update the weights of the old network to the current network after each update

        self.clear_memory()

        self.actor_losses = torch.cat((self.actor_losses, actor_loss.unsqueeze(0)))
        self.critic_losses = torch.cat((self.critic_losses, critic_loss.unsqueeze(0)))