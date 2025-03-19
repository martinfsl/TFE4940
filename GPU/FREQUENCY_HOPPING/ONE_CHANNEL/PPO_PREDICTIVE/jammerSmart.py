import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

import copy

from constants import *

#################################################################################
### Defining class for the RNN for the smart jammer
#################################################################################

class jammerRNNQN(nn.Module):
    def __init__(self):
        super(jammerRNNQN, self).__init__()

        self.input_size = NUM_JAMMER_SENSE_CHANNELS + 1
        # self.input_size = NUM_CHANNELS + 1
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
        
        self.device = device

        # Parameters for the RNN network
        self.batch_size = BATCH_SIZE
        self.memory_state = torch.empty((0, NUM_JAMMER_SENSE_CHANNELS+1), device=self.device)
        # self.memory_state = torch.empty((0, NUM_CHANNELS+1), device=self.device)
        self.memory_action = torch.empty((0, 1), device=self.device)
        self.memory_reward = torch.empty((0, 1), device=self.device)
        self.memory_next_state = torch.empty((0, NUM_JAMMER_SENSE_CHANNELS+1), device=self.device)
        # self.memory_next_state = torch.empty((0, NUM_CHANNELS+1), device=self.device)

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
            observation = observation.unsqueeze(0)
            hidden = self.q_network.init_hidden(1).to(self.device)
            with torch.no_grad():
                q_values, _ = self.q_network(observation, hidden)
            return torch.argmax(q_values).detach()

    def update_target_q_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def store_experience_in(self, state, action, reward, next_state):
        if self.memory_state.size(0) >= MAXIMUM_MEMORY_SIZE:
            self.memory_state = self.memory_state[1:]
            self.memory_action = self.memory_action[1:]
            self.memory_reward = self.memory_reward[1:]
            self.memory_next_state = self.memory_next_state[1:]

        self.memory_state = torch.cat((self.memory_state, state.unsqueeze(0)), dim=0)
        self.memory_action = torch.cat((self.memory_action, action.unsqueeze(0)), dim=0)
        self.memory_reward = torch.cat((self.memory_reward, reward.unsqueeze(0)), dim=0)
        self.memory_next_state = torch.cat((self.memory_next_state, next_state.unsqueeze(0)), dim=0)
    
    def replay(self):
        if self.memory_state.size(0) >= MEMORY_SIZE_BEFORE_TRAINING:
            # Selecting a random index and getting the batch from that index onwards
            index = random.randint(0, self.memory_state.size(0) - self.batch_size)
            batch_state = self.memory_state[index:index + self.batch_size]
            batch_action = self.memory_action[index:index + self.batch_size]
            batch_reward = self.memory_reward[index:index + self.batch_size]
            batch_next_state = self.memory_next_state[index:index + self.batch_size]

            total_loss = 0
            for i in range(self.batch_size):
                state = batch_state[i].unsqueeze(0)
                action = batch_action[i].unsqueeze(0)
                reward = batch_reward[i].unsqueeze(0)
                next_state = batch_next_state[i].unsqueeze(0)

                hidden = self.q_network.init_hidden(1).to(self.device)
                q_values, _ = self.q_network(state, hidden)
                q_value = q_values[0][action.long()]

                hidden_next = self.q_network.init_hidden(1).to(self.device)
                q_values_next, _ = self.q_network(next_state, hidden_next)
                a_argmax = torch.argmax(q_values_next).detach()

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

        self.input_size = NUM_JAMMER_SENSE_CHANNELS + 1
        # self.input_size = NUM_CHANNELS + 1
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

        self.device = device

        # Parameters for the FNN network
        self.batch_size = BATCH_SIZE
        self.memory_state = torch.empty((0, NUM_JAMMER_SENSE_CHANNELS+1), device=self.device)
        # self.memory_state = torch.empty((0, NUM_CHANNELS+1), device=self.device)
        self.memory_action = torch.empty((0, 1), device=self.device)
        self.memory_reward = torch.empty((0, 1), device=self.device)
        self.memory_next_state = torch.empty((0, NUM_CHANNELS+1), device=self.device)

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
            with torch.no_grad():
                q_values = self.q_network(observation)
            return torch.argmax(q_values).detach()

    def update_target_q_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def store_experience_in(self, state, action, reward, next_state):
        if self.memory_state.size(0) >= MAXIMUM_MEMORY_SIZE:
            self.memory_state = self.memory_state[1:]
            self.memory_action = self.memory_action[1:]
            self.memory_reward = self.memory_reward[1:]
            self.memory_next_state = self.memory_next_state[1:]

        self.memory_state = torch.cat((self.memory_state, state.unsqueeze(0)), dim=0)
        self.memory_action = torch.cat((self.memory_action, action.unsqueeze(0)), dim=0)
        self.memory_reward = torch.cat((self.memory_reward, reward.unsqueeze(0)), dim=0)
        self.memory_next_state = torch.cat((self.memory_next_state, next_state.unsqueeze(0)), dim=0)

    def replay(self):
        if self.memory_state.size(0) >= MEMORY_SIZE_BEFORE_TRAINING:
            indices = random.sample(range(self.memory_state.size(0)), self.batch_size)
            batch_state = self.memory_state[indices]
            batch_action = self.memory_action[indices]
            batch_reward = self.memory_reward[indices]
            batch_next_state = self.memory_next_state[indices]

            total_loss = 0
            for i in range(self.batch_size):
                state = batch_state[i].unsqueeze(0)
                action = batch_action[i].unsqueeze(0)
                reward = batch_reward[i].unsqueeze(0)
                next_state = batch_next_state[i].unsqueeze(0)

                q_values = self.q_network(state)
                q_value = q_values[0][action.long()]

                q_values_next = self.q_network(next_state)
                a_argmax = torch.argmax(q_values_next).detach()

                target = reward + self.gamma*self.target_network(next_state)[0][a_argmax].detach()

                loss = nn.MSELoss()(q_value, target)
                total_loss += loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

#################################################################################
### Defining class for the PPO for the smart jammer
#################################################################################

class jammerPPOActor(nn.Module):
    def __init__(self, device = "cpu"):
        super(jammerPPOActor, self).__init__()

        self.input_size = NUM_JAMMER_SENSE_CHANNELS + 1
        # self.input_size = NUM_CHANNELS + 1
        self.hidden_size = 128
        self.output_size = NUM_CHANNELS

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)

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
        self.hidden_size = 128
        self.output_size = 1

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)

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
        self.learning_rate = 0.001
        # self.learning_rate = 0.005

        # self.LAMBDA = lambda_param
        self.LAMBDA = 0.40
        self.epsilon_clip = epsilon_clip

        self.k = k
        self.m = m
        self.c1 = 0.5
        # self.c2 = 0.01
        self.c2 = 0.05

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
    def __init__(self, type = "RNN", device = "cpu"):
        self.power = JAMMER_TRANSMIT_POWER
        self.h_jt_variance = H_JT_VARIANCE # Variance of the Rayleigh distribution for the Rayleigh fading from jammer to transmitter
        self.h_jr_variance = H_JR_VARIANCE # Variance of the Rayleigh distribution for the Rayleigh fading from jammer to receiver

        self.type = type

        if self.type == "RNN":
            self.agent = jammerRNNQNAgent(device=device)
        elif self.type == "FNN":
            self.agent = jammerFNNDDQNAgent(device=device)
        elif self.type == "PPO":
            self.agent = jammerPPOAgent(device=device)

    def choose_action(self, observation):
        if self.type == "PPO":
            action, logprob, value = self.agent.choose_action(observation)
            return action.unsqueeze(0), logprob.unsqueeze(0), value
        else:
            return self.agent.choose_action(observation)
        
    # Used for the DDQN smart jammer
    def update_target_q_network(self):
        self.agent.update_target_q_network()
    
    # Used for the DDQN smart jammer
    def store_experience_in(self, state, action, reward, next_state):
        self.agent.store_experience_in(state, action, reward, next_state)

    # Used for the PPO smart jammer
    def store_in_memory(self, state, action, logprob, reward, value):
        self.agent.store_in_memory(state, action, logprob, reward, value)

    # Used for the PPO smart jammer
    def clear_memory(self):
        self.agent.clear_memory()

    def update(self):
        if self.type == "RNN" or self.type == "FNN":
            self.agent.replay()
        elif self.type == "PPO":
            self.agent.update()