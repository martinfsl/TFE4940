import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

import copy

from constants import *
from fh_pattern import FH_Pattern

from agentNetworks import *

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
        self.memory_state = torch.empty((0, PPO_NETWORK_INPUT_SIZE), device=self.device)
        self.memory_action = torch.empty((0, 1), device=self.device)
        self.memory_logprob = torch.empty((0, 1), device=self.device)
        self.memory_reward = torch.empty((0, 1), device=self.device)
        self.memory_value = torch.empty((0, 1), device=self.device)

        self.previous_seeds = torch.empty((0, 1), device=self.device)

        # Policy network (Actor network)
        self.actor_network = ActorNetwork()
        self.actor_network.to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.learning_rate)

        self.actor_network_old = copy.deepcopy(self.actor_network).to(self.device)

        # Value network (Critic network)
        self.critic_network = CriticNetwork()
        self.critic_network.to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.learning_rate)

        # Predictive network remains unchanged
        self.pred_agent = PredictionNNAgent(device=self.device)

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
        self.memory_state = torch.empty((0, PPO_NETWORK_INPUT_SIZE), device=self.device)
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

    def get_observation(self, state, action, seed):
        observation_pattern = torch.zeros(STATE_SPACE_SIZE-1, device=self.device)

        for i in range(NUM_HOPS):
            # Same as before: create observation by concatenating state and action data.
            if NUM_SENSE_CHANNELS < NUM_CHANNELS:
                observation = torch.zeros(NUM_SENSE_CHANNELS + 1, device=self.device)
                half_sense_channels = NUM_SENSE_CHANNELS // 2
                for j in range(-half_sense_channels, half_sense_channels + 1):
                    index = (action[i] + j) % len(state[i])
                    observation[j + half_sense_channels] = state[i][index.long()]
                observation[-1] = action[i]
            else:
                observation = torch.cat((state[i], action[i].unsqueeze(0)), dim=0)
        
            observation_pattern[i*(NUM_SENSE_CHANNELS+1):(i+1)*(NUM_SENSE_CHANNELS+1)] = observation

        observation_pattern = torch.concat((observation_pattern, seed))

        return observation_pattern
    
    def concat_predicted_action(self, observation, pred_observation):
        # Get the predicted action from the predictive network
        with torch.no_grad():
            pred_action = self.pred_agent.predict_action(pred_observation)
        # Concatenate original observation and predicted action.
        observation = torch.concat((observation, pred_action))

        predicted_action = torch.argmax(pred_action).unsqueeze(0)

        return observation, predicted_action

    def get_value(self, observation):
        return self.critic_network(observation)

    # Only returning the most probable action
    def choose_action(self, observation):
        with torch.no_grad():
            policy = nn.Softmax(dim=0)(self.actor_network(observation))
            values = self.get_value(observation)

        action = torch.argmax(policy)
        action_logprob = torch.log(torch.gather(policy, 0, action.unsqueeze(0)))
        # self.fh_seeds_used = torch.cat((self.fh_seeds_used, action.unsqueeze(0)))

        additional_actions = torch.argsort(policy, descending=True)[1:NUM_EXTRA_RECEIVE+1]
        additional_actions_logprob = torch.log(torch.gather(policy, 0, additional_actions))
        additional_sense = torch.argsort(policy, descending=True)[NUM_EXTRA_RECEIVE+1:NUM_EXTRA_RECEIVE+NUM_EXTRA_ACTIONS+1]

        return action.unsqueeze(0), action_logprob, values, additional_actions, additional_actions_logprob, additional_sense

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