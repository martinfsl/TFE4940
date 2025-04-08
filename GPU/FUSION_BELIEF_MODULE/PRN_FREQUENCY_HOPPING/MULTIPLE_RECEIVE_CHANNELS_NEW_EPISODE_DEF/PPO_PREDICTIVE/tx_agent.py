import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

import copy

from constants import *
from fh_pattern import FH_Pattern

from agentNetworks import *

class txPPOAgent:
    def __init__(self, gamma=GAMMA, learning_rate = LEARNING_RATE, 
                lambda_param = LAMBDA, epsilon_clip = EPSILON_CLIP, 
                k = K, m = M, c1 = C1, c2 = C2, c3 = C3,
                device = "cpu"):
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.LAMBDA = lambda_param
        self.epsilon_clip = epsilon_clip

        self.k = k
        self.m = m
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

        self.power = TX_USER_TRANSMIT_POWER
        self.h_tr_variance = H_TR_VARIANCE
        self.h_tj_variance = H_JT_VARIANCE

        self.device = device

        # PPO on-policy storage (use lists to store one episode/trajectory)
        self.memory_state = torch.empty((0, PPO_NETWORK_INPUT_SIZE), device=self.device)
        self.memory_action = torch.empty((0, 1), device=self.device)
        self.memory_logprob = torch.empty((0, 1), device=self.device)
        self.memory_reward = torch.empty((0, 1), device=self.device)
        self.memory_value = torch.empty((0, 1), device=self.device)

        self.memory_belief_input = torch.empty((0, PREDICTION_NETWORK_INPUT_SIZE), device=self.device)
        self.memory_rx_action = torch.empty((0, 1), device=self.device)
        self.memory_belief_mask = torch.empty((0, 1), device=self.device)

        self.previous_seeds = torch.empty((0, 1), device=self.device)

        # Actor-Critic network with belief module
        self.ac_network = ActorCriticNetwork_().to(self.device)
        self.ac_network_optimizer = optim.Adam(self.ac_network.parameters(), lr=self.learning_rate)

        # Logging actor and critic losses
        self.actor_losses = torch.tensor([], device=self.device)
        self.critic_losses = torch.tensor([], device=self.device)
        self.belief_losses = torch.tensor([], device=self.device)

        # Logging the channel selections
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

        self.memory_belief_input = torch.empty((0, PREDICTION_NETWORK_INPUT_SIZE), device=self.device)
        self.memory_rx_action = torch.empty((0, 1), device=self.device)
        self.memory_belief_mask = torch.empty((0, 1), device=self.device)

    def store_in_memory(self, state, action, logprob, reward, value, belief_input, observed_rx_action, belief_mask):
        self.memory_state = torch.cat((self.memory_state, state.unsqueeze(0)), dim=0)
        self.memory_action = torch.cat((self.memory_action, action.unsqueeze(0)), dim=0)
        self.memory_logprob = torch.cat((self.memory_logprob, logprob.unsqueeze(0)), dim=0)
        self.memory_reward = torch.cat((self.memory_reward, reward.unsqueeze(0)), dim=0)
        self.memory_value = torch.cat((self.memory_value, value.unsqueeze(0)), dim=0)

        self.memory_belief_input = torch.cat((self.memory_belief_input, belief_input.unsqueeze(0)), dim=0)
        self.memory_rx_action = torch.cat((self.memory_rx_action, observed_rx_action.unsqueeze(0)), dim=0)
        self.memory_belief_mask = torch.cat((self.memory_belief_mask, belief_mask.unsqueeze(0)), dim=0)

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
        # return self.critic_network(observation)

        batch_size = observation.size(0)
        _, state_value = self.ac_network(observation, torch.zeros((batch_size, PREDICTION_NETWORK_INPUT_SIZE), device=self.device), dimension=1)
        return state_value
    
    # Only returning the most probable action
    def choose_action(self, observation, pred_observation):
        with torch.no_grad():
            action_probs, state_value = self.ac_network(observation, pred_observation)
            policy = nn.Softmax(dim=0)(action_probs)
            values = state_value

        action = torch.argmax(policy)
        action_logprob = torch.log(torch.gather(policy, 0, action.unsqueeze(0)))
        # self.fh_seeds_used = torch.cat((self.fh_seeds_used, action.unsqueeze(0)))

        additional_sense = torch.argsort(policy, descending=True)[1:NUM_EXTRA_ACTIONS+1]

        return action.unsqueeze(0), action_logprob, values, additional_sense
    
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

        data_size = self.memory_state.size(0)

        for epoch in range(self.k):
            batch_indices = torch.randperm(data_size)[:self.m]

            batch_state = self.memory_state[batch_indices]
            batch_action = self.memory_action[batch_indices]
            batch_logprob = self.memory_logprob[batch_indices]
            batch_return = returns[batch_indices].detach()
            batch_advantage = advantages[batch_indices].detach()

            batch_belief_input = self.memory_belief_input[batch_indices]
            batch_rx_action = self.memory_rx_action[batch_indices]
            batch_belief_mask = self.memory_belief_mask[batch_indices]

            new_logits, batch_value = self.ac_network(batch_state, batch_belief_input, dimension=1)
            new_policy = nn.Softmax(dim=1)(new_logits)
            new_logprobs = torch.log(torch.gather(new_policy, 1, batch_action.long()))

            # Actor loss
            ratio = torch.exp(new_logprobs - batch_logprob)
            surr1 = ratio*batch_advantage
            surr2 = torch.clamp(ratio, 1-self.epsilon_clip, 1+self.epsilon_clip)*batch_advantage
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            # batch_value = self.get_value(batch_state)
            critic_loss = nn.MSELoss()(batch_value, batch_return)

            # Entropy loss
            entropy = (new_policy*torch.log(new_policy + 1e-8)).sum(dim=1).mean()
            
            if USE_PREDICTION:
                # Belief loss
                usable_indices = batch_belief_mask.squeeze() == 1
                if usable_indices.sum() > 0:
                    usable_logits = self.ac_network.belief_encoder(batch_belief_input[usable_indices])
                    usable_targets = batch_rx_action[usable_indices].squeeze().long()
                    if usable_logits.shape[0] == 1:
                        usable_targets = usable_targets.unsqueeze(0)
                    belief_loss = nn.CrossEntropyLoss()(usable_logits, usable_targets)
                else:
                    belief_loss = torch.tensor(0.0, device=self.device)
            else:
                belief_loss = torch.tensor(0.0, device=self.device)

            total_loss = actor_loss + self.c1*critic_loss + self.c2*entropy + self.c3*belief_loss

            self.ac_network_optimizer.zero_grad()
            total_loss.backward()
            self.ac_network_optimizer.step()

        self.clear_memory()

        self.actor_losses = torch.cat((self.actor_losses, actor_loss.unsqueeze(0)))
        self.critic_losses = torch.cat((self.critic_losses, critic_loss.unsqueeze(0)))
        self.belief_losses = torch.cat((self.belief_losses, belief_loss.unsqueeze(0)))