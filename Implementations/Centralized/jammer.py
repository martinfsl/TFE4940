import random
import numpy as np

import copy

from constants import *

import torch
import torch.nn as nn
import torch.optim as optim
from jammerSmart import SmartJammer
from jammerGenie import GenieJammer

#################################################################################
### Defining class for jammers and interfering users
#################################################################################

# Class that represents the other users in the environment
# Possible behaviors = [random, fixed, sweep, probabilistic, tracking]
# Possible weights = any vector with size NUM_CHANNELS and integers >= 1 ----> ONLY APPLICABLE FOR behavior = probabilistic
# Initialization can look like: jammer = Jammer(behavior = "probabilistic", weights = [5, 2, 2]) for a system with three channels
class Jammer:
    def __init__(self, behavior = "fixed", channel = 0, weights = [1]*NUM_CHANNELS, sweep_interval = 1, smart_type = "PPO", device = "cpu"):
        self.behavior = behavior # Used to determine how the actions are chosen
        self.channel = channel # Used to choose channel in static, fixed behavior, initialization for spectrum sensing
        self.weights = weights # Used for probabilistic behavior
        
        self.index_sweep = channel # To keep track of the channel for round robin
        self.sweep_interval = sweep_interval # How many time steps to wait before sweeping to the next channel
        self.sweep_counter = 0

        self.power = JAMMER_TRANSMIT_POWER
        self.h_jt_variance = H_JT_VARIANCE # Variance of the Rayleigh distribution for the Rayleigh fading from jammer to transmitter
        self.h_jr_variance = H_JR_VARIANCE # Variance of the Rayleigh distribution for the Rayleigh fading from jammer to receiver

        # Used by the smart jammer
        self.observed_tx_power = 0
        self.observed_noise = 0
        self.observed_reward = 0
        self.num_jammed = 0
        self.type = smart_type
        self.device = device

        if self.behavior == "genie":
            self.agent = GenieJammer(type=self.type, device=self.device)
        else:
            self.agent = SmartJammer(type=self.type, device=self.device)

        # Logging the channels selected
        self.channels_selected = torch.tensor([], device=self.device)

    def tracking_transition(self):
        curr_channel = self.channel
        curr_channel_plus_1 = min(self.channel+1, NUM_CHANNELS-1)
        curr_channel_minus_1 = max(0, self.channel-1)
        curr_channel_plus_2 = min(self.channel+2, NUM_CHANNELS-1)
        curr_channel_minus_2 = max(0, self.channel-2)
        curr_channel_plus_3 = min(self.channel+3, NUM_CHANNELS-1)
        curr_channel_minus_3 = max(0, self.channel-3)

        options = [curr_channel, curr_channel_plus_1, curr_channel_minus_1, curr_channel_plus_2, curr_channel_minus_2, curr_channel_plus_3, curr_channel_minus_3]
        probabilities = [TRANSITION_STAY, TRANSITION_1, TRANSITION_1, TRANSITION_2, TRANSITION_2, TRANSITION_3, TRANSITION_3]
        
        return random.choices(options, weights=probabilities, k=1)[0]
    
    # Used for the tracking behavior to observe the channels that the transmitter is on
    def observe_channels(self, observation):
        return torch.argmax(observation).detach() # Return the channel with the highest power

    def get_observation(self, state, action):
        if self.behavior == "smart" or self.behavior == "genie":
            # observation = state.clone()
            # observation = torch.cat((observation, torch.tensor([action], device=observation.device)))
            # return observation

            if NUM_JAMMER_SENSE_CHANNELS < NUM_CHANNELS:
                observation = torch.zeros(NUM_JAMMER_SENSE_CHANNELS + 1, device=self.device)
                half_sense_channels = NUM_JAMMER_SENSE_CHANNELS // 2
                for i in range(-half_sense_channels, half_sense_channels + 1):
                    index = (action + i) % len(state)
                    observation[i + half_sense_channels] = state[index]
                observation[-1] = action
            else:
                observation = torch.cat((state, action), dim=0)

            return observation
        # elif self.behavior == "tracking" or self.behavior == "genie":
        elif self.behavior == "tracking":
            return state.clone()
        else:
            return torch.tensor([], device=state.device)

    def choose_action(self, observation, tx_channel):
        if self.behavior == "smart" or self.behavior == "genie":
            action = torch.tensor(self.agent.choose_action(observation), device=observation.device)
        else:
            if self.behavior == "random":
                action = torch.tensor(random.randint(0, NUM_CHANNELS-1), device=observation.device)
            elif self.behavior == "fixed": # Should have a probability of not transmitting as well
                if random.uniform(0, 1) < 0.9:
                    action = torch.tensor(self.channel, device=observation.device)
                else:
                    action = torch.tensor(random.randint(0, NUM_CHANNELS-1), device=observation.device) # Random channel 10% of the time
            elif self.behavior == "sweep": # Round-robin across all channels + no transmission
                if self.sweep_counter >= self.sweep_interval:
                    self.index_sweep = (self.index_sweep+1) % NUM_CHANNELS
                    self.sweep_counter = 0
                self.sweep_counter += 1
                action = torch.tensor(self.index_sweep, device=observation.device)
            elif self.behavior == "probabilistic":
                action = torch.tensor(random.choices(range(NUM_CHANNELS), weights=self.weights, k=1)[0], device=observation.device)
            elif self.behavior == "tracking":
                self.channel = self.observe_channels(observation) # Update the channel that the jammer is on based on the observation
                curr_jam = self.tracking_transition()
                # self.channel = tx_channel # Update the channel that the jammer is on based on the transmitter's channel, without observation
                action = torch.tensor(curr_jam, device=observation.device)

        if self.behavior == "smart":
            self.channels_selected = torch.cat((self.channels_selected, action[0].unsqueeze(0)))
        elif self.behavior != "genie" and self.behavior != "smart":
            self.channels_selected = torch.cat((self.channels_selected, action.unsqueeze(0)))

        return action

    # Function that returns the transmit power of the jammer / interfering user
    # The power is multiplied by the Rayleigh fading magnitude
    def get_transmit_power(self, direction):
        if CONSIDER_FADING:
            if direction == "transmitter":
                # h = np.abs(np.random.normal(0, self.h_jt_variance, 1) + 1j*np.random.normal(0, self.h_jt_variance, 1))
                h_real = torch.normal(mean=0.0, std=self.h_jt_variance, size=(1,), device=self.device)
                h_imag = torch.normal(mean=0.0, std=self.h_jt_variance, size=(1,), device=self.device)
                h = torch.abs(torch.complex(h_real, h_imag))
            elif direction == "receiver":
                # h = np.abs(np.random.normal(0, self.h_jr_variance, 1) + 1j*np.random.normal(0, self.h_jr_variance, 1))
                h_real = torch.normal(mean=0.0, std=self.h_jr_variance, size=(1,), device=self.device)
                h_imag = torch.normal(mean=0.0, std=self.h_jr_variance, size=(1,), device=self.device)
                h = torch.abs(torch.complex(h_real, h_imag))
            received_power = (h*self.power)[0]
        else:
            received_power = self.power

        return received_power