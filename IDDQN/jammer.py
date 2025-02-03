import random
import numpy as np

from constants import *

#################################################################################
### Defining class for jammers and interfering users
#################################################################################

# Class that represents the other users in the environment
# Possible behaviors = [random, fixed, sweep, probabilistic, spectrum sensing]
# Possible weights = any vector with size NUM_CHANNELS and integers >= 1 ----> ONLY APPLICABLE FOR behavior = probabilistic
# Initialization can look like: other_user = UserEnvironment(behavior = "probabilistic", weights = [5, 2, 2]) for a system with three channels
class UserEnvironment:
    def __init__(self, behavior = "fixed", channel = 0, weights = [1]*NUM_CHANNELS, sweep_interval = 1):
        self.behavior = behavior # Used to determine how the actions are chosen
        self.channel = channel # Used to choose channel in static, fixed behavior, initialization for spectrum sensing
        self.weights = weights # Used for probabilistic behavior
        
        self.index_sweep = channel # To keep track of the channel for round robin
        self.sweep_interval = sweep_interval # How many time steps to wait before sweeping to the next channel
        self.sweep_counter = 0

        self.power = JAMMER_TRANSMIT_POWER
        self.h_jt_variance = H_JT_VARIANCE # Variance of the Rayleigh distribution for the Rayleigh fading from jammer to transmitter
        self.h_jr_variance = H_JR_VARIANCE # Variance of the Rayleigh distribution for the Rayleigh fading from jammer to receiver

    def choose_action(self):
        if self.behavior == "random":
            return random.randint(0, NUM_CHANNELS-1)
        elif self.behavior == "fixed": # Should have a probability of not transmitting as well
            if random.uniform(0, 1) < 0.9:
                return self.channel
            else:
                return random.randint(0, NUM_CHANNELS-1) # Random channel 10% of the time
        elif self.behavior == "sweep": # Round-robin across all channels + no transmission
            if self.sweep_counter >= self.sweep_interval:
                self.index_sweep = (self.index_sweep+1) % NUM_CHANNELS
                self.sweep_counter = 0
            self.sweep_counter += 1
            return self.index_sweep
        elif self.behavior == "probabilistic":
            return random.choices(range(NUM_CHANNELS), weights=self.weights, k=1)[0]
        elif self.behavior == "spectrum sensing":
            return self.channel

    def select_and_get_action(self):
        action = self.choose_action()
        
        return action

    # Function that returns the transmit power of the jammer / interfering user
    # The power is multiplied by the Rayleigh fading magnitude
    def get_transmit_power(self, direction):
        if CONSIDER_FADING:
            if direction == "transmitter":
                h = np.abs(np.random.normal(0, self.h_jt_variance, 1) + 1j*np.random.normal(0, self.h_jt_variance, 1))
            elif direction == "receiver":
                h = np.abs(np.random.normal(0, self.h_jr_variance, 1) + 1j*np.random.normal(0, self.h_jr_variance, 1))
            received_power = (h*self.power)[0]
        else:
            received_power = self.power

        return received_power