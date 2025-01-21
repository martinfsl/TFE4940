#############################################
"""
This simulation is closer to how the final simulator should work.
Here there is a main user that utilizes Q-learning for effective spectrum access.
There are other users as well, but they use algorithms such as:
    * Random behavior
    * Fixed behavior
    * Probabilistic behavior
"""
#############################################

import numpy as np
import random
# random.seed(1488)

from collections import defaultdict

import matplotlib.pyplot as plt

from constants import *

from tqdm import tqdm

import copy

import torch
import torch.nn as nn
import torch.optim as optim

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

#################################################################################
### Defining classes RNNQN and RNNQN-agent
#################################################################################

class RNNQN(nn.Module):
    def __init__(self):
        super(RNNQN, self).__init__()

        self.input_size = NUM_CHANNELS + 1
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

class RNNQNAgent:
    def __init__(self, gamma = GAMMA, epsilon = EPSILON):
        self.gamma = gamma
        self.epsilon = epsilon

        self.learning_rate = LEARNING_RATE

        # Power
        self.power = USER_TRANSMIT_POWER
        self.h_tr_variance = H_TR_VARIANCE # Variance of the Rayleigh distribution for the Rayleigh fading from transmitter to receiver

        # Parameters for the RNN network A
        self.batch_size = DQN_BATCH_SIZE
        self.memory = []

        self.q_network = RNNQN()
        self.target_network = RNNQN()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr = self.learning_rate)

    def get_transmit_power(self):
        if CONSIDER_FADING:
            h = np.abs(np.random.normal(0, self.h_tr_variance, 1) + 1j*np.random.normal(0, self.h_tr_variance, 1))
            received_power = (h*self.power)[0]
        else:
            received_power = self.power

        return received_power
    
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            self.epsilon *= EPSILON_REDUCTION
            return random.choice(range(NUM_CHANNELS))
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            hidden = self.q_network.init_hidden(1)
            q_values, _ = self.q_network(state, hidden)
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

#################################################################################
### Calculate the reward using SINR
#################################################################################

def calculate_reward(action, transmit_power, channel_noise_receiver, channel_noise_transmitter):
    sinr = 10*np.log10(transmit_power / channel_noise_receiver[action]) # SINR at the receiver

    if USE_SINR_AS_REWARD:
        reward = sinr
    else:
        if sinr > SINR_THRESHOLD:
            reward = REWARD_SUCCESSFUL
        else:
            reward = REWARD_INTERFERENCE

    # ACK goes back through the fading channel
    sinr_ack = 10*np.log10(transmit_power / channel_noise_transmitter[action]) # SINR of the ACK at the transmitter
    if sinr_ack < SINR_THRESHOLD:
        if USE_SINR_AS_REWARD:
            reward = -100 # ACK not received, the SINR reward is very low
        else:
            reward = REWARD_INTERFERENCE

    return reward

#################################################################################
### Train the DQN, extended state space
#################################################################################

def train_dqn_extended_state_space(agent, other_users):
    print("Testing what it might look like with the extended state space\n\n")

    accumulated_rewards = []
    average_rewards = []

    # Initialize the start channel for the sweep
    for ou in other_users:
        if ou.behavior == "sweep":
            ou.index_sweep = ou.channel

    ###################################
    # Initializing the first state
    curr_state = []

    prev_user_action = 0

    # Initialize the channel noise for the current (and first) time step
    channel_noise_transmitter = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))

    # Select actions for the jammers and other interfering users
    for ou in other_users:
        ou_action = ou.select_and_get_action()

        channel_noise_transmitter[ou_action] += ou.get_transmit_power(direction = "transmitter")

    # Set the current state based on the observed power spectrum
    curr_state = channel_noise_transmitter.tolist()
    curr_state.append(prev_user_action)
    ###################################

    for episode in tqdm(range(NUM_EPISODES)):
        # The agent chooses an action based on the current state
        action = agent.choose_action(curr_state)

        # Set a new channel noise for the next state
        channel_noise_transmitter = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))
        channel_noise_receiver = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))

        # Select actions for the jammers and other interfering users again
        for ou in other_users:
            ou_action = ou.select_and_get_action()

            channel_noise_transmitter[ou_action] += ou.get_transmit_power(direction = "transmitter")
            channel_noise_receiver[ou_action] += ou.get_transmit_power(direction = "receiver")

        # Add the new observed power spectrum to the next state
        next_state = channel_noise_transmitter.tolist()
        next_state.append(action)

        for ou in other_users:
            if ou.behavior == "spectrum sensing":
                ou.channel = action

        # Calculate the reward based on the action taken
        reward = calculate_reward(action, agent.get_transmit_power(), channel_noise_receiver, channel_noise_transmitter)

        if episode == 0:
            accumulated_rewards.append(reward)
            average_rewards.append(reward)
        else:
            accumulated_rewards.append(accumulated_rewards[-1] + reward)
            average_rewards.append(accumulated_rewards[-1]/len(accumulated_rewards))

        # Store the experience in the agent's memory
        # Replay the agent's memory
        agent.store_experience_in(curr_state, action, reward, next_state)
        agent.replay()

        # Periodic update of the target Q-network
        if episode % 10 == 0:
            agent.update_target_q_network()

        curr_state = next_state

    print("Training complete")

    return average_rewards

#################################################################################
### Test the DQN, extended state space
#################################################################################

def test_dqn_extended_state_space(agent, other_users):
    print("Testing what it might look like with the extended state space")

    num_successful_transmissions = 0
    num_channel_selected = np.zeros(NUM_CHANNELS)

    # Initialize the start channel for the sweep
    for ou in other_users:
        if ou.behavior == "sweep":
            ou.index_sweep = ou.channel
        
    ###################################
    # Initializing the first state
    curr_state = []

    prev_user_action = 0

    # Initialize the channel noise for the current (and first) time step
    channel_noise_transmitter = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))

    # Select actions for the jammers and other interfering users
    for ou in other_users:
        ou_action = ou.select_and_get_action()

        channel_noise_transmitter[ou_action] += ou.get_transmit_power(direction = "transmitter")

    # Set the current state based on the observed power spectrum
    curr_state = channel_noise_transmitter.tolist()
    curr_state.append(prev_user_action)

    ###################################

    for run in tqdm(range(NUM_TEST_RUNS)):

        # The agent chooses an action based on the current state
        action = agent.choose_action(curr_state)
        num_channel_selected[action] += 1

        # Set a new channel noise for the next state
        channel_noise_transmitter = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))
        channel_noise_receiver = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))

        # Select actions for the jammers and other interfering users again
        for ou in other_users:
            ou_action = ou.select_and_get_action()

            channel_noise_transmitter[ou_action] += ou.get_transmit_power(direction = "transmitter")
            channel_noise_receiver[ou_action] += ou.get_transmit_power(direction = "receiver")

        # Add the new observed power spectrum to the next state
        next_state = channel_noise_transmitter.tolist()
        next_state.append(action)

        # Calculate the SINR
        transmit_power = agent.get_transmit_power()

        for ou in other_users:
            if ou.behavior == "spectrum sensing":
                ou.channel = action

        # Calculate the reward based on the action taken
        reward = calculate_reward(action, transmit_power, channel_noise_receiver, channel_noise_transmitter)
        if reward >= 0:
            num_successful_transmissions += 1

        # Set the state for the next iteration based on the new observed power spectrum
        curr_state = next_state

    print("\nTesting a random state to see the Q-values\n\n")

    print("Last (next) state in the test: ", curr_state)
    print("Last user action in the test: ", action)

    probability_channel_selected = num_channel_selected / np.sum(num_channel_selected)

    # Testing a random state to see the Q-values
    channel_noise_once_transmitter = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))
    
    # Select actions for the jammers and other interfering users
    ou_actions = []
    for ou in other_users:
        ou_action = ou.select_and_get_action()
        ou_actions.append(ou_action)

        channel_noise_once_transmitter[ou_action] += ou.get_transmit_power(direction = "transmitter")

    print("Actions of other users in random state test: ", ou_actions)

    print("Channel noise at the transmitter in random state test: ", channel_noise_once_transmitter)

    print("\n\n")
    print("Current state in the test: ", curr_state, "\n")
    new_action = agent.choose_action(curr_state)
    next_state = channel_noise_once_transmitter.tolist()
    next_state.append(new_action)
    print("Next state that the agent will observe: ", next_state, "\n")

    state_once = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)

    # Q-values from the Q-network A for the random state
    q_values = agent.q_network(state_once)

    return num_successful_transmissions, probability_channel_selected, state_once, q_values

#################################################################################
### Plotting the results
#################################################################################

def plot_results(average_rewards, probability_channel_selected, state_once, q_values):
    plt.figure(1, figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(average_rewards)
    plt.axvline(x=2*DQN_BATCH_SIZE, color='r', linestyle='--', label='2*Batch Size (Here training begins)')
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.title("Average reward over episodes during training")
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.bar(np.arange(1, NUM_CHANNELS+1, 1), probability_channel_selected)
    plt.xlabel("Channel")
    plt.ylabel("Probability of channel selection")
    plt.title("Probability of channel selection during testing")

    plt.subplot(2, 2, 3)
    plt.bar(np.arange(1, NUM_CHANNELS+1, 1), state_once.detach().numpy()[0][0:NUM_CHANNELS])
    plt.xlabel("Channel")
    plt.ylabel("Power level")
    plt.title("Latest power spectrum that transmitter observes")

    plt.subplot(2, 2, 4)
    plt.bar(np.arange(1, NUM_CHANNELS+1, 1), q_values[0].detach().numpy()[0])
    plt.xlabel("Channel")
    plt.ylabel("Q-values")
    plt.title("Q-values for the random state")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.suptitle(f"GRU (Sequential training), {NUM_CHANNELS} channels, 1 spectrum sensing, {DQN_BATCH_SIZE} sequence length")
    # plt.suptitle(f"GRU (Sequential training), {NUM_CHANNELS} channels, 1 sweeping, {DQN_BATCH_SIZE} sequence length")
    # plt.suptitle(f"GRU (Sequential training), {NUM_CHANNELS} channels, 1 probabilistic [5, 1, 1, 1, 3], {DQN_BATCH_SIZE} sequence length")
    plt.suptitle(f"GRU (Sequential training), {NUM_CHANNELS} channels, 1 random, {DQN_BATCH_SIZE} sequence length")
    plt.show()

#################################################################################
### main()
#################################################################################

if __name__ == '__main__':

    success_rates = []
    
    num_runs = 5

    for run in range(num_runs):
        print(f"Run: {run+1}")

        # agent = DQNAgent()
        agent = RNNQNAgent()

        list_of_other_users = []

        # random_1 = UserEnvironment(behavior = "random")
        # list_of_other_users.append(random_1)

        sweep_1 = UserEnvironment(behavior = "sweep", channel = 2, sweep_interval = 1)
        list_of_other_users.append(sweep_1)

        # prob_1 = UserEnvironment(behavior = "probabilistic", weights = [5, 1, 1, 1, 3]*(NUM_CHANNELS//5))
        # list_of_other_users.append(prob_1)

        # spec_sense_1 = UserEnvironment(behavior = "spectrum sensing", channel = 0)
        # list_of_other_users.append(spec_sense_1)

        average_rewards = train_dqn_extended_state_space(agent, list_of_other_users)

        num_successful_transmissions, probability_channel_selected, state_once, q_values = test_dqn_extended_state_space(agent, list_of_other_users)

        print("Finished testing the DQN:")
        print("Successful transmission rate: ", (num_successful_transmissions/NUM_TEST_RUNS)*100, "%")
        success_rates.append((num_successful_transmissions/NUM_TEST_RUNS)*100)

    plot_results(average_rewards, probability_channel_selected, state_once, q_values)

    if num_runs > 1:
        print("Success rates: ", success_rates)
        print("Average success rate: ", np.mean(success_rates), "%")

    total_params = sum(p.numel() for p in agent.q_network.parameters())
    total_params = sum(p.numel() for p in agent.target_network.parameters())
    print(f"Number of parameters: {total_params}")