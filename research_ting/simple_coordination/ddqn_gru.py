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

from tx_agent import txRNNQN, txRNNQNAgent
from rx_agent import rxRNNQN, rxRNNQNAgent
from jammer import UserEnvironment

#################################################################################
### Calculate the reward using SINR
#################################################################################

def received_signal_rx(channel, received_power, channel_noise_receiver):
    sinr = 10*np.log10(received_power / channel_noise_receiver[channel]) # SINR at the receiver

    return sinr > SINR_THRESHOLD

def received_signal_tx(channel, received_power, channel_noise_transmitter):
    sinr = 10*np.log10(received_power / channel_noise_transmitter[channel]) # SINR at the receiver

    return sinr > SINR_THRESHOLD

#################################################################################
### Train the DQN, extended state space
#################################################################################

def train_dqn_extended_state_space(tx_agent, other_users):
    print("Training the DDQN")
    accumulated_rewards = []
    average_rewards = []

    # Initialize the start channel for the sweep
    for ou in other_users:
        if ou.behavior == "sweep":
            ou.index_sweep = ou.channel

    ###################################
    # Initializing the first state
    curr_state = []

    prev_tx_channel = 0

    # Initialize the channel noise for the current (and first) time step
    channel_noise_transmitter = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))

    # Select actions for the jammers and other interfering users
    for ou in other_users:
        ou_action = ou.select_and_get_action()

        channel_noise_transmitter[ou_action] += ou.get_transmit_power(direction = "transmitter")

    # Set the current state based on the observed power spectrum
    curr_state = channel_noise_transmitter.tolist()
    curr_state.append(prev_tx_channel)
    ###################################

    for episode in tqdm(range(NUM_EPISODES)):
        # The agent chooses an action based on the current state
        tx_channel = tx_agent.choose_action(curr_state)

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
        next_state.append(tx_channel)

        for ou in other_users:
            if ou.behavior == "spectrum sensing":
                ou.channel = tx_channel

        # Calculate the reward based on the action taken
        # ACK is sent from the receiver
        if received_signal_rx(tx_channel, tx_agent.get_transmit_power(), channel_noise_receiver):
            # ACK is received at the transmitter
            if received_signal_tx(tx_channel, tx_agent.get_transmit_power(), channel_noise_transmitter): # power should be changed to rx_agent later
                reward = REWARD_SUCCESSFUL
        else:
            reward = REWARD_INTERFERENCE

        if episode == 0:
            accumulated_rewards.append(reward)
            average_rewards.append(reward)
        else:
            accumulated_rewards.append(accumulated_rewards[-1] + reward)
            average_rewards.append(accumulated_rewards[-1]/len(accumulated_rewards))

        # Store the experience in the agent's memory
        # Replay the agent's memory
        tx_agent.store_experience_in(curr_state, tx_channel, reward, next_state)
        tx_agent.replay()

        # Periodic update of the target Q-network
        if episode % 10 == 0:
            tx_agent.update_target_q_network()

        curr_state = next_state

    print("Training complete")

    return average_rewards

#################################################################################
### Test the DQN, extended state space
#################################################################################

def test_dqn_extended_state_space(tx_agent, other_users):
    print("Testing the DDQN")
    num_successful_transmissions = 0
    num_channel_selected = np.zeros(NUM_CHANNELS)

    # Initialize the start channel for the sweep
    for ou in other_users:
        if ou.behavior == "sweep":
            ou.index_sweep = ou.channel
        
    ###################################
    # Initializing the first state
    curr_state = []

    prev_tx_channel = 0

    # Initialize the channel noise for the current (and first) time step
    channel_noise_transmitter = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))

    # Select actions for the jammers and other interfering users
    for ou in other_users:
        ou_action = ou.select_and_get_action()

        channel_noise_transmitter[ou_action] += ou.get_transmit_power(direction = "transmitter")

    # Set the current state based on the observed power spectrum
    curr_state = channel_noise_transmitter.tolist()
    curr_state.append(prev_tx_channel)

    ###################################

    for run in tqdm(range(NUM_TEST_RUNS)):

        # The agent chooses an action based on the current state
        tx_channel = tx_agent.choose_action(curr_state)
        num_channel_selected[tx_channel] += 1

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
        next_state.append(tx_channel)

        # Calculate the SINR
        transmit_power = tx_agent.get_transmit_power()

        for ou in other_users:
            if ou.behavior == "spectrum sensing":
                ou.channel = tx_channel

        # Calculate the reward based on the action taken
        # ACK is sent from the receiver
        if received_signal_rx(tx_channel, tx_agent.get_transmit_power(), channel_noise_receiver):
            # ACK is received at the transmitter
            if received_signal_tx(tx_channel, tx_agent.get_transmit_power(), channel_noise_transmitter): # power should be changed to rx_agent later
                reward = REWARD_SUCCESSFUL
        else:
            reward = REWARD_INTERFERENCE
        if reward >= 0:
            num_successful_transmissions += 1

        # Set the state for the next iteration based on the new observed power spectrum
        curr_state = next_state

    probability_channel_selected = num_channel_selected / np.sum(num_channel_selected)

    return num_successful_transmissions, probability_channel_selected

#################################################################################
### Plotting the results
#################################################################################

def plot_results(average_rewards, probability_channel_selected, jammer_type):
    plt.figure(1, figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(average_rewards)
    plt.axvline(x=2*DQN_BATCH_SIZE, color='r', linestyle='--', label='2*Batch Size (Here training begins)')
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.title("Average reward over episodes during training")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.bar(np.arange(1, NUM_CHANNELS+1, 1), probability_channel_selected)
    plt.xlabel("Channel")
    plt.ylabel("Probability of channel selection")
    plt.title("Probability of channel selection during testing")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f"DDQN GRU, {NUM_CHANNELS} channels, 1 {jammer_type}, {DQN_BATCH_SIZE} sequence length")
    plt.show()

#################################################################################
### main()
#################################################################################

if __name__ == '__main__':

    success_rates = []
    
    num_runs = 1

    for run in range(num_runs):
        print("----------------------------------------")
        print("----------------------------------------")

        print(f"Run: {run+1}")

        tx_agent = txRNNQNAgent()

        list_of_other_users = []

        # random_1 = UserEnvironment(behavior = "random")
        # list_of_other_users.append(random_1)
        # jammer_type = "random"

        # sweep_1 = UserEnvironment(behavior = "sweep", channel = 2, sweep_interval = 1)
        # list_of_other_users.append(sweep_1)
        # jammer_type = "sweeping"

        # prob_1 = UserEnvironment(behavior = "probabilistic", weights = [5, 1, 1, 1, 3]*(NUM_CHANNELS//5))
        # list_of_other_users.append(prob_1)
        # jammer_type = "probabilistic [5, 1, 1, 1, 3]"

        spec_sense_1 = UserEnvironment(behavior = "spectrum sensing", channel = 0)
        list_of_other_users.append(spec_sense_1)
        jammer_type = "spectrum sensing"

        average_rewards = train_dqn_extended_state_space(tx_agent, list_of_other_users)

        num_successful_transmissions, probability_channel_selected = test_dqn_extended_state_space(tx_agent, list_of_other_users)

        print("Finished testing the DQN:")
        print("Successful transmission rate: ", (num_successful_transmissions/NUM_TEST_RUNS)*100, "%")
        success_rates.append((num_successful_transmissions/NUM_TEST_RUNS)*100)

    plot_results(average_rewards, probability_channel_selected, jammer_type)

    if num_runs > 1:
        print("Success rates: ", success_rates)
        print("Average success rate: ", np.mean(success_rates), "%")

    total_params = sum(p.numel() for p in tx_agent.q_network.parameters())
    total_params = sum(p.numel() for p in tx_agent.target_network.parameters())
    print(f"Number of parameters: {total_params}")

    np.savetxt("research_ting/simple_coordination/aa_comparison/average_reward_only_tx.txt", average_rewards)