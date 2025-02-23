import numpy as np
import random
# random.seed(1488)

import os

from collections import defaultdict

import matplotlib.pyplot as plt

from plotting import plot_results, plot_probability_selection, plot_results_smart_jammer

from constants import *

from tqdm import tqdm

import copy

import torch
import torch.nn as nn
import torch.optim as optim

from tx_agent import txRNNQN, txRNNQNAgent
from rx_agent import rxRNNQN, rxRNNQNAgent
from jammer import Jammer

#################################################################################
### Calculate the reward using SINR
#################################################################################

def received_signal_rx(tx_channel, rx_channel, received_power, channel_noise_receiver):
    sinr = 10*np.log10(received_power / channel_noise_receiver[rx_channel]) # SINR at the receiver

    return (sinr > SINR_THRESHOLD) and (np.abs(tx_channel - rx_channel) <= CHANNEL_OFFSET_THRESHOLD)

def received_signal_tx(tx_channel, rx_channel, received_power, channel_noise_transmitter):
    sinr = 10*np.log10(received_power / channel_noise_transmitter[tx_channel]) # SINR at the receiver

    return (sinr > SINR_THRESHOLD) and (np.abs(tx_channel - rx_channel) <= CHANNEL_OFFSET_THRESHOLD)

def sensed_signal_jammer(jammer_channel, tx_channel, jammer_power, channel_noise_jammer):
    sinr = 10*np.log10(jammer_power / channel_noise_jammer[jammer_channel]) # SINR at the jammer

    return (sinr > SINR_THRESHOLD) and (jammer_channel == tx_channel)

#################################################################################
### Train the DQN, extended state space
#################################################################################

def train_dqn(tx_agent, rx_agent, jammers):
    print("Training")
    tx_accumulated_rewards = []
    tx_average_rewards = []

    rx_accumulated_rewards = []
    rx_average_rewards = []

    jammer_accumulated_rewards = []
    jammer_average_rewards = []

    # Initialize the start channel for the sweep
    for jammer in jammers:
        if jammer.behavior == "sweep":
            jammer.index_sweep = jammer.channel

    ###################################
    # Initializing the first state consisting of just noise
    tx_state = []
    rx_state = []
    jammer_states = []
    for jammer in jammers:
        jammer_states.append([])

    tx_channel = 0
    rx_channel = 0
    jammer_channels = [0]*len(jammers)

    # Initialize the channel noise for the current (and first) time step
    tx_channel_noise = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))
    rx_channel_noise = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))
    jammer_channel_noises = []
    for jammer in jammers:
        noise = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))
        jammer_channel_noises.append(noise)
        if jammer.behavior == "smart":
            jammer.observed_noise = copy.deepcopy(noise)

    # Set the current state based on the total power spectrum
    tx_state = tx_channel_noise.tolist()
    rx_state = rx_channel_noise.tolist()
    for i in range(len(jammers)):
        jammer_states[i] = jammer_channel_noises[i].tolist()
    ###################################

    for episode in tqdm(range(NUM_EPISODES)):
        # The agents chooses an action based on the current state
        tx_observation = tx_agent.get_observation(tx_state, tx_channel)
        tx_channel = tx_agent.choose_action(tx_observation)
        print("Tx channel: ", tx_channel)
        tx_channel = tx_channel[0]
        rx_observation = rx_agent.get_observation(rx_state, rx_channel)
        rx_channel = rx_agent.choose_action(rx_observation)
        print("Rx channel: ", rx_channel)
        rx_channel = rx_channel[0]
        jammer_observations = []
        for i in range(len(jammers)):
            jammer_observation = jammers[i].get_observation(jammer_states[i], jammer_channels[i])
            jammer_observations.append(jammer_observation)
            jammer_channels[i] = jammers[i].choose_action(jammer_observation, tx_channel)

        # Set a new channel noise for the next state
        tx_channel_noise = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))
        rx_channel_noise = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))
        jammer_channel_noises = []
        for jammer in jammers:
            noise = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))
            jammer_channel_noises.append(noise)
            if jammer.behavior == "smart":
                jammer.observed_noise = copy.deepcopy(noise)

        # Add the power of the jammers to the channel noise for Tx and Rx
        # Add the power from the Tx to the jammers as well
        for i in range(len(jammers)):
            tx_channel_noise[jammer_channels[i]] += jammers[i].get_transmit_power(direction = "transmitter")
            rx_channel_noise[jammer_channels[i]] += jammers[i].get_transmit_power(direction = "receiver")
            jammers[i].observed_tx_power = tx_agent.get_transmit_power(direction = "jammer")
            jammer_channel_noises[i][tx_channel] += jammers[i].observed_tx_power

        # Add the new observed power spectrum to the next state
        tx_next_state = tx_channel_noise.tolist()
        tx_next_observation = tx_agent.get_observation(tx_next_state, tx_channel)
        rx_next_state = rx_channel_noise.tolist()
        rx_next_observation = rx_agent.get_observation(rx_next_state, rx_channel)
        jammer_next_states = []
        jammer_next_observations = []
        for i in range(len(jammers)):
            jammer_next_states.append(jammer_channel_noises[i].tolist())
            jammer_next_observations.append(jammers[i].get_observation(jammer_next_states[i], jammer_channels[i]))

        # Calculate the reward based on the action taken
        # ACK is sent from the receiver
        if received_signal_rx(tx_channel, rx_channel, tx_agent.get_transmit_power(direction = "receiver"), rx_channel_noise):
            rx_reward = REWARD_SUCCESSFUL
            
            # ACK is received at the transmitter
            if received_signal_tx(tx_channel, rx_channel, rx_agent.get_transmit_power(direction = "transmitter"), tx_channel_noise): # power should be changed to rx_agent later
                tx_reward = REWARD_SUCCESSFUL
            else:
                tx_reward = REWARD_INTERFERENCE
        else:
            rx_reward = REWARD_INTERFERENCE
            tx_reward = REWARD_INTERFERENCE

        for i in range(len(jammers)):
            if jammers[i].behavior == "smart":
                if sensed_signal_jammer(jammer_channels[i], tx_channel, jammers[i].observed_tx_power, jammers[i].observed_noise):
                    jammers[i].observed_reward = REWARD_SUCCESSFUL
                else:
                    jammers[i].observed_reward = REWARD_UNSUCCESSFUL

        if episode == 0:
            tx_accumulated_rewards.append(tx_reward)
            tx_average_rewards.append(tx_reward)

            rx_accumulated_rewards.append(rx_reward)
            rx_average_rewards.append(rx_reward)

            for i in range(len(jammers)):
                if jammers[i].behavior == "smart":
                    jammer_accumulated_rewards.append(jammers[i].observed_reward)
                    jammer_average_rewards.append(jammers[i].observed_reward)
        else:
            tx_accumulated_rewards.append(tx_accumulated_rewards[-1] + tx_reward)
            tx_average_rewards.append(tx_accumulated_rewards[-1]/len(tx_accumulated_rewards))

            rx_accumulated_rewards.append(rx_accumulated_rewards[-1] + rx_reward)
            rx_average_rewards.append(rx_accumulated_rewards[-1]/len(rx_accumulated_rewards))

            for i in range(len(jammers)):
                if jammers[i].behavior == "smart":
                    jammer_accumulated_rewards.append(jammer_accumulated_rewards[-1] + jammers[i].observed_reward)
                    jammer_average_rewards.append(jammer_accumulated_rewards[-1]/len(jammer_accumulated_rewards))

        # Store the experience in the agent's memory
        # Replay the agent's memory
        tx_agent.store_experience_in(tx_observation, tx_channel, tx_reward, tx_next_observation)
        tx_agent.replay()

        rx_agent.store_experience_in(rx_observation, rx_channel, rx_reward, rx_next_observation)
        rx_agent.replay()

        for i in range(len(jammers)):
            if jammers[i].behavior == "smart":
                jammers[i].agent.store_experience_in(jammer_observations[i], jammer_channels[i], jammers[i].observed_reward, jammer_next_observations[i])
                jammers[i].agent.replay()

        # Periodic update of the target Q-network
        if episode % 10 == 0:
            tx_agent.update_target_q_network()
            rx_agent.update_target_q_network()
            for i in range(len(jammers)):
                if jammers[i].behavior == "smart":
                    jammers[i].agent.update_target_q_network()

        tx_state = copy.deepcopy(tx_next_state)
        rx_state = copy.deepcopy(rx_next_state)
        for i in range(len(jammers)):
            jammer_states[i] = copy.deepcopy(jammer_next_states[i])

    print("Training complete")

    return tx_average_rewards, rx_average_rewards, jammer_average_rewards

#################################################################################
### Test the DQN, extended state space
#################################################################################

def test_dqn(tx_agent, rx_agent, jammers):
    print("Testing")
    num_successful_transmissions = 0
    num_tx_channel_selected = np.zeros(NUM_CHANNELS)
    num_rx_channel_selected = np.zeros(NUM_CHANNELS)
    num_jammer_channel_selected = np.zeros((len(jammers), NUM_CHANNELS))

    # Initialize the start channel for the sweep
    for jammer in jammers:
        if jammer.behavior == "sweep":
            jammer.index_sweep = jammer.channel
        
    ###################################
    # Initializing the first state
    tx_state = []
    rx_state = []
    jammer_states = []
    for jammer in jammers:
        jammer_states.append([])

    tx_channel = 0
    rx_channel = 0
    jammer_channels = [0]*len(jammers)

    # Initialize the first state consisting of just noise
    tx_channel_noise = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))
    rx_channel_noise = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))
    jammer_channel_noises = []
    for jammer in jammers:
        noise = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))
        jammer_channel_noises.append(noise)
        if jammer.behavior == "smart":
            jammer.observed_noise = copy.deepcopy(noise)

    # Set the current state based on the total power spectrum
    tx_state = tx_channel_noise.tolist()
    rx_state = rx_channel_noise.tolist()
    for i in range(len(jammers)):
        jammer_states[i] = jammer_channel_noises[i].tolist()
    ###################################

    for run in tqdm(range(NUM_TEST_RUNS)):
        # The agent chooses an action based on the current state
        tx_observation = tx_agent.get_observation(tx_state, tx_channel)
        tx_channel = tx_agent.choose_action(tx_observation)
        rx_observation = rx_agent.get_observation(rx_state, rx_channel)
        rx_channel = rx_agent.choose_action(rx_observation)
        jammer_observations = []
        for i in range(len(jammers)):
            jammer_observation = jammers[i].get_observation(jammer_states[i], jammer_channels[i])
            jammer_observations.append(jammer_observation)
            jammer_channels[i] = jammers[i].choose_action(jammer_observation, tx_channel)

        num_tx_channel_selected[tx_channel] += 1
        num_rx_channel_selected[rx_channel] += 1
        for i in range(len(jammers)):
            num_jammer_channel_selected[i][jammer_channels[i]] += 1

        # Set a new channel noise for the next state
        tx_channel_noise = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))
        rx_channel_noise = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))
        jammer_channel_noises = []
        for jammer in jammers:
            noise = np.abs(np.random.normal(0, NOISE_VARIANCE, NUM_CHANNELS))
            jammer_channel_noises.append(noise)
            if jammer.behavior == "smart":
                jammer.observed_noise = copy.deepcopy(noise)

        # Add the power of the jammers to the channel noise for Tx and Rx
        # Add the power from the Tx to the jammers as well
        for i in range(len(jammers)):
            tx_channel_noise[jammer_channels[i]] += jammers[i].get_transmit_power(direction = "transmitter")
            rx_channel_noise[jammer_channels[i]] += jammers[i].get_transmit_power(direction = "receiver")
            jammers[i].observed_tx_power = tx_agent.get_transmit_power(direction = "jammer")
            jammer_channel_noises[i][tx_channel] += jammers[i].observed_tx_power

        # Add the new observed power spectrum to the next state
        tx_next_state = tx_channel_noise.tolist()
        rx_next_state = rx_channel_noise.tolist()
        jammer_next_states = []
        for i in range(len(jammers)):
            jammer_next_states.append(jammer_channel_noises[i].tolist())

        # Calculate the reward based on the action taken
        # ACK is sent from the receiver
        if received_signal_rx(tx_channel, rx_channel, tx_agent.get_transmit_power(direction = "receiver"), rx_channel_noise):
            rx_reward = REWARD_SUCCESSFUL
            
            # ACK is received at the transmitter
            if received_signal_tx(tx_channel, rx_channel, rx_agent.get_transmit_power(direction = "transmitter"), tx_channel_noise): # power should be changed to rx_agent later
                tx_reward = REWARD_SUCCESSFUL
            else:
                tx_reward = REWARD_INTERFERENCE
        else:
            rx_reward = REWARD_INTERFERENCE
            tx_reward = REWARD_INTERFERENCE
        # If the tx_reward is positive, then both the communication link was successful both ways
        if tx_reward >= 0:
            num_successful_transmissions += 1

        for i in range(len(jammers)):
            if jammers[i].behavior == "smart":
                if sensed_signal_jammer(jammer_channels[i], tx_channel, jammers[i].observed_tx_power, jammers[i].observed_noise):
                    jammers[i].observed_reward = REWARD_SUCCESSFUL
                else:
                    jammers[i].observed_reward = REWARD_UNSUCCESSFUL
                
                if jammers[i].observed_reward >= 0:
                    jammers[i].num_jammed += 1

        # Set the state for the next iteration based on the new observed power spectrum
        tx_state = copy.deepcopy(tx_next_state)
        rx_state = copy.deepcopy(rx_next_state)
        for i in range(len(jammers)):
            jammer_states[i] = copy.deepcopy(jammer_next_states[i])

    probability_tx_channel_selected = num_tx_channel_selected / np.sum(num_tx_channel_selected)
    probability_rx_channel_selected = num_rx_channel_selected / np.sum(num_rx_channel_selected)
    probability_jammer_channel_selected = []
    for i in range(len(jammers)):
        probability_jammer_channel_selected.append(num_jammer_channel_selected[i] / np.sum(num_jammer_channel_selected[i]))

    return num_successful_transmissions, probability_tx_channel_selected, probability_rx_channel_selected, probability_jammer_channel_selected

#################################################################################
### main()
#################################################################################

if __name__ == '__main__':

    success_rates = []
    
    num_runs = 5

    for run in range(num_runs):
        print("----------------------------------------")
        print("----------------------------------------")

        print(f"Run: {run+1}")

        tx_agent = txRNNQNAgent()
        rx_agent = rxRNNQNAgent()

        list_of_other_users = []

        # random_1 = Jammer(behavior = "random")
        # list_of_other_users.append(random_1)
        # jammer_type = "random"

        # prob_1 = Jammer(behavior = "probabilistic", weights = [5, 1, 1, 1, 3]*(NUM_CHANNELS//5))
        # list_of_other_users.append(prob_1)
        # jammer_type = "probabilistic [5, 1, 1, 1, 3]"

        # sweep_1 = Jammer(behavior = "sweep", channel = 2, sweep_interval = 1)
        # list_of_other_users.append(sweep_1)
        # jammer_type = "sweeping"

        tracking_1 = Jammer(behavior = "tracking", channel = 0)
        list_of_other_users.append(tracking_1)
        jammer_type = "tracking"

        # smart = Jammer(behavior = "smart", smart_type = "RNN")
        # list_of_other_users.append(smart)
        # jammer_type = "smart_rnn"

        # smart = Jammer(behavior = "smart", smart_type = "FNN")
        # list_of_other_users.append(smart)
        # jammer_type = "smart_fnn"

        tx_average_rewards, rx_average_rewards, jammer_average_rewards = train_dqn(tx_agent, rx_agent, list_of_other_users)
        print("Jammer average rewards size: ", len(jammer_average_rewards))

        num_successful_transmissions, prob_tx_channel, prob_rx_channel, prob_jammer_channel = test_dqn(tx_agent, rx_agent, list_of_other_users)

        print("Finished testing:")
        print("Successful transmission rate: ", (num_successful_transmissions/NUM_TEST_RUNS)*100, "%")
        success_rates.append((num_successful_transmissions/NUM_TEST_RUNS)*100)
        if jammer_type == "smart":
            print("Jamming rate: ", (smart.num_jammed/NUM_TEST_RUNS)*100, "%")

    if jammer_type == "smart":
        plot_results_smart_jammer(tx_average_rewards, rx_average_rewards, jammer_average_rewards, prob_tx_channel, prob_rx_channel, prob_jammer_channel)
    else:
        plot_results(tx_average_rewards, rx_average_rewards, prob_tx_channel, prob_rx_channel, jammer_type)
        # plot_results(tx_average_rewards, rx_average_rewards, jammer_type)

        plot_probability_selection(prob_tx_channel, prob_rx_channel, prob_jammer_channel, jammer_type)

    if num_runs > 1:
        print("Success rates: ", success_rates)
        print("Average success rate: ", np.mean(success_rates), "%")

    tx_total_params = sum(p.numel() for p in tx_agent.q_network.parameters())
    tx_total_params = sum(p.numel() for p in tx_agent.target_network.parameters())
    print(f"Number of parameters in transmitter: {tx_total_params}")

    rx_total_params = sum(p.numel() for p in rx_agent.q_network.parameters())
    rx_total_params = sum(p.numel() for p in rx_agent.target_network.parameters())
    print(f"Number of parameters in receiver: {rx_total_params}")

    # relative_path = f"Comparison/09_02/Test_1/IDDQN_performance/{NUM_EPISODES}_episodes/{NUM_CHANNELS}_channels"
    # relative_path = f"Comparison/parameter_testing/IDDQN_discount_factor/{str(GAMMA).replace('.', '_')}"
    # relative_path = f"Comparison/Basic_vs_Realistic_Sensing/tracking/Realistic_Sensing_15"
    # relative_path = f"Comparison/15_02/observation_vs_no_observation/{NUM_CHANNELS}_channels"
    # relative_path = f"Comparison/tracking_vs_smart/{NUM_CHANNELS}_channels/{jammer_type}"
    # relative_path = f"Comparison/16_02/{NUM_CHANNELS}_channels/{NUM_EPISODES}_episodes/{jammer_type}"
    # relative_path = f"Comparison/16_02/tracking_obs_vs_no_obs/{NUM_CHANNELS}_channels/no_obs"
    relative_path = f"Comparison/february_tests/18_02/jammer_tests/{jammer_type}"
    if not os.path.exists(relative_path):
        os.makedirs(relative_path)

    # np.savetxt(f"{relative_path}/average_reward_both_tx.txt", tx_average_rewards)
    # np.savetxt(f"{relative_path}/average_reward_both_rx.txt", rx_average_rewards)
    # np.savetxt(f"{relative_path}/average_reward_jammer.txt", jammer_average_rewards)
    # np.savetxt(f"{relative_path}/success_rates.txt", success_rates)

    # np.savetxt(f"{relative_path}/all_success_rates.txt", success_rates)
    # np.savetxt(f"{relative_path}/average_success_rate.txt", [np.mean(success_rates)])