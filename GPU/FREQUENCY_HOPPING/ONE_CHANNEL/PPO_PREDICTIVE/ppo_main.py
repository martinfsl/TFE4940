import numpy as np
import random
# random.seed(1488)

import os

from collections import defaultdict

import matplotlib.pyplot as plt

from plotting import *
from saving_plots import *

from constants import *

from tqdm import tqdm

import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler
import time

from tx_agent import txPPOAgent
from rx_agent import rxPPOAgent
from jammer import Jammer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
print("Device: ", device)

#################################################################################
### Calculate the reward using SINR
#################################################################################

def received_signal_rx(tx_channel, rx_channel, received_power, channel_noise_receiver):
    sinr = 10*torch.log10(received_power / channel_noise_receiver[rx_channel]) # SINR at the receiver

    return (sinr > SINR_THRESHOLD) and (tx_channel == rx_channel)

def sensed_signal_rx(tx_channel, rx_sense_channels, received_power, channel_noise_receiver):
    sinr = 10*torch.log10(received_power / channel_noise_receiver[tx_channel])

    if (tx_channel in rx_sense_channels) and (sinr > SINR_THRESHOLD):
        return tx_channel
    else:
        return -1
    
def sensed_signal_tx(rx_channel, tx_sense_channels, received_power, channel_noise_transmitter):
    sinr = 10*torch.log10(received_power / channel_noise_transmitter[rx_channel])

    if (rx_channel in tx_sense_channels) and (sinr > SINR_THRESHOLD):
        return rx_channel
    else:
        return -1

def received_signal_tx(tx_channel, rx_channel, received_power, channel_noise_transmitter):
    sinr = 10*torch.log10(received_power / channel_noise_transmitter[tx_channel]) # SINR at the receiver

    return (sinr > SINR_THRESHOLD) and (tx_channel == rx_channel)

def sensed_signal_jammer(jammer_channel, tx_channel, jammer_power, channel_noise_jammer):
    sinr = 10*torch.log10(jammer_power / channel_noise_jammer[jammer_channel]) # SINR at the jammer

    return (sinr > SINR_THRESHOLD) and (jammer_channel == tx_channel)

#################################################################################
### Train the DQN, extended state space
#################################################################################

def train_ppo(tx_agent, rx_agent, jammers):
    IS_SMART_JAMMER = False
    
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

        if jammer.behavior == "smart":
            IS_SMART_JAMMER = True

    ###################################
    # Initializing the first state consisting of just noise
    tx_state = torch.empty(NUM_CHANNELS, device=device)
    rx_state = torch.empty(NUM_CHANNELS, device=device)
    jammer_states = [torch.empty(NUM_CHANNELS, device=device) for _ in jammers]

    tx_transmit_channel = torch.tensor([0], device=device)
    rx_receive_channel = torch.tensor([0], device=device)
    jammer_channels = [torch.tensor([0], device=device) for _ in jammers]
    jammer_logprobs = [torch.tensor([0], device=device) for _ in jammers]
    jammer_values = [torch.tensor([0], device=device) for _ in jammers]

    # Initialize the channel noise for the current (and first) time step
    tx_channel_noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_CHANNELS,), device=device))
    rx_channel_noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_CHANNELS,), device=device))
    jammer_channel_noises = torch.empty((len(jammers), NUM_CHANNELS), device=device)
    for i, jammer in enumerate(jammers):
        noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_CHANNELS,), device=device))
        jammer_channel_noises[i] = noise
        if jammer.behavior == "smart":
            jammer.observed_noise = noise.clone()

    # Set the current state based on the total power spectrum
    tx_state = tx_channel_noise.clone()
    rx_state = rx_channel_noise.clone()
    for i in range(len(jammers)):
        jammer_states[i] = jammer_channel_noises[i].clone()
    ###################################

    # Printing FH pattern 1
    print("Tx FH pattern 1: ", tx_agent.fh.get_pattern(0))
    print("Rx FH pattern 1: ", rx_agent.fh.get_pattern(0))
    print("------")
    print("Tx FH pattern: ", tx_agent.fh.print_patterns())
    print("Rx FH pattern: ", rx_agent.fh.print_patterns())

    for episode in tqdm(range(NUM_EPISODES)):
        # The agents chooses an action based on the current state
        tx_observation_without_pred_action = tx_agent.get_observation(tx_state, tx_transmit_channel)
        tx_observation = tx_agent.concat_predicted_action(tx_observation_without_pred_action)
        tx_channels, tx_prob_action, tx_value = tx_agent.choose_action(tx_observation)
        tx_transmit_channel = tx_channels[0].unsqueeze(0)
        tx_agent.add_previous_action(tx_transmit_channel)
        tx_sense_channels = tx_channels[1:]

        rx_observation_without_pred_action = rx_agent.get_observation(rx_state, rx_receive_channel)
        rx_observation = rx_agent.concat_predicted_action(rx_observation_without_pred_action)
        rx_channels, rx_prob_action, rx_value = rx_agent.choose_action(rx_observation)
        rx_receive_channel = rx_channels[0].unsqueeze(0)
        rx_agent.add_previous_action(rx_receive_channel)
        rx_sense_channels = rx_channels[1:]

        if IS_SMART_JAMMER:
            jammer_observations = torch.empty((len(jammers), NUM_JAMMER_SENSE_CHANNELS+1), device=device)
            # jammer_observations = torch.empty((len(jammers), NUM_CHANNELS+1), device=device)
        else:
            jammer_observations = torch.empty((len(jammers), NUM_CHANNELS), device=device)
        for i in range(len(jammers)):
            if jammers[i].behavior == "smart" and jammers[i].type == "PPO":
                jammer_observation = jammers[i].get_observation(jammer_states[i], jammer_channels[i])
                jammer_observations[i] = jammer_observation
                jammer_channel, jammer_logprobs[i], jammer_values[i] = jammers[i].choose_action(jammer_observation, tx_transmit_channel)
                jammer_channel = jammer_channel.unsqueeze(0)
                jammer_channels[i] = jammer_channel.long()
            else:
                jammer_observation = jammers[i].get_observation(jammer_states[i], jammer_channels[i])
                jammer_observations[i] = jammer_observation
                jammer_channels[i] = jammers[i].choose_action(jammer_observation, tx_transmit_channel).unsqueeze(0)

        # Set a new channel noise for the next state
        tx_channel_noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_CHANNELS,), device=device))
        rx_channel_noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_CHANNELS,), device=device))
        jammer_channel_noises = torch.empty((len(jammers), NUM_CHANNELS), device=device)
        for i, jammer in enumerate(jammers):
            noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_CHANNELS,), device=device))
            jammer_channel_noises[i] = noise
            if jammer.behavior == "smart":
                jammer.observed_noise = noise.clone()

        # Add the power of the jammers to the channel noise for Tx and Rx
        # Add the power from the Tx to the jammers as well
        for i in range(len(jammers)):
            tx_channel_noise[jammer_channels[i]] += jammers[i].get_transmit_power(direction = "transmitter")
            rx_channel_noise[jammer_channels[i]] += jammers[i].get_transmit_power(direction = "receiver")
            jammers[i].observed_tx_power = tx_agent.get_transmit_power(direction = "jammer")
            jammer_channel_noises[i][tx_transmit_channel] += jammers[i].observed_tx_power

        # Add the new observed power spectrum to the next state
        tx_next_state = tx_channel_noise.clone()
        tx_next_observation = tx_agent.get_observation(tx_next_state, tx_transmit_channel)
        rx_next_state = rx_channel_noise.clone()
        rx_next_observation = rx_agent.get_observation(rx_next_state, rx_receive_channel)
        jammer_next_states = torch.empty((len(jammers), NUM_CHANNELS), device=device)
        if IS_SMART_JAMMER:
            jammer_next_observations = torch.empty((len(jammers), NUM_JAMMER_SENSE_CHANNELS+1), device=device)
            # jammer_next_observations = torch.empty((len(jammers), NUM_CHANNELS+1), device=device)
        else:
            jammer_next_observations = torch.empty((len(jammers), NUM_CHANNELS), device=device)
        for i in range(len(jammers)):
            jammer_next_states[i] = jammer_channel_noises[i].clone()
            jammer_next_observations[i] = jammers[i].get_observation(jammer_next_states[i], jammer_channels[i])

        # Calculate the reward based on the action taken
        # ACK is sent from the receiver
        rx_observed_power = tx_agent.get_transmit_power(direction = "receiver")
        tx_observed_power = rx_agent.get_transmit_power(direction = "transmitter")
        if received_signal_rx(tx_transmit_channel, rx_receive_channel, rx_observed_power, rx_channel_noise):
            rx_reward = REWARD_SUCCESSFUL

            # Update the predictive network in the Rx agent
            rx_agent.pred_agent.store_experience_in(rx_observation_without_pred_action, rx_receive_channel)

            # ACK is received at the transmitter
            if received_signal_tx(tx_transmit_channel, rx_receive_channel, tx_observed_power, tx_channel_noise): # power should be changed to rx_agent later
                tx_reward = REWARD_SUCCESSFUL

                # Update the predictive network in the Tx agent
                # tx_agent.pred_agent.store_experience_in(tx_observation, torch.tensor(tx_transmit_channel, device=device))
                tx_agent.pred_agent.store_experience_in(tx_observation_without_pred_action, tx_transmit_channel)
            else:
                tx_reward = REWARD_INTERFERENCE

                # tx_sensing = sensed_signal_tx(rx_receive_channel, tx_sense_channels, tx_observed_power, tx_channel_noise)
                # if tx_sensing != -1:
                #     tx_agent.pred_agent.store_experience_in(tx_observation_without_pred_action, torch.tensor(tx_sensing, device=device))

                #     tx_reward += REWARD_SENSE
        else:
            rx_reward = REWARD_INTERFERENCE
            tx_reward = REWARD_INTERFERENCE

            rx_sensing = sensed_signal_rx(tx_transmit_channel, rx_sense_channels, rx_observed_power, rx_channel_noise)
            # Managed to sense the Tx but not receive the signal
            if rx_sensing != -1:
                rx_agent.pred_agent.store_experience_in(rx_observation_without_pred_action, torch.tensor(rx_sensing, device=device))

                rx_reward += REWARD_SENSE

            tx_sensing = sensed_signal_tx(rx_receive_channel, tx_sense_channels, tx_observed_power, tx_channel_noise)
            # Managed to sense the Rx but not receive the signal
            if tx_sensing != -1:
                tx_agent.pred_agent.store_experience_in(tx_observation_without_pred_action, torch.tensor(tx_sensing, device=device))

                tx_reward += REWARD_SENSE

        # If tx_transmit_channel is in tx_agent.previous_actions, then the reward is penalized
        if tx_transmit_channel in tx_agent.previous_actions:
            tx_reward -= PENALTY_NONDIVERSE
        else:
            tx_reward += REWARD_DIVERSE
        # If rx_receive_channel is in rx_agent.previous_actions, then the reward is penalized
        if rx_receive_channel in rx_agent.previous_actions:
            rx_reward -= PENALTY_NONDIVERSE
        else:
            rx_reward += REWARD_DIVERSE

        for i in range(len(jammers)):
            if jammers[i].behavior == "smart":
                if sensed_signal_jammer(jammer_channels[i], tx_transmit_channel, jammers[i].observed_tx_power, jammers[i].observed_noise):
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
        # tx_agent.store_experience_in(tx_observation, tx_transmit_channel, torch.tensor([tx_reward], device=device), tx_next_observation)
        tx_agent.store_in_memory(tx_observation, tx_transmit_channel, tx_prob_action, torch.tensor([tx_reward], device=device), tx_value)
        rx_agent.store_in_memory(rx_observation, rx_receive_channel, rx_prob_action, torch.tensor([rx_reward], device=device), rx_value)

        for i in range(len(jammers)):
            if jammers[i].behavior == "smart" and (jammers[i].type == "RNN" or jammers[i].type == "FNN"):
                jammers[i].agent.store_experience_in(jammer_observations[i], jammer_channels[i], torch.tensor([jammers[i].observed_reward], device=device), jammer_next_observations[i])
            elif jammers[i].behavior == "smart" and jammers[i].type == "PPO":
                jammers[i].agent.store_in_memory(jammer_observations[i], jammer_channels[i], jammer_logprobs[i], torch.tensor([jammers[i].observed_reward], device=device), jammer_values[i])

        # Only changing the policy after a certain number of episodes, trajectory length T
        if (episode+1) % (T+1) == T:
            # tx_agent.update()
            # rx_agent.update()
            tx_agent.update_epochs_random()
            rx_agent.update_epochs_random()
            # tx_agent.update_epochs_sequential()
            # rx_agent.update_epochs_sequential()

            tx_agent.pred_agent.train()
            rx_agent.pred_agent.train()

            for i in range(len(jammers)):
                if jammers[i].behavior == "smart" and jammers[i].type == "PPO":
                    jammers[i].agent.update()

        # Update for each episode if the jammer uses Q-learning
        for i in range(len(jammers)):
            if jammers[i].behavior == "smart" and (jammers[i].type == "RNN" or jammers[i].type == "FNN"):
                jammers[i].agent.update()

        # Periodic update of the target Q-network
        if episode % 10 == 0:
            for i in range(len(jammers)):
                if jammers[i].behavior == "smart" and (jammers[i].type == "RNN" or jammers[i].type == "FNN"):
                    jammers[i].agent.update_target_q_network()

        tx_state = tx_next_state.clone()
        rx_state = rx_next_state.clone()
        for i in range(len(jammers)):
            jammer_states[i] = jammer_next_states[i].clone()

    print("Training complete")

    return tx_average_rewards, rx_average_rewards, jammer_average_rewards

#################################################################################
### Test the DQN, extended state space
#################################################################################

def test_ppo(tx_agent, rx_agent, jammers):
    IS_SMART_JAMMER = False

    print("Testing")
    num_successful_transmissions = 0
    num_tx_channel_selected = np.zeros(NUM_CHANNELS)
    num_rx_channel_selected = np.zeros(NUM_CHANNELS)
    num_jammer_channel_selected = np.zeros((len(jammers), NUM_CHANNELS))

    # Initialize the start channel for the sweep
    for jammer in jammers:
        if jammer.behavior == "sweep":
            jammer.index_sweep = jammer.channel
        
        if jammer.behavior == "smart":
            IS_SMART_JAMMER = True

    ###################################
    # Initializing the first state
    tx_state = torch.empty(NUM_CHANNELS, device=device)
    rx_state = torch.empty(NUM_CHANNELS, device=device)
    jammer_states = [torch.empty(NUM_CHANNELS, device=device) for _ in jammers]

    tx_transmit_channel = torch.tensor([0], device=device)
    rx_receive_channel = torch.tensor([0], device=device)
    jammer_channels = [torch.tensor([0], device=device) for _ in jammers]

    # Initialize the channel noise for the current (and first) time step
    tx_channel_noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_CHANNELS,), device=device))
    rx_channel_noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_CHANNELS,), device=device))
    jammer_channel_noises = []
    for jammer in jammers:
        noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_CHANNELS,), device=device))
        jammer_channel_noises.append(noise)
        if jammer.behavior == "smart":
            jammer.observed_noise = noise.clone()

    # Set the current state based on the total power spectrum
    tx_state = tx_channel_noise.clone()
    rx_state = rx_channel_noise.clone()
    for i in range(len(jammers)):
        jammer_states[i] = jammer_channel_noises[i].clone()
    ###################################

    for run in tqdm(range(NUM_TEST_RUNS)):
        # The agent chooses an action based on the current state
        tx_observation_without_pred_action = tx_agent.get_observation(tx_state, tx_transmit_channel)
        tx_observation = tx_agent.concat_predicted_action(tx_observation_without_pred_action)
        tx_channels, _, _ = tx_agent.choose_action(tx_observation)
        tx_transmit_channel = tx_channels[0].unsqueeze(0)
        # tx_sense_channels = tx_channels[1:]

        rx_observation_without_pred_action = rx_agent.get_observation(rx_state, rx_receive_channel)
        rx_observation = rx_agent.concat_predicted_action(rx_observation_without_pred_action)
        rx_channels, _, _ = rx_agent.choose_action(rx_observation)
        rx_receive_channel = rx_channels[0].unsqueeze(0)
        # rx_sense_channels = rx_channels[1:]

        if IS_SMART_JAMMER:
            jammer_observations = torch.empty((len(jammers), NUM_JAMMER_SENSE_CHANNELS+1), device=device)
            # jammer_observations = torch.empty((len(jammers), NUM_CHANNELS+1), device=device)
        else:
            jammer_observations = torch.empty((len(jammers), NUM_CHANNELS), device=device)
        for i in range(len(jammers)):
            if jammers[i].behavior == "smart" and jammers[i].type == "PPO":
                jammer_observation = jammers[i].get_observation(jammer_states[i], jammer_channels[i])
                jammer_observations[i] = jammer_observation
                jammer_channel, _, _ = jammers[i].choose_action(jammer_observation, tx_transmit_channel)
                jammer_channel = jammer_channel.unsqueeze(0)
                jammer_channels[i] = jammer_channel.long()
            else:
                jammer_observation = jammers[i].get_observation(jammer_states[i], jammer_channels[i])
                jammer_observations[i] = jammer_observation
                jammer_channels[i] = jammers[i].choose_action(jammer_observation, tx_transmit_channel).unsqueeze(0)

        num_tx_channel_selected[tx_transmit_channel] += 1
        num_rx_channel_selected[rx_receive_channel] += 1
        for i in range(len(jammers)):
            num_jammer_channel_selected[i][jammer_channels[i]] += 1

        # Set a new channel noise for the next state
        tx_channel_noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_CHANNELS,), device=device))
        rx_channel_noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_CHANNELS,), device=device))
        jammer_channel_noises = []
        for jammer in jammers:
            noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_CHANNELS,), device=device))
            jammer_channel_noises.append(noise)
            if jammer.behavior == "smart":
                jammer.observed_noise = noise.clone()

        # Add the power of the jammers to the channel noise for Tx and Rx
        # Add the power from the Tx to the jammers as well
        for i in range(len(jammers)):
            tx_channel_noise[jammer_channels[i]] += jammers[i].get_transmit_power(direction = "transmitter")
            rx_channel_noise[jammer_channels[i]] += jammers[i].get_transmit_power(direction = "receiver")
            jammers[i].observed_tx_power = tx_agent.get_transmit_power(direction = "jammer")
            jammer_channel_noises[i][tx_transmit_channel] += jammers[i].observed_tx_power

        # Add the new observed power spectrum to the next state
        tx_next_state = tx_channel_noise.clone()
        # tx_next_observation = tx_agent.get_observation(tx_next_state, tx_transmit_channel)
        rx_next_state = rx_channel_noise.clone()
        # rx_next_observation = rx_agent.get_observation(rx_next_state, rx_receive_channel)
        jammer_next_states = torch.empty((len(jammers), NUM_CHANNELS), device=device)
        if IS_SMART_JAMMER:
            jammer_next_observations = torch.empty((len(jammers), NUM_JAMMER_SENSE_CHANNELS+1), device=device)
            # jammer_next_observations = torch.empty((len(jammers), NUM_CHANNELS+1), device=device)
        else:
            jammer_next_observations = torch.empty((len(jammers), NUM_CHANNELS), device=device)
        for i in range(len(jammers)):
            jammer_next_states[i] = jammer_channel_noises[i].clone()
            jammer_next_observations[i] = jammers[i].get_observation(jammer_next_states[i], jammer_channels[i])

        # Calculate the reward based on the action taken
        # ACK is sent from the receiver
        if received_signal_rx(tx_transmit_channel, rx_receive_channel, tx_agent.get_transmit_power(direction = "receiver"), rx_channel_noise):
            # rx_reward = REWARD_SUCCESSFUL
            
            # ACK is received at the transmitter
            if received_signal_tx(tx_transmit_channel, rx_receive_channel, rx_agent.get_transmit_power(direction = "transmitter"), tx_channel_noise): # power should be changed to rx_agent later
                tx_reward = REWARD_SUCCESSFUL
            else:
                tx_reward = REWARD_INTERFERENCE
        else:
            # rx_reward = REWARD_INTERFERENCE
            tx_reward = REWARD_INTERFERENCE
        # If the tx_reward is positive, then both the communication link was successful both ways
        if tx_reward >= 0:
            num_successful_transmissions += 1

        for i in range(len(jammers)):
            if jammers[i].behavior == "smart":
                if sensed_signal_jammer(jammer_channels[i], tx_transmit_channel, jammers[i].observed_tx_power, jammers[i].observed_noise):
                    jammers[i].observed_reward = REWARD_SUCCESSFUL
                else:
                    jammers[i].observed_reward = REWARD_UNSUCCESSFUL
                
                if jammers[i].observed_reward >= 0:
                    jammers[i].num_jammed += 1

        # Set the state for the next iteration based on the new observed power spectrum
        tx_state = tx_next_state.clone()
        rx_state = rx_next_state.clone()
        for i in range(len(jammers)):
            jammer_states[i] = jammer_next_states[i].clone()

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

    # relative_path = f"Comparison/february_tests/PPO_parameter_tuning/discount_factor/{str(GAMMA).replace('.', '_')}"
    # relative_path = f"Comparison/february_tests/PPO_parameter_tuning/trajectory_length/{str(T).replace('.', '_')}"
    # relative_path = f"Comparison/PPO_tests/update_random/further_tests/gamma/{str(GAMMA).replace('.', '_')}"
    # relative_path = f"Comparison/PPO_tests/update_random/further_tests/lambda_param/{str(LAMBDA).replace('.', '_')}"
    # relative_path = f"Comparison/march_tests/PPO/ppo_smart_jammer/12_03"
    # relative_path = f"Comparison/march_tests/PPO/ppo_smart_jammer/12_03/param_testing/lambda_param/{str(LAMBDA).replace('.', '_')}"
    # relative_path = f"Comparison/march_tests/PPO/ppo_smart_jammer/tracking_vs_smartppo/test_1/tracking"
    # relative_path = f"Comparison/march_tests/PPO/ppo_smart_jammer/tracking_vs_smartppo/test_1/smart_rnn"
    # relative_path = f"Comparison/march_tests/PPO/ppo_smart_jammer/tracking_vs_smartppo/test_1/smart_ppo"
    # relative_path = f"Comparison/march_tests/PPO/diversity/tracking_vs_smartppo/test_2/tracking"
    # relative_path = f"Comparison/march_tests/PPO/diversity/tracking_vs_smartppo/test_2/smart_ppo"
    # relative_path = f"Comparison/march_tests/PPO/receive_one_vs_multiple/receive_one/tracking"
    relative_path = f"Comparison/march_tests/PPO/receive_one_vs_multiple/receive_one/smart_ppo"
    if not os.path.exists(relative_path):
        os.makedirs(relative_path)

    for run in range(num_runs):
        print("----------------------------------------")
        print("----------------------------------------")

        print(f"Run: {run+1}")

        tx_agent = txPPOAgent(device=device)
        rx_agent = rxPPOAgent(device=device)

        list_of_other_users = []

        # random_1 = Jammer(behavior = "random", device = device)
        # list_of_other_users.append(random_1)
        # jammer_type = "random"
        # jammer_behavior = "naive"

        # prob_1 = Jammer(behavior = "probabilistic", weights = [5, 1, 1, 1, 3]*(NUM_CHANNELS//5), device = device)
        # list_of_other_users.append(prob_1)
        # jammer_type = "probabilistic [5, 1, 1, 1, 3]"
        # jammer_behavior = "naive"

        # sweep_1 = Jammer(behavior = "sweep", channel = 2, sweep_interval = 1, device = device)
        # list_of_other_users.append(sweep_1)
        # jammer_type = "sweeping"
        # jammer_behavior = "naive"

        # tracking_1 = Jammer(behavior = "tracking", channel = 0, device = device)
        # list_of_other_users.append(tracking_1)
        # jammer_type = "tracking"
        # jammer_behavior = "naive"

        # smart = Jammer(behavior = "smart", smart_type = "RNN", device = device)
        # list_of_other_users.append(smart)
        # jammer_type = "smart_rnn"
        # jammer_behavior = "smart"

        # smart = Jammer(behavior = "smart", smart_type = "FNN", device = device)
        # list_of_other_users.append(smart)
        # jammer_type = "smart_fnn"
        # jammer_behavior = "smart"

        smart = Jammer(behavior = "smart", smart_type = "PPO", device = device)
        list_of_other_users.append(smart)
        jammer_type = "smart_ppo"
        jammer_behavior = "smart"

        tx_average_rewards, rx_average_rewards, jammer_average_rewards = train_ppo(tx_agent, rx_agent, list_of_other_users)
        print("Jammer average rewards size: ", len(jammer_average_rewards))

        num_successful_transmissions, prob_tx_channel, prob_rx_channel, prob_jammer_channel = test_ppo(tx_agent, rx_agent, list_of_other_users)

        print("Finished testing:")
        print("Successful transmission rate: ", (num_successful_transmissions/NUM_TEST_RUNS)*100, "%")
        success_rates.append((num_successful_transmissions/NUM_TEST_RUNS)*100)
        if jammer_type == "smart":
            print("Jamming rate: ", (smart.num_jammed/NUM_TEST_RUNS)*100, "%")

        relative_path_run = f"{relative_path}/{run+1}"
        if not os.path.exists(relative_path_run):
            os.makedirs(relative_path_run)

        # Save data in textfiles
        np.savetxt(f"{relative_path_run}/average_reward_tx.txt", tx_average_rewards)
        np.savetxt(f"{relative_path_run}/actor_losses_tx.txt", tx_agent.actor_losses.cpu().detach().numpy())
        np.savetxt(f"{relative_path_run}/critic_losses_tx.txt", tx_agent.critic_losses.cpu().detach().numpy())
        np.savetxt(f"{relative_path_run}/average_reward_rx.txt", rx_average_rewards)
        np.savetxt(f"{relative_path_run}/actor_losses_rx.txt", rx_agent.actor_losses.cpu().detach().numpy())
        np.savetxt(f"{relative_path_run}/critic_losses_rx.txt", rx_agent.critic_losses.cpu().detach().numpy())
        np.savetxt(f"{relative_path_run}/average_reward_jammer.txt", jammer_average_rewards)
        np.savetxt(f"{relative_path_run}/success_rates.txt", [(num_successful_transmissions/NUM_TEST_RUNS)*100])

        save_results_plot(tx_average_rewards, rx_average_rewards, prob_tx_channel, prob_rx_channel, jammer_type, filepath = relative_path_run)
        save_probability_selection(prob_tx_channel, prob_rx_channel, prob_jammer_channel, jammer_type, filepath = relative_path_run)
        save_results_losses(tx_agent.actor_losses.cpu().detach().numpy(), tx_agent.critic_losses.cpu().detach().numpy(), 
                            rx_agent.actor_losses.cpu().detach().numpy(), rx_agent.critic_losses.cpu().detach().numpy(), filepath = relative_path_run)
        save_channel_selection_training(tx_agent.channels_selected.cpu().detach().numpy(), rx_agent.channels_selected.cpu().detach().numpy(), 
                                        list_of_other_users[0].channels_selected.cpu().detach().numpy(), filepath = relative_path_run)
        if jammer_behavior == "smart":
            save_jammer_results_plot(jammer_average_rewards, filepath = relative_path_run)

    # if jammer_type == "smart":
    #     plot_results_smart_jammer(tx_average_rewards, rx_average_rewards, jammer_average_rewards, prob_tx_channel, prob_rx_channel, prob_jammer_channel)
    # else:
    #     plot_results(tx_average_rewards, rx_average_rewards, prob_tx_channel, prob_rx_channel, jammer_type)
    #     # plot_results(tx_average_rewards, rx_average_rewards, jammer_type)

    #     plot_probability_selection(prob_tx_channel, prob_rx_channel, prob_jammer_channel, jammer_type)

    if num_runs > 1:
        print("Success rates: ", success_rates)
        print("Average success rate: ", np.mean(success_rates), "%")

    np.savetxt(f"{relative_path}/all_success_rates.txt", success_rates)
    np.savetxt(f"{relative_path}/average_success_rate.txt", [np.mean(success_rates)])