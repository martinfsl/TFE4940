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
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

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
    if tx_channel in rx_channel:
        same_channel = True
        
        sinr = 10*torch.log10(received_power / channel_noise_receiver[tx_channel]) # SINR at the receiver
    else:
        same_channel = False
        sinr = torch.tensor(-1000, device=device)

    return (sinr > SINR_THRESHOLD) and (same_channel)

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
    if tx_channel in rx_channel:
        same_channel = True
        
        sinr = 10*torch.log10(received_power / channel_noise_transmitter[tx_channel])
    else:
        same_channel = False
        sinr = torch.tensor(-1000, device=device)

    return (sinr > SINR_THRESHOLD) and (same_channel)

def sensed_signal_jammer(jammer_channel, tx_channel, jammer_power, channel_noise_jammer):
    sinr = 10*torch.log10(jammer_power / channel_noise_jammer[jammer_channel]) # SINR at the jammer

    return (sinr > SINR_THRESHOLD) and (jammer_channel == tx_channel)

#################################################################################
### Train the DQN, extended state space
#################################################################################

def train_ppo(tx_agent, rx_agent, jammers):
    IS_SMART_OR_GENIE_JAMMER = False
    IS_TRACKING_JAMMER = False
    
    print("Training")
    tx_accumulated_rewards = []
    tx_average_rewards = []

    rx_accumulated_rewards = []
    rx_average_rewards = []

    jammer_accumulated_rewards = []
    jammer_average_rewards = []

    num_successful_transmissions = 0
    num_jammed = 0
    num_fading = 0
    num_missed = 0
    num_tx_corr_pred = 0
    num_rx_corr_pred = 0

    # Initialize the start channel for the sweep
    for jammer in jammers:
        if jammer.behavior == "sweep":
            jammer.index_sweep = jammer.channel

        if jammer.behavior == "smart" or jammer.behavior == "genie":
            IS_SMART_OR_GENIE_JAMMER = True
        elif jammer.behavior == "tracking":
            IS_TRACKING_JAMMER = True

    ###################################
    # Initializing the first state consisting of just noise
    tx_state = torch.empty(STATE_SPACE_SIZE, device=device)
    rx_state = torch.empty(STATE_SPACE_SIZE, device=device)
    jammer_states = [torch.empty(NUM_CHANNELS, device=device) for _ in jammers]
    
    # Intializing the first channels
    tx_transmit_channel = torch.tensor([0], device=device)
    rx_receive_channel = torch.tensor([0], device=device)

    tx_seed = torch.tensor([0], device=device)
    rx_seed = torch.tensor([0], device=device)
    tx_hops = torch.tensor(NUM_HOPS*[0], device=device)
    rx_hops = torch.tensor(NUM_HOPS*[0], device=device)

    tx_observed_rx_seed = torch.tensor([-1], device=device)
    rx_observed_tx_seed = torch.tensor([-1], device=device)

    jammer_channels = [torch.tensor([0], device=device) for _ in jammers]
    jammer_logprobs = [torch.tensor([0], device=device) for _ in jammers]
    jammer_values = [torch.tensor([0], device=device) for _ in jammers]
    jammer_seeds = [torch.tensor([0], device=device) for _ in jammers]

    # Initialize the channel noise for the current (and first) time step
    # tx_channel_noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_CHANNELS,), device=device))
    # rx_channel_noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_CHANNELS,), device=device))
    tx_channel_noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_HOPS, NUM_CHANNELS), device=device))
    rx_channel_noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_HOPS, NUM_CHANNELS), device=device))
    jammer_channel_noises = torch.empty((len(jammers), NUM_CHANNELS), device=device)
    for i, jammer in enumerate(jammers):
        noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_CHANNELS,), device=device))
        jammer_channel_noises[i] = noise
        if jammer.behavior == "smart" or jammer.behavior == "genie":
            jammer.observed_noise = noise.clone()

    # Set the current state based on the total power spectrum
    tx_state = tx_channel_noise.clone()
    rx_state = rx_channel_noise.clone()
    for i in range(len(jammers)):
        jammer_states[i] = jammer_channel_noises[i].clone()
    ###################################

    for episode in tqdm(range(int(NUM_EPISODES/NUM_HOPS))):
        # The agents chooses an action based on the current state
        tx_observation = tx_agent.get_observation(tx_state, tx_hops, tx_seed)
        if USE_PREDICTION:
            tx_pred_observation = tx_agent.get_pred_observation()
        else:
            tx_pred_observation = torch.tensor([], device=device)
        tx_seed, tx_prob_action, tx_value, tx_sense_seeds, tx_pred_rx_action = tx_agent.choose_action(tx_observation, tx_pred_observation)
        tx_agent.add_previous_seed(tx_seed)

        rx_observation = rx_agent.get_observation(rx_state, rx_hops, rx_seed)
        if USE_PREDICTION:
            rx_pred_observation = rx_agent.get_pred_observation()
        else:
            rx_pred_observation = torch.tensor([], device=device)
        rx_seed, rx_prob_action, rx_value, rx_additional_seeds, \
            rx_prob_additional_actions, rx_sense_seeds, rx_pred_tx_action = rx_agent.choose_action(rx_observation, rx_pred_observation)
        rx_agent.add_previous_seed(rx_seed)

        if USE_PREDICTION and USE_STANDALONE_BELIEF_MODULE:
            if tx_seed == rx_pred_tx_action:
                num_rx_corr_pred += 1
            if (rx_seed == tx_pred_rx_action) or (tx_pred_rx_action in rx_additional_seeds):
                num_tx_corr_pred += 1

        # print("--------------------------------")
        # print("Episode - ", episode)
        # print("tx_seed:", tx_seed)
        # print("tx_sense_seeds: ", tx_sense_seeds)
        # print("rx_seed:", rx_seed)
        # print("rx_additional_seeds: ", rx_additional_seeds)
        # print("rx_sense_seeds: ", rx_sense_seeds)

        # print("tx_pred_rx_action: ", tx_pred_rx_action)
        # print("rx_pred_tx_action: ", rx_pred_tx_action)

        tx_agent.fh.generate_sequence()
        tx_hops = tx_agent.fh.get_sequence(tx_seed)
        tx_sense_hops = torch.tensor([], device=device)
        for seed in tx_sense_seeds:
            tx_sense_hops = torch.cat((tx_sense_hops, tx_agent.fh.get_sequence(seed.unsqueeze(0)).unsqueeze(0)), dim=0)
        rx_agent.fh.generate_sequence()
        rx_hops = rx_agent.fh.get_sequence(rx_seed)
        rx_additional_hops = torch.tensor([], device=device)
        for seed in rx_additional_seeds:
            rx_additional_hops = torch.cat((rx_additional_hops, rx_agent.fh.get_sequence(seed.unsqueeze(0)).unsqueeze(0)), dim=0)
        rx_sense_hops = torch.tensor([], device=device)
        for seed in rx_sense_seeds:
            rx_sense_hops = torch.cat((rx_sense_hops, rx_agent.fh.get_sequence(seed.unsqueeze(0)).unsqueeze(0)), dim=0)

        # print("tx_hops: ", tx_hops)
        # print("tx_sense_hops: ", tx_sense_hops)
        # print("rx_hops: ", rx_hops)
        # print("rx_additional_hops: ", rx_additional_hops)
        # print("rx_sense_hops: ", rx_sense_hops)
        # print()
        # print("rx_prob_action: ", rx_prob_action)
        # print("rx_prob_additional_actions: ", rx_prob_additional_actions)
        # print()

        # Arrays for storing Tx and Rx observation of each others selected channels
        tx_observed_rx = torch.tensor([], device=device)
        rx_observed_tx = torch.tensor([], device=device)

        # Set a new channel noise for the next state
        tx_channel_noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_HOPS, NUM_CHANNELS), device=device))
        rx_channel_noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_HOPS, NUM_CHANNELS), device=device))

        ##################
        # One iteration through the hopping pattern is done here

        tx_reward = 0
        rx_reward = 0

        for i in range(NUM_HOPS):
            tx_transmit_channel = tx_hops[i].unsqueeze(0)
            tx_agent.fh_seeds_used = torch.cat((tx_agent.fh_seeds_used, tx_seed), dim=0)
            tx_agent.channels_selected = torch.concat((tx_agent.channels_selected, tx_transmit_channel), dim=0)
            rx_receive_channel = rx_hops[i].unsqueeze(0)
            rx_agent.fh_seeds_used = torch.cat((rx_agent.fh_seeds_used, rx_seed), dim=0)
            rx_agent.channels_selected = torch.concat((rx_agent.channels_selected, rx_receive_channel), dim=0)

            tx_reward_hop = 0
            rx_reward_hop = 0

            rx_extra_receive_channels = torch.tensor([], device=device)
            for j in range(len(rx_additional_hops)):
                rx_extra_receive_channels = torch.cat((rx_extra_receive_channels, rx_additional_hops[j][i].unsqueeze(0)), dim=0)

            rx_receive_channels = torch.cat((rx_receive_channel, rx_extra_receive_channels))

            tx_sense_channels = torch.tensor([], device=device)
            rx_sense_channels = torch.tensor([], device=device)
            for j in range(len(tx_sense_hops)):
                tx_sense_channels = torch.cat((tx_sense_channels, tx_sense_hops[j][i].unsqueeze(0)), dim=0)
            for j in range(len(rx_sense_hops)):
                rx_sense_channels = torch.cat((rx_sense_channels, rx_sense_hops[j][i].unsqueeze(0)), dim=0)

            # print("-----------")
            # print("Episode - ", episode, " | Hop - ", i)
            # print("tx_transmit_channel:", tx_transmit_channel)
            # print("tx_sense_channels: ", tx_sense_channels)
            # print("rx_receive_channels: ", rx_receive_channels)
            # print("rx_extra_receive_channels: ", rx_extra_receive_channels)
            # print("rx_sense_channels: ", rx_sense_channels)

            if IS_SMART_OR_GENIE_JAMMER:
                jammer_observations = torch.empty((len(jammers), NUM_JAMMER_SENSE_CHANNELS+1), device=device)
                # jammer_observations = torch.empty((len(jammers), NUM_CHANNELS+1), device=device)
            elif IS_TRACKING_JAMMER:
                jammer_observations = torch.empty((len(jammers), NUM_CHANNELS), device=device)
            else:
                jammer_observations = torch.empty((len(jammers), 0))
            for j in range(len(jammers)):
                if jammers[j].behavior == "smart":
                    jammer_observation = jammers[j].get_observation(jammer_states[j], jammer_channels[j])
                    jammer_observations[j] = jammer_observation
                    jammer_channel, jammer_logprobs[j], jammer_values[j] = jammers[j].choose_action(jammer_observation, tx_transmit_channel)
                    jammer_channel = jammer_channel.unsqueeze(0)
                    jammer_channels[j] = jammer_channel.long()
                elif jammers[j].behavior == "genie":
                    jammer_observation = jammers[j].get_observation(jammer_states[j], jammer_channels[j])
                    jammer_observations[j] = jammer_observation
                    jammer_seed, jammer_logprobs[j], jammer_values[j] = jammers[j].choose_action(jammer_observation, tx_transmit_channel)
                    jammer_seeds[j] = jammer_seed
                    jammers[j].agent.fh.generate_sequence()
                    jammer_channel = jammers[j].agent.fh.get_sequence(jammer_seed.unsqueeze(0).long())
                    jammers[j].channels_selected = torch.cat((jammers[j].channels_selected, jammer_channel))
                    jammer_channels[j] = jammer_channel.long()
                else:
                    jammer_observation = jammers[j].get_observation(jammer_states[j], jammer_channels[j])
                    jammer_observations[j] = jammer_observation
                    jammer_channels[j] = jammers[j].choose_action(jammer_observation, tx_transmit_channel).unsqueeze(0)
            
            # Set a new channel noise for the next state for the jammer
            jammer_channel_noises = torch.empty((len(jammers), NUM_CHANNELS), device=device)
            for j, jammer in enumerate(jammers):
                noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_CHANNELS,), device=device))
                jammer_channel_noises[j] = noise
                if jammer.behavior == "smart" or jammer.behavior == "genie":
                    jammer.observed_noise = noise.clone()

            # Add the power of the jammers to the channel noise for Tx and Rx
            # Add the power from the Tx to the jammers as well
            for j in range(len(jammers)):
                tx_channel_noise[i][jammer_channels[j]] += jammers[j].get_transmit_power(direction = "transmitter")
                rx_channel_noise[i][jammer_channels[j]] += jammers[j].get_transmit_power(direction = "receiver")
                jammers[j].observed_tx_power = tx_agent.get_transmit_power(direction = "jammer")
                jammer_channel_noises[j][tx_transmit_channel.long()] += jammers[j].observed_tx_power

            rx_observed_power = tx_agent.get_transmit_power(direction = "receiver")
            tx_observed_power = rx_agent.get_transmit_power(direction = "transmitter")
            # Calculate the reward based on the patterns chosen
            # ACK is sent from the receiver
            success = False
            jamming = False
            fading = False
            missed = False
            if received_signal_rx(tx_transmit_channel.long(), rx_receive_channels.long(), rx_observed_power, rx_channel_noise[i]):
                rx_reward_hop = REWARD_SUCCESSFUL

                rx_observed_tx = torch.concat((rx_observed_tx, tx_transmit_channel), dim=0)

                # ACK is received at the transmitter
                if received_signal_tx(tx_transmit_channel.long(), rx_receive_channels.long(), tx_observed_power, tx_channel_noise[i]):
                    tx_reward_hop = REWARD_SUCCESSFUL
                    success = True

                    tx_observed_rx = torch.concat((tx_observed_rx, tx_transmit_channel), dim=0)
                else:
                    # tx_reward += REWARD_INTERFERENCE
                    tx_reward_hop = REWARD_MISS
                    
                    if tx_transmit_channel in jammer_channels:
                        jamming = True
                    else:
                        fading = True

                    tx_observed_rx = torch.concat((tx_observed_rx, torch.tensor([-1], device=device)), dim=0)
            else:
                rx_reward_hop = REWARD_MISS
                tx_reward_hop = REWARD_MISS
                # rx_reward += REWARD_INTERFERENCE
                # tx_reward += REWARD_INTERFERENCE

                # Rx might still be able to sense the Tx's action even though it did not receive the message
                rx_sensing = sensed_signal_rx(tx_transmit_channel.long(), rx_sense_channels, rx_observed_power, rx_channel_noise[i])
                if rx_sensing != -1:
                    # print("rx_sensing: ", rx_sensing)
                    rx_observed_tx = torch.concat((rx_observed_tx, rx_sensing), dim=0)
                else:
                    rx_observed_tx = torch.concat((rx_observed_tx, torch.tensor([-1], device=device)), dim=0)

                # Tx might still be able to sense the Rx's action even though it did not receive the ACK
                tx_sensing = sensed_signal_tx(rx_receive_channel.long(), tx_sense_channels, tx_observed_power, tx_channel_noise[i])
                if tx_sensing != -1:
                    # print("tx_sensing: ", tx_sensing)
                    tx_observed_rx = torch.concat((tx_observed_rx, tx_sensing), dim=0)
                else:
                    tx_observed_rx = torch.concat((tx_observed_rx, torch.tensor([-1], device=device)), dim=0)

                if tx_transmit_channel in rx_receive_channels:
                    if tx_transmit_channel in jammer_channels:
                        jamming = True
                    else:
                        fading = True
                else:
                    missed = True

            if success:
                num_successful_transmissions += 1
            if jamming:
                num_jammed += 1
            if fading:
                num_fading += 1
            if missed:
                num_missed += 1

            for j in range(len(jammers)):
                if jammers[j].behavior == "smart" or jammers[j].behavior == "genie":
                    if sensed_signal_jammer(jammer_channels[j], tx_transmit_channel, jammers[j].observed_tx_power, jammers[j].observed_noise):
                        jammers[j].observed_reward = REWARD_SUCCESSFUL
                    else:
                        jammers[j].observed_reward = REWARD_UNSUCCESSFUL

            jammer_next_states = torch.empty((len(jammers), NUM_CHANNELS), device=device)
            if IS_SMART_OR_GENIE_JAMMER:
                jammer_next_observations = torch.empty((len(jammers), NUM_JAMMER_SENSE_CHANNELS+1), device=device)
                # jammer_observations = torch.empty((len(jammers), NUM_CHANNELS+1), device=device)
            elif IS_TRACKING_JAMMER:
                jammer_next_observations = torch.empty((len(jammers), NUM_CHANNELS), device=device)
            else:
                jammer_next_observations = torch.empty((len(jammers), 0))
            for j in range(len(jammers)):
                jammer_next_states[j] = jammer_channel_noises[j].clone()
                jammer_next_observations[j] = jammers[j].get_observation(jammer_next_states[j], jammer_channels[j])

            if episode == 0:
                for j in range(len(jammers)):
                    if jammers[j].behavior == "smart" or jammers[j].behavior == "genie":
                        jammer_accumulated_rewards.append(jammers[j].observed_reward)
                        jammer_average_rewards.append(jammers[j].observed_reward)
            else:
                for j in range(len(jammers)):
                    if jammers[j].behavior == "smart" or jammers[j].behavior == "genie":
                        jammer_accumulated_rewards.append(jammer_accumulated_rewards[-1] + jammers[j].observed_reward)
                        jammer_average_rewards.append(jammer_accumulated_rewards[-1]/len(jammer_accumulated_rewards))

            # Store the experience in the jammer's memory
            for j in range(len(jammers)):
                if jammers[j].behavior == "smart":
                    jammers[j].agent.store_in_memory(jammer_observations[j], jammer_channels[j], 
                                                    jammer_logprobs[j], torch.tensor([jammers[j].observed_reward], device=device), jammer_values[j])
                elif jammers[j].behavior == "genie":
                    jammers[j].agent.store_in_memory(jammer_observations[j], jammer_seeds[j].unsqueeze(0), 
                                                    jammer_logprobs[j], torch.tensor([jammers[j].observed_reward], device=device), jammer_values[j])
                    
            # Only changing the policy after a certain number of episodes, trajectory length T
            if ((episode*NUM_HOPS) + i + 1) % (T+1) == T:
                for j in range(len(jammers)):
                    if jammers[j].behavior == "smart" or jammers[j].behavior == "genie":
                        jammers[j].agent.update()

            for j in range(len(jammers)):
                jammer_states[j] = jammer_next_states[j].clone()

            # print()
            # print("tx_reward: ", tx_reward)
            # print("rx_reward: ", rx_reward)
            # print("-----------")

            tx_reward += tx_reward_hop
            rx_reward += rx_reward_hop

            if episode == 0 and i == 0:
                tx_accumulated_rewards.append(tx_reward_hop)
                tx_average_rewards.append(tx_reward_hop)

                rx_accumulated_rewards.append(rx_reward_hop)
                rx_average_rewards.append(rx_reward_hop)
            else:
                tx_accumulated_rewards.append(tx_accumulated_rewards[-1] + tx_reward_hop)
                tx_average_rewards.append(tx_accumulated_rewards[-1]/len(tx_accumulated_rewards))

                rx_accumulated_rewards.append(rx_accumulated_rewards[-1] + rx_reward_hop)
                rx_average_rewards.append(rx_accumulated_rewards[-1]/len(rx_accumulated_rewards))

        ##################

        tx_reward = tx_reward/NUM_HOPS
        rx_reward = rx_reward/NUM_HOPS

        # print()
        # print("tx_reward: ", tx_reward)
        # print("rx_reward: ", rx_reward)
        # print()
        # print("tx_observed_rx: ", tx_observed_rx)
        # print("rx_observed_tx: ", rx_observed_tx)

        if NUM_HOPS > 1:
            tx_observed_rx_seed = tx_agent.fh.get_seed(tx_observed_rx)
            rx_observed_tx_seed = rx_agent.fh.get_seed(rx_observed_tx)
        else:
            tx_observed_rx_seed = tx_observed_rx
            rx_observed_tx_seed = rx_observed_tx

        # print("tx_observed_rx_seed: ", tx_observed_rx_seed)
        # print("rx_observed_tx_seed: ", rx_observed_tx_seed)

        tx_agent.add_action_observation(tx_observed_rx_seed)
        rx_agent.add_action_observation(rx_observed_tx_seed)

        # Add the new observed power spectrum to the next state
        tx_next_state = tx_channel_noise.clone()
        rx_next_state = rx_channel_noise.clone()

        # If Rx chose the same seed as Tx, but as an additional action, it needs to use the correctly selected seed in the training
        if rx_reward >= ((REWARD_SUCCESSFUL+REWARD_MISS)/2) and rx_observed_tx_seed != -1:
            rx_seed_correct = rx_observed_tx_seed

            rx_prob_action_correct = rx_prob_action
            if rx_seed_correct != rx_seed:
                for i in range(len(rx_additional_seeds)):
                    if rx_additional_seeds[i] == rx_seed_correct:
                        rx_prob_action_correct = rx_prob_additional_actions[i].unsqueeze(0)
        else:
            rx_seed_correct = rx_seed
            rx_prob_action_correct = rx_prob_action

        # The hops used for the observation should be based on the correct seed
        rx_hops = rx_agent.fh.get_sequence(rx_seed_correct)
        # The seed used for the observation should correspond to this
        rx_seed = rx_seed_correct

        # print("rx_seed_correct: ", rx_seed_correct)
        # print("rx_prob_action_correct: ", rx_prob_action_correct)

        # Store the experience in the agent's memory
        tx_agent.store_in_memory(tx_observation, tx_seed, tx_prob_action, torch.tensor([tx_reward], device=device), tx_value, 
                                tx_pred_observation, tx_pred_rx_action, tx_observed_rx_seed)
        rx_agent.store_in_memory(rx_observation, rx_seed_correct, rx_prob_action_correct, torch.tensor([rx_reward], device=device), rx_value, 
                                rx_pred_observation, rx_pred_tx_action, rx_observed_tx_seed)

        # Only changing the policy after a certain number of episodes, trajectory length T
        if (episode+1) % (T+1) == T:
            tx_agent.update()
            rx_agent.update()

        tx_state = tx_next_state.clone()
        rx_state = rx_next_state.clone()

        # print("--------------------------------")

    print("Training complete")

    return tx_average_rewards, rx_average_rewards, jammer_average_rewards, \
        num_successful_transmissions, num_jammed, num_fading, num_missed, num_tx_corr_pred, num_rx_corr_pred

#################################################################################
### Test the DQN, extended state space
#################################################################################

def test_ppo(tx_agent, rx_agent, jammers):
    IS_SMART_OR_GENIE_JAMMER = False
    IS_TRACKING_JAMMER = False

    print("Testing")
    num_successful_transmissions = 0
    num_jammed = 0
    num_fading = 0
    num_missed = 0
    num_tx_successful_hop_transmissions = 0
    num_rx_successful_hop_transmissions = 0
    num_tx_corr_pred = 0
    num_rx_corr_pred = 0
    num_tx_pattern_selected = np.zeros(NUM_SEEDS)
    num_rx_pattern_selected = np.zeros(NUM_SEEDS)
    num_tx_channel_selected = np.zeros(NUM_CHANNELS)
    num_rx_channel_selected = np.zeros(NUM_CHANNELS)
    num_jammer_channel_selected = np.zeros((len(jammers), NUM_CHANNELS))

    tx_agent.channels_selected = torch.tensor([], device=tx_agent.device)
    rx_agent.channels_selected = torch.tensor([], device=rx_agent.device)
    tx_agent.fh_seeds_used = torch.tensor([], device=tx_agent.device)
    rx_agent.fh_seeds_used = torch.tensor([], device=rx_agent.device)
    for i in range(len(jammers)):
        jammers[i].channels_selected = torch.tensor([], device=jammers[i].device)
        if jammers[i].behavior == "genie":
            jammers[i].fh_seeds_used = torch.tensor([], device=jammers[i].device)

    tx_agent.predicted_actions = torch.tensor([], device=tx_agent.device)
    rx_agent.predicted_actions = torch.tensor([], device=rx_agent.device)

    # Initialize the start channel for the sweep
    for jammer in jammers:
        if jammer.behavior == "sweep":
            jammer.index_sweep = jammer.channel
        
        if jammer.behavior == "smart" or jammer.behavior == "genie":
            IS_SMART_OR_GENIE_JAMMER = True
        elif jammer.behavior == "tracking":
            IS_TRACKING_JAMMER = True

    ###################################
    # Initializing the first state
    tx_state = torch.empty(STATE_SPACE_SIZE, device=device)
    rx_state = torch.empty(STATE_SPACE_SIZE, device=device)
    jammer_states = [torch.empty(NUM_CHANNELS, device=device) for _ in jammers]

    # Intializing the first channels
    tx_transmit_channel = torch.tensor([0], device=device)
    rx_receive_channel = torch.tensor([0], device=device)

    tx_seed = torch.tensor([0], device=device)
    rx_seed = torch.tensor([0], device=device)
    tx_hops = torch.tensor(NUM_HOPS*[0], device=device)
    rx_hops = torch.tensor(NUM_HOPS*[0], device=device)

    tx_observed_rx_seed = torch.tensor([-1], device=device)
    rx_observed_tx_seed = torch.tensor([-1], device=device)

    jammer_channels = [torch.tensor([0], device=device) for _ in jammers]

    # Initialize the channel noise for the current (and first) time step
    tx_channel_noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_HOPS, NUM_CHANNELS), device=device))
    rx_channel_noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_HOPS, NUM_CHANNELS), device=device))
    jammer_channel_noises = []
    for jammer in jammers:
        noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_CHANNELS,), device=device))
        jammer_channel_noises.append(noise)
        if jammer.behavior == "smart" or jammer.behavior == "genie":
            jammer.observed_noise = noise.clone()

    # Set the current state based on the total power spectrum
    tx_state = tx_channel_noise.clone()
    rx_state = rx_channel_noise.clone()
    for i in range(len(jammers)):
        jammer_states[i] = jammer_channel_noises[i].clone()
    ###################################

    for episode in tqdm(range(int(NUM_EPISODES/NUM_HOPS))):
        # The agents chooses an action based on the current state
        tx_observation = tx_agent.get_observation(tx_state, tx_hops, tx_seed)
        if USE_PREDICTION:
            tx_pred_observation = tx_agent.get_pred_observation()
        else:
            tx_pred_observation = torch.tensor([], device=device)
        tx_seed, tx_prob_action, tx_value, tx_sense_seeds, tx_pred_rx_action = tx_agent.choose_action(tx_observation, tx_pred_observation)
        tx_agent.add_previous_seed(tx_seed)

        rx_observation = rx_agent.get_observation(rx_state, rx_hops, rx_seed)
        if USE_PREDICTION:
            rx_pred_observation = rx_agent.get_pred_observation()
        else:
            rx_pred_observation = torch.tensor([], device=device)
        rx_seed, rx_prob_action, rx_value, rx_additional_seeds, \
            rx_prob_additional_actions, rx_sense_seeds, rx_pred_tx_action = rx_agent.choose_action(rx_observation, rx_pred_observation)
        rx_agent.add_previous_seed(rx_seed)

        if USE_PREDICTION and USE_STANDALONE_BELIEF_MODULE:
            # Checking if the prediction was correct
            if tx_seed == rx_pred_tx_action:
                num_rx_corr_pred += 1
            if (rx_seed == tx_pred_rx_action) or (tx_pred_rx_action in rx_additional_seeds):
                num_tx_corr_pred += 1

        tx_agent.fh.generate_sequence()
        tx_hops = tx_agent.fh.get_sequence(tx_seed)
        tx_sense_hops = torch.tensor([], device=device)
        for seed in tx_sense_seeds:
            tx_sense_hops = torch.cat((tx_sense_hops, tx_agent.fh.get_sequence(seed.unsqueeze(0)).unsqueeze(0)), dim=0)
        rx_agent.fh.generate_sequence()
        rx_hops = rx_agent.fh.get_sequence(rx_seed)
        rx_additional_hops = torch.tensor([], device=device)
        for seed in rx_additional_seeds:
            rx_additional_hops = torch.cat((rx_additional_hops, rx_agent.fh.get_sequence(seed.unsqueeze(0)).unsqueeze(0)), dim=0)
        rx_sense_hops = torch.tensor([], device=device)
        for seed in rx_sense_seeds:
            rx_sense_hops = torch.cat((rx_sense_hops, rx_agent.fh.get_sequence(seed.unsqueeze(0)).unsqueeze(0)), dim=0)

        # Arrays for storing Tx and Rx observation of each others selected channels
        tx_observed_rx = torch.tensor([], device=device)
        rx_observed_tx = torch.tensor([], device=device)

        # Set a new channel noise for the next state
        tx_channel_noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_HOPS, NUM_CHANNELS), device=device))
        rx_channel_noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_HOPS, NUM_CHANNELS), device=device))

        ##################
        # One iteration through the hopping pattern is done here

        tx_reward = 0
        rx_reward = 0

        for i in range(NUM_HOPS):
            tx_transmit_channel = tx_hops[i].unsqueeze(0)
            tx_agent.fh_seeds_used = torch.cat((tx_agent.fh_seeds_used, tx_seed), dim=0)
            tx_agent.channels_selected = torch.concat((tx_agent.channels_selected, tx_transmit_channel), dim=0)
            rx_receive_channel = rx_hops[i].unsqueeze(0)
            rx_agent.fh_seeds_used = torch.cat((rx_agent.fh_seeds_used, rx_seed), dim=0)
            rx_agent.channels_selected = torch.concat((rx_agent.channels_selected, rx_receive_channel), dim=0)

            rx_extra_receive_channels = torch.tensor([], device=device)
            for j in range(len(rx_additional_hops)):
                rx_extra_receive_channels = torch.cat((rx_extra_receive_channels, rx_additional_hops[j][i].unsqueeze(0)), dim=0)

            rx_receive_channels = torch.cat((rx_receive_channel, rx_extra_receive_channels))

            tx_sense_channels = torch.tensor([], device=device)
            rx_sense_channels = torch.tensor([], device=device)
            for j in range(len(tx_sense_hops)):
                tx_sense_channels = torch.cat((tx_sense_channels, tx_sense_hops[j][i].unsqueeze(0)), dim=0)
            for j in range(len(rx_sense_hops)):
                rx_sense_channels = torch.cat((rx_sense_channels, rx_sense_hops[j][i].unsqueeze(0)), dim=0)

            if IS_SMART_OR_GENIE_JAMMER:
                jammer_observations = torch.empty((len(jammers), NUM_JAMMER_SENSE_CHANNELS+1), device=device)
                # jammer_observations = torch.empty((len(jammers), NUM_CHANNELS+1), device=device)
            elif IS_TRACKING_JAMMER:
                jammer_observations = torch.empty((len(jammers), NUM_CHANNELS), device=device)
            else:
                jammer_observations = torch.empty((len(jammers), 0))
            for j in range(len(jammers)):
                if jammers[j].behavior == "smart":
                    jammer_observation = jammers[j].get_observation(jammer_states[j], jammer_channels[j])
                    jammer_observations[j] = jammer_observation
                    jammer_channel, _, _ = jammers[j].choose_action(jammer_observation, tx_transmit_channel)
                    jammer_channel = jammer_channel.unsqueeze(0)
                    jammer_channels[j] = jammer_channel.long()
                elif jammers[j].behavior == "genie":
                    jammer_observation = jammers[j].get_observation(jammer_states[j], jammer_channels[j])
                    jammer_observations[j] = jammer_observation
                    jammer_seed, _, _ = jammers[j].choose_action(jammer_observation, tx_transmit_channel)
                    jammers[j].agent.fh.generate_sequence()
                    jammer_channel = jammers[j].agent.fh.get_sequence(jammer_seed.unsqueeze(0).long())
                    jammers[j].channels_selected = torch.cat((jammers[j].channels_selected, jammer_channel))
                    jammer_channels[j] = jammer_channel.long()
                else:
                    jammer_observation = jammers[j].get_observation(jammer_states[j], jammer_channels[j])
                    jammer_observations[j] = jammer_observation
                    jammer_channels[j] = jammers[j].choose_action(jammer_observation, tx_transmit_channel).unsqueeze(0)

            num_tx_channel_selected[tx_transmit_channel.long()] += 1
            num_rx_channel_selected[rx_receive_channel.long()] += 1
            for j in range(len(jammers)):
                num_jammer_channel_selected[j][jammer_channels[j]] += 1

            # Set a new channel noise for the next state for the jammer
            jammer_channel_noises = torch.empty((len(jammers), NUM_CHANNELS), device=device)
            for j, jammer in enumerate(jammers):
                noise = torch.abs(torch.normal(0, NOISE_VARIANCE, size=(NUM_CHANNELS,), device=device))
                jammer_channel_noises[j] = noise
                if jammer.behavior == "smart" or jammer.behavior == "genie":
                    jammer.observed_noise = noise.clone()

            # Add the power of the jammers to the channel noise for Tx and Rx
            # Add the power from the Tx to the jammers as well
            for j in range(len(jammers)):
                tx_channel_noise[i][jammer_channels[j]] += jammers[j].get_transmit_power(direction = "transmitter")
                rx_channel_noise[i][jammer_channels[j]] += jammers[j].get_transmit_power(direction = "receiver")
                jammers[j].observed_tx_power = tx_agent.get_transmit_power(direction = "jammer")
                jammer_channel_noises[j][tx_transmit_channel.long()] += jammers[j].observed_tx_power

            rx_observed_power = tx_agent.get_transmit_power(direction = "receiver")
            tx_observed_power = rx_agent.get_transmit_power(direction = "transmitter")

            # Calculate the reward based on the patterns chosen
            # ACK is sent from the receiver
            success = False
            jamming = False
            fading = False
            missed = False
            if received_signal_rx(tx_transmit_channel.long(), rx_receive_channels.long(), rx_observed_power, rx_channel_noise[i]):
                rx_reward += REWARD_SUCCESSFUL

                rx_observed_tx = torch.concat((rx_observed_tx, tx_transmit_channel), dim=0)

                # ACK is received at the transmitter
                if received_signal_tx(tx_transmit_channel.long(), rx_receive_channels.long(), tx_observed_power, tx_channel_noise[i]):
                    tx_reward += REWARD_SUCCESSFUL
                    success = True

                    tx_observed_rx = torch.concat((tx_observed_rx, tx_transmit_channel), dim=0)
                else:
                    tx_reward += REWARD_INTERFERENCE
                    
                    if tx_transmit_channel in jammer_channels:
                        jamming = True
                    else:
                        fading = True

                    tx_observed_rx = torch.concat((tx_observed_rx, torch.tensor([-1], device=device)), dim=0)
            else:
                rx_reward += REWARD_MISS
                tx_reward += REWARD_MISS
                # rx_reward += REWARD_INTERFERENCE
                # tx_reward += REWARD_INTERFERENCE

                # Rx might still be able to sense the Tx's action even though it did not receive the message
                rx_sensing = sensed_signal_rx(tx_transmit_channel.long(), rx_sense_channels, rx_observed_power, rx_channel_noise[i])
                if rx_sensing != -1:
                    rx_observed_tx = torch.concat((rx_observed_tx, rx_sensing), dim=0)
                else:
                    rx_observed_tx = torch.concat((rx_observed_tx, torch.tensor([-1], device=device)), dim=0)
                
                # Tx might still be able to sense the Rx's action even though it did not receive the ACK
                tx_sensing = sensed_signal_tx(rx_receive_channel.long(), tx_sense_channels, tx_observed_power, tx_channel_noise[i])
                if tx_sensing != -1:
                    tx_observed_rx = torch.concat((tx_observed_rx, tx_sensing), dim=0)
                else:
                    tx_observed_rx = torch.concat((tx_observed_rx, torch.tensor([-1], device=device)), dim=0)

                if tx_transmit_channel in rx_receive_channels:
                    if tx_transmit_channel in jammer_channels:
                        jamming = True
                    else:
                        fading = True
                else:
                    missed = True

            if success:
                num_successful_transmissions += 1
            if jamming:
                num_jammed += 1
            if fading:
                num_fading += 1
            if missed:
                num_missed += 1

            for j in range(len(jammers)):
                if jammers[j].behavior == "smart" or jammers[j].behavior == "genie":
                    if sensed_signal_jammer(jammer_channels[j], tx_transmit_channel, jammers[j].observed_tx_power, jammers[j].observed_noise):
                        jammers[j].observed_reward = REWARD_SUCCESSFUL
                    else:
                        jammers[j].observed_reward = REWARD_UNSUCCESSFUL
                    
                    if jammers[j].observed_reward >= 0:
                        jammers[j].num_jammed += 1

            jammer_next_states = torch.empty((len(jammers), NUM_CHANNELS), device=device)
            for j in range(len(jammers)):
                jammer_next_states[j] = jammer_channel_noises[j].clone()

            for j in range(len(jammers)):
                jammer_states[j] = jammer_next_states[j].clone()

        ##################

        if NUM_HOPS > 1:
            tx_observed_rx_seed = tx_agent.fh.get_seed(tx_observed_rx)
            rx_observed_tx_seed = rx_agent.fh.get_seed(rx_observed_tx)
        else:
            tx_observed_rx_seed = tx_observed_rx
            rx_observed_tx_seed = rx_observed_tx

        tx_agent.add_action_observation(tx_observed_rx_seed)
        rx_agent.add_action_observation(rx_observed_tx_seed)

        num_tx_successful_hop_transmissions += tx_reward/NUM_HOPS
        num_rx_successful_hop_transmissions += rx_reward/NUM_HOPS

        if ((rx_reward/NUM_HOPS) >= ((REWARD_SUCCESSFUL+REWARD_MISS)/2)) and rx_observed_tx_seed != -1:
            rx_seed_correct = rx_observed_tx_seed
        else:
            rx_seed_correct = rx_seed

        # The hops used for the observation should be based on the correct seed
        rx_hops = rx_agent.fh.get_sequence(rx_seed_correct)
        # The seed used for the observation should correspond to this
        rx_seed = rx_seed_correct

        # Add the new observed power spectrum to the next state
        tx_next_state = tx_channel_noise.clone()
        rx_next_state = rx_channel_noise.clone()

        # Set the state for the next iteration based on the new observed power spectrum
        tx_state = tx_next_state.clone()
        rx_state = rx_next_state.clone()

    probability_tx_channel_selected = num_tx_channel_selected / np.sum(num_tx_channel_selected)
    probability_rx_channel_selected = num_rx_channel_selected / np.sum(num_rx_channel_selected)
    probability_jammer_channel_selected = []
    for i in range(len(jammers)):
        probability_jammer_channel_selected.append(num_jammer_channel_selected[i] / np.sum(num_jammer_channel_selected[i]))

    return num_successful_transmissions, num_jammed, num_fading, num_missed, num_tx_successful_hop_transmissions, \
        num_rx_successful_hop_transmissions, probability_tx_channel_selected, probability_rx_channel_selected, probability_jammer_channel_selected, \
        num_tx_corr_pred, num_rx_corr_pred

#################################################################################
### main()
#################################################################################

if __name__ == '__main__':

    success_training_rates = []
    jammed_training_rates = []
    fading_training_rates = []
    missed_training_rates = []

    tx_corr_pred_training_rates = []
    rx_corr_pred_training_rates = []

    success_rates = []
    jammed_rates = []
    fading_rates = []
    missed_rates = []

    tx_corr_pred_rates = []
    rx_corr_pred_rates = []
    
    num_runs = 5

    jammer_types_idx = {"random": 0, "probabilistic": 1, "sweeping": 2, "tracking": 3, "smart_ppo": 4, "PPO_Genie": 5}
    jammer_types = ["random", "probabilistic", "sweeping", "tracking", "smart_ppo", "PPO_Genie"]
    jammer_type = "smart_ppo"
    # jammer_type = "tracking"

    # relative_path = f"A_Architecture_Tests/gated_fusion_belief_module/test_1/0_receive_0_sense"
    # relative_path = f"A_Architecture_Tests/a1_belief_module/test_1_pts/0_receive_5_sense"
    # relative_path = f"A_Architecture_Tests/a1_non_pred_nn_belief_module/test_1_pts/0_receive_0_sense"
    # relative_path = f"temp_tests/alternate_belief_module/fh"
    # relative_path = f"A_Architecture_Tests/fusion/test_4/5_receive_0_sense"
    # relative_path = f"A_A_Tests_Maybe_Final/fusion/Standard/test_pts/0_receive_0_sense"
    relative_path = f"A_A_A_Final_Tests/fusion/Hop_length_100/test/{jammer_type}/0_receive_5_sense"
    if not os.path.exists(relative_path):
        os.makedirs(relative_path)

    for run in range(num_runs):
        print("----------------------------------------")
        print("----------------------------------------")

        print(f"Run: {run+1}")

        tx_agent = txPPOAgent(device=device)
        rx_agent = rxPPOAgent(device=device)

        list_of_other_users = []

        if jammer_type == "random":
            random_1 = Jammer(behavior = "random", device = device)
            list_of_other_users.append(random_1)
            jammer_behavior = "naive"

        if jammer_type == "probabilistic":
            prob_1 = Jammer(behavior = "probabilistic", weights = [5, 1, 1, 1, 3]*(NUM_CHANNELS//5), device = device)
            list_of_other_users.append(prob_1)
            jammer_behavior = "naive"

        if jammer_type == "sweeping":
            sweep_1 = Jammer(behavior = "sweep", channel = 2, sweep_interval = 1, device = device)
            list_of_other_users.append(sweep_1)
            jammer_behavior = "naive"

        if jammer_type == "tracking":
            tracking_1 = Jammer(behavior = "tracking", channel = 0, device = device)
            list_of_other_users.append(tracking_1)
            jammer_behavior = "naive"

        if jammer_type == "smart_ppo":
            smart = Jammer(behavior = "smart", smart_type = "PPO", device = device)
            list_of_other_users.append(smart)
            jammer_behavior = "smart"

        if jammer_type == "PPO_Genie":
            genie = Jammer(behavior = "genie", smart_type = "PPO_Genie", device = device)
            list_of_other_users.append(genie)
            jammer_behavior = "genie"

        tx_average_rewards, rx_average_rewards, jammer_average_rewards, num_successsful_training, \
            num_jammed_training, num_fading_training, num_missed_training, \
            num_tx_corr_pred_training, num_rx_corr_pred_training = train_ppo(tx_agent, rx_agent, list_of_other_users)
        print("Jammer average rewards size: ", len(jammer_average_rewards))

        tx_channel_selection_training = tx_agent.channels_selected.cpu().detach().numpy()
        rx_channel_selection_training = rx_agent.channels_selected.cpu().detach().numpy()
        tx_seed_selection_training = tx_agent.fh_seeds_used.cpu().detach().numpy()
        rx_seed_selection_training = rx_agent.fh_seeds_used.cpu().detach().numpy()
        jammer_channel_selection_training = list_of_other_users[0].channels_selected.cpu().detach().numpy()
        if jammer_behavior == "genie":
            jammer_seed_selection_training = list_of_other_users[0].agent.fh_seeds_used.cpu().detach().numpy()

        tx_predicted_rx_actions_training = tx_agent.predicted_actions.cpu().detach().numpy()
        rx_predicted_tx_actions_training = rx_agent.predicted_actions.cpu().detach().numpy()

        successful_transmission_training_rate = (num_successsful_training/NUM_TEST_RUNS)*100
        jammed_training_rate = (num_jammed_training/NUM_TEST_RUNS)*100
        fading_training_rate = (num_fading_training/NUM_TEST_RUNS)*100
        missed_training_rate = (num_missed_training/NUM_TEST_RUNS)*100
        tx_corr_pred_training_rate = (num_tx_corr_pred_training/(NUM_TEST_RUNS/NUM_HOPS))*100
        rx_corr_pred_training_rate = (num_rx_corr_pred_training/(NUM_TEST_RUNS/NUM_HOPS))*100

        print("Finished training:")
        print("Successful transmission rate: ", successful_transmission_training_rate, "%")
        print("Jammed transmission rate: ", jammed_training_rate, "%")
        print("Faded transmission rate: ", fading_training_rate, "%")
        print("Num missed rate: ", missed_training_rate, "%")
        if USE_PREDICTION:
            print("Tx correct prediction rate: ", tx_corr_pred_training_rate)
            print("Rx correct prediction rate: ", rx_corr_pred_training_rate)
        success_training_rates.append(successful_transmission_training_rate)
        jammed_training_rates.append(jammed_training_rate)
        fading_training_rates.append(fading_training_rate)
        missed_training_rates.append(missed_training_rate)
        tx_corr_pred_training_rates.append(tx_corr_pred_training_rate)
        rx_corr_pred_training_rates.append(rx_corr_pred_training_rate)

        num_successful_transmissions, num_jammed, num_fading, num_missed, num_tx_successful_hop_transmissions, \
            num_rx_successful_hop_transmissions, prob_tx_channel, prob_rx_channel, prob_jammer_channel, \
            num_tx_corr_pred, num_rx_corr_pred = test_ppo(tx_agent, rx_agent, list_of_other_users)

        tx_channel_selection_testing = tx_agent.channels_selected.cpu().detach().numpy()
        rx_channel_selection_testing = rx_agent.channels_selected.cpu().detach().numpy()
        tx_seed_selection_testing = tx_agent.fh_seeds_used.cpu().detach().numpy()
        rx_seed_selection_testing = rx_agent.fh_seeds_used.cpu().detach().numpy()
        jammer_channel_selection_testing = list_of_other_users[0].channels_selected.cpu().detach().numpy()
        if jammer_behavior == "genie":
            jammer_seed_selection_testing = list_of_other_users[0].agent.fh_seeds_used.cpu().detach().numpy()

        tx_predicted_rx_actions_testing = tx_agent.predicted_actions.cpu().detach().numpy()
        rx_predicted_tx_actions_testing = rx_agent.predicted_actions.cpu().detach().numpy()

        successful_transmission_rate = (num_successful_transmissions/NUM_TEST_RUNS)*100
        jammed_rate = (num_jammed/NUM_TEST_RUNS)*100
        fading_rate = (num_fading/NUM_TEST_RUNS)*100
        missed_rate = (num_missed/NUM_TEST_RUNS)*100

        tx_corr_pred_rate = (num_tx_corr_pred/(NUM_TEST_RUNS/NUM_HOPS))*100
        rx_corr_pred_rate = (num_rx_corr_pred/(NUM_TEST_RUNS/NUM_HOPS))*100

        print("Finished testing:")
        print("Successful transmission rate: ", successful_transmission_rate, "%")
        print("Jammed transmission rate: ", jammed_rate, "%")
        print("Faded transmission rate: ", fading_rate, "%")
        print("Num missed rate: ", missed_rate, "%")
        if USE_PREDICTION:
            print("Tx correct prediction rate: ", tx_corr_pred_rate)
            print("Rx correct prediction rate: ", rx_corr_pred_rate)
        success_rates.append(successful_transmission_rate)
        jammed_rates.append(jammed_rate)
        fading_rates.append(fading_rate)
        missed_rates.append(missed_rate)
        tx_corr_pred_rates.append(tx_corr_pred_rate)
        rx_corr_pred_rates.append(rx_corr_pred_rate)

        relative_path_run = f"{relative_path}/{run+1}"
        if not os.path.exists(relative_path_run):
            os.makedirs(relative_path_run)
        if not os.path.exists(f"{relative_path_run}/plots"):
            os.makedirs(f"{relative_path_run}/plots")
        if not os.path.exists(f"{relative_path_run}/data"):
            os.makedirs(f"{relative_path_run}/data")
        if not os.path.exists(f"{relative_path_run}/data/channel_pattern"):
            os.makedirs(f"{relative_path_run}/data/channel_pattern")
        if not os.path.exists(f"{relative_path_run}/training_data"):
            os.makedirs(f"{relative_path_run}/training_data")
        if not os.path.exists(f"{relative_path_run}/testing_data"):
            os.makedirs(f"{relative_path_run}/testing_data")

        # Save data in textfiles
        np.savetxt(f"{relative_path_run}/data/average_reward_tx.txt", tx_average_rewards)
        np.savetxt(f"{relative_path_run}/data/actor_losses_tx.txt", tx_agent.actor_losses.cpu().detach().numpy())
        np.savetxt(f"{relative_path_run}/data/critic_losses_tx.txt", tx_agent.critic_losses.cpu().detach().numpy())
        np.savetxt(f"{relative_path_run}/data/average_reward_rx.txt", rx_average_rewards)
        np.savetxt(f"{relative_path_run}/data/actor_losses_rx.txt", rx_agent.actor_losses.cpu().detach().numpy())
        np.savetxt(f"{relative_path_run}/data/critic_losses_rx.txt", rx_agent.critic_losses.cpu().detach().numpy())
        np.savetxt(f"{relative_path_run}/data/average_reward_jammer.txt", jammer_average_rewards)
        np.savetxt(f"{relative_path_run}/training_data/success_training_rates.txt", [successful_transmission_training_rate])
        np.savetxt(f"{relative_path_run}/training_data/jammed_training_rates.txt", [jammed_training_rate])
        np.savetxt(f"{relative_path_run}/training_data/fading_training_rates.txt", [fading_training_rate])
        np.savetxt(f"{relative_path_run}/training_data/missed_training_rates.txt", [missed_training_rate])
        if USE_PREDICTION:
            np.savetxt(f"{relative_path_run}/training_data/tx_corr_pred_training_rates.txt", [tx_corr_pred_training_rate])
            np.savetxt(f"{relative_path_run}/training_data/rx_corr_pred_training_rates.txt", [rx_corr_pred_training_rate])
        np.savetxt(f"{relative_path_run}/testing_data/success_rates.txt", [successful_transmission_rate])
        np.savetxt(f"{relative_path_run}/testing_data/jammed_rates.txt", [jammed_rate])
        np.savetxt(f"{relative_path_run}/testing_data/fading_rates.txt", [fading_rate])
        np.savetxt(f"{relative_path_run}/testing_data/missed_rates.txt", [missed_rate])
        if USE_PREDICTION:
            np.savetxt(f"{relative_path_run}/testing_data/tx_corr_pred_rates.txt", [tx_corr_pred_rate])
            np.savetxt(f"{relative_path_run}/testing_data/rx_corr_pred_rates.txt", [rx_corr_pred_rate])
        
        # Saving the channel and pattern selections
        np.savetxt(f"{relative_path_run}/data/channel_pattern/tx_channel_selection_training.txt", tx_channel_selection_training)
        np.savetxt(f"{relative_path_run}/data/channel_pattern/rx_channel_selection_training.txt", rx_channel_selection_training)
        np.savetxt(f"{relative_path_run}/data/channel_pattern/jammer_channel_selection_training.txt", jammer_channel_selection_training)
        np.savetxt(f"{relative_path_run}/data/channel_pattern/tx_seed_selection_training.txt", tx_seed_selection_training)
        np.savetxt(f"{relative_path_run}/data/channel_pattern/rx_seed_selection_training.txt", rx_seed_selection_training)
        np.savetxt(f"{relative_path_run}/data/channel_pattern/tx_channel_selection_testing.txt", tx_channel_selection_testing)
        np.savetxt(f"{relative_path_run}/data/channel_pattern/rx_channel_selection_testing.txt", rx_channel_selection_testing)
        np.savetxt(f"{relative_path_run}/data/channel_pattern/jammer_channel_selection_testing.txt", jammer_channel_selection_testing)
        np.savetxt(f"{relative_path_run}/data/channel_pattern/tx_seed_selection_testing.txt", tx_seed_selection_testing)
        np.savetxt(f"{relative_path_run}/data/channel_pattern/rx_seed_selection_testing.txt", rx_seed_selection_testing)
        if jammer_behavior == "genie":
            np.savetxt(f"{relative_path_run}/data/channel_pattern/jammer_seed_selection_training.txt", jammer_seed_selection_training)
            np.savetxt(f"{relative_path_run}/data/channel_pattern/jammer_seed_selection_testing.txt", jammer_seed_selection_testing)

        save_results_plot(tx_average_rewards, rx_average_rewards, prob_tx_channel, prob_rx_channel, jammer_type, filepath = relative_path_run+"/plots")
        save_probability_selection(prob_tx_channel, prob_rx_channel, prob_jammer_channel, jammer_type, filepath = relative_path_run+"/plots")
        save_results_losses(tx_agent.actor_losses.cpu().detach().numpy(), tx_agent.critic_losses.cpu().detach().numpy(),
                            tx_agent.belief_losses.cpu().detach().numpy(), tx_agent.entropy_losses.cpu().detach().numpy(),
                            rx_agent.actor_losses.cpu().detach().numpy(), rx_agent.critic_losses.cpu().detach().numpy(), 
                            rx_agent.belief_losses.cpu().detach().numpy(), rx_agent.entropy_losses.cpu().detach().numpy(),
                            filepath = relative_path_run+"/plots")
        # save_channel_selection_training(tx_agent.channels_selected.cpu().detach().numpy(), rx_agent.channels_selected.cpu().detach().numpy(), 
        #                                 list_of_other_users[0].channels_selected.cpu().detach().numpy(), filepath = relative_path_run+"/plots")
        save_channel_selection(tx_channel_selection_training, rx_channel_selection_training, jammer_channel_selection_training,
                                tx_channel_selection_testing, rx_channel_selection_testing, jammer_channel_selection_testing, filepath = relative_path_run+"/plots")
        save_seed_selection(tx_seed_selection_training, rx_seed_selection_training, tx_seed_selection_testing, 
                               rx_seed_selection_testing, filepath = relative_path_run+"/plots")
        if jammer_behavior == "smart" or jammer_behavior == "genie":
            save_jammer_results_plot(jammer_average_rewards, filepath = relative_path_run+"/plots")
        if jammer_behavior == "genie":
            save_seed_selection_jammer(tx_seed_selection_training, rx_seed_selection_training, jammer_seed_selection_training,
                                       tx_seed_selection_testing, rx_seed_selection_testing, jammer_seed_selection_testing, filepath = relative_path_run+"/plots")
        save_predicted_actions(tx_predicted_rx_actions_training, rx_predicted_tx_actions_training,
                               tx_predicted_rx_actions_testing, rx_predicted_tx_actions_testing, filepath = relative_path_run+"/plots")

    # if jammer_type == "smart":
    #     plot_results_smart_jammer(tx_average_rewards, rx_average_rewards, jammer_average_rewards, prob_tx_channel, prob_rx_channel, prob_jammer_channel)
    # else:
    #     plot_results(tx_average_rewards, rx_average_rewards, prob_tx_channel, prob_rx_channel, jammer_type)
    #     # plot_results(tx_average_rewards, rx_average_rewards, jammer_type)

    #     plot_probability_selection(prob_tx_channel, prob_rx_channel, prob_jammer_channel, jammer_type)

    if num_runs > 1:
        print("Average training data:")
        print("Average success rate: ", np.mean(success_training_rates), "%")
        print("Average jammed rate: ", np.mean(jammed_training_rates), "%")
        print("Average fading rate: ", np.mean(fading_training_rates), "%")
        print("Average missed rate: ", np.mean(missed_training_rates), "%")
        if USE_PREDICTION:
            print("Average Tx correct prediction rate: ", np.mean(tx_corr_pred_training_rates))
            print("Average Rx correct prediction rate: ", np.mean(rx_corr_pred_training_rates))
        print()
        print("Average testing data:")
        print("Average success rate: ", np.mean(success_rates), "%")
        print("Average jammed rate: ", np.mean(jammed_rates), "%")
        print("Average fading rate: ", np.mean(fading_rates), "%")
        print("Average missed rate: ", np.mean(missed_rates), "%")
        if USE_PREDICTION:
            print("Average Tx correct prediction rate: ", np.mean(tx_corr_pred_rates))
            print("Average Rx correct prediction rate: ", np.mean(rx_corr_pred_rates))

    if not os.path.exists(f"{relative_path}/{jammer_type}/training_data"):
        os.makedirs(f"{relative_path}/{jammer_type}/training_data")
    if not os.path.exists(f"{relative_path}/{jammer_type}/testing_data"):
        os.makedirs(f"{relative_path}/{jammer_type}/testing_data")

    np.savetxt(f"{relative_path}/{jammer_type}/training_data/all_success_training_rates.txt", success_training_rates)
    np.savetxt(f"{relative_path}/{jammer_type}/training_data/all_jammed_training_rates.txt", jammed_training_rates)
    np.savetxt(f"{relative_path}/{jammer_type}/training_data/all_fading_training_rates.txt", fading_training_rates)
    np.savetxt(f"{relative_path}/{jammer_type}/training_data/all_missed_training_rates.txt", missed_training_rates)
    np.savetxt(f"{relative_path}/{jammer_type}/training_data/all_tx_corr_pred_training_rates.txt", tx_corr_pred_training_rates)
    np.savetxt(f"{relative_path}/{jammer_type}/training_data/all_rx_corr_pred_training_rates.txt", rx_corr_pred_training_rates)
    np.savetxt(f"{relative_path}/{jammer_type}/training_data/average_success_training_rate.txt", [np.mean(success_training_rates)])
    np.savetxt(f"{relative_path}/{jammer_type}/training_data/average_jammed_training_rates.txt", [np.mean(jammed_training_rates)])
    np.savetxt(f"{relative_path}/{jammer_type}/training_data/average_missed_training_rates.txt", [np.mean(missed_training_rates)])
    if USE_PREDICTION:
        np.savetxt(f"{relative_path}/{jammer_type}/training_data/average_tx_corr_pred_training_rates.txt", [np.mean(tx_corr_pred_training_rates)])
        np.savetxt(f"{relative_path}/{jammer_type}/training_data/average_rx_corr_pred_training_rates.txt", [np.mean(rx_corr_pred_training_rates)])

    np.savetxt(f"{relative_path}/{jammer_type}/testing_data/all_success_rates.txt", success_rates)
    np.savetxt(f"{relative_path}/{jammer_type}/testing_data/all_jammed_rates.txt", jammed_rates)
    np.savetxt(f"{relative_path}/{jammer_type}/testing_data/all_fading_rates.txt", fading_rates)
    np.savetxt(f"{relative_path}/{jammer_type}/testing_data/all_missed_rates.txt", missed_rates)
    np.savetxt(f"{relative_path}/{jammer_type}/testing_data/all_tx_corr_pred_rates.txt", tx_corr_pred_rates)
    np.savetxt(f"{relative_path}/{jammer_type}/testing_data/all_rx_corr_pred_rates.txt", rx_corr_pred_rates)
    np.savetxt(f"{relative_path}/{jammer_type}/testing_data/average_success_rate.txt", [np.mean(success_rates)])
    np.savetxt(f"{relative_path}/{jammer_type}/testing_data/average_jammed_rates.txt", [np.mean(jammed_rates)])
    np.savetxt(f"{relative_path}/{jammer_type}/testing_data/average_fading_rates.txt", [np.mean(fading_rates)])
    np.savetxt(f"{relative_path}/{jammer_type}/testing_data/average_missed_rates.txt", [np.mean(missed_rates)])
    if USE_PREDICTION:
        np.savetxt(f"{relative_path}/{jammer_type}/testing_data/average_tx_corr_pred_rates.txt", [np.mean(tx_corr_pred_rates)])
        np.savetxt(f"{relative_path}/{jammer_type}/testing_data/average_rx_corr_pred_rates.txt", [np.mean(rx_corr_pred_rates)])