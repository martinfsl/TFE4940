import matplotlib.pyplot as plt

from constants import *

#################################################################################
### Plotting the results
#################################################################################

def plot_results(tx_average_rewards, rx_average_rewards, probability_tx_channel_selected, probability_rx_channel_selected, jammer_type):
    plt.figure(1, figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(tx_average_rewards)
    # plt.axvline(x=2*DQN_BATCH_SIZE, color='r', linestyle='--', label='2*Batch Size (Here training begins)')
    plt.xlabel("Episode")
    plt.ylabel("Tx average reward")
    plt.title("Tx average reward over episodes during training")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(rx_average_rewards)
    # plt.axvline(x=2*DQN_BATCH_SIZE, color='r', linestyle='--', label='2*Batch Size (Here training begins)')
    plt.xlabel("Episode")
    plt.ylabel("Rx average reward")
    plt.title("Rx average reward over episodes during training")
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.bar(np.arange(1, NUM_CHANNELS+1, 1), probability_tx_channel_selected)
    plt.xlabel("Channel")
    plt.ylabel("Probability of channel selection")
    plt.title("Probability of channel selection for Tx during testing")

    plt.subplot(2, 2, 4)
    plt.bar(np.arange(1, NUM_CHANNELS+1, 1), probability_rx_channel_selected)
    plt.xlabel("Channel")
    plt.ylabel("Probability of channel selection")
    plt.title("Probability of channel selection for Rx during testing")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f"DDQN GRU, {NUM_CHANNELS} channels, 1 {jammer_type}, {BATCH_SIZE} sequence length")
    # plt.show()

#################################################################################
### Plotting the probability of selections
#################################################################################

def plot_probability_selection(probability_tx_channel_selected, probability_rx_channel_selected, probability_jammer_channel_selected, jammer_type):
    plt.figure(2, figsize=(8, 4))

    plt.bar(np.arange(1, NUM_CHANNELS+1, 1), probability_tx_channel_selected)
    plt.xlabel("Channel")
    plt.ylabel("Probability of channel selection")
    plt.title("Probability of channel selection for Tx during testing")


    plt.figure(3, figsize=(8, 4))

    plt.bar(np.arange(1, NUM_CHANNELS+1, 1), probability_rx_channel_selected)
    plt.xlabel("Channel")
    plt.ylabel("Probability of channel selection")
    plt.title("Probability of channel selection for Rx during testing")

    for i in range(len(probability_jammer_channel_selected)):
        plt.figure(4+i, figsize=(8, 4))
        plt.bar(np.arange(1, NUM_CHANNELS+1, 1), probability_jammer_channel_selected[i])
        plt.xlabel("Channel")
        plt.ylabel("Probability of channel selection")
        plt.title(f"Probability of channel selection for Jammer {i+1} during testing")

    plt.show()

#################################################################################
### Plotting the results with the smart jammer
#################################################################################

def plot_results_smart_jammer(tx_average_rewards, rx_average_rewards, jammer_average_rewards, probability_tx_channel_selected, probability_rx_channel_selected, probability_jammer_channel_selected):
    # Normalize the rewards
    # tx_average_rewards = (np.array(tx_average_rewards) + 1)/2
    # rx_average_rewards = (np.array(rx_average_rewards) + 1)/2
    # jammer_average_rewards = (np.array(jammer_average_rewards) + 1)/2
    
    plt.figure(1, figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(tx_average_rewards, label = "Tx")
    plt.plot(rx_average_rewards, label = "Rx")
    plt.plot(jammer_average_rewards, label = "Jammer")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.title("Average Reward over episodes during training")

    plt.subplot(2, 2, 2)
    plt.bar(np.arange(1, NUM_CHANNELS+1, 1), probability_tx_channel_selected, label = "Tx")
    plt.xlabel("Channel")
    plt.ylabel("Probability of channel selection")
    plt.title("Probability of Tx channel selection during testing")

    plt.subplot(2, 2, 3)
    plt.bar(np.arange(1, NUM_CHANNELS+1, 1), probability_rx_channel_selected, label = "Rx")
    plt.xlabel("Channel")
    plt.ylabel("Probability of channel selection")
    plt.title("Probability of Rx channel selection during testing")

    plt.subplot(2, 2, 4)
    plt.bar(np.arange(1, NUM_CHANNELS+1, 1), probability_jammer_channel_selected[0], label = f"Jammer")
    plt.xlabel("Channel")
    plt.ylabel("Probability of channel selection")
    plt.title(f"Probability of Smart Jammer channel selection during testing")

    plt.show()