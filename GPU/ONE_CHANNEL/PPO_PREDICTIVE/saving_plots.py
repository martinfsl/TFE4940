import matplotlib.pyplot as plt

from constants import *

#################################################################################
### Plotting the results
#################################################################################

def save_results_plot(tx_average_rewards, rx_average_rewards, probability_tx_channel_selected, probability_rx_channel_selected, jammer_type, filepath = None):
    plt.figure(figsize=(12, 8))

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
    plt.suptitle(f"PPO algorithm, {NUM_CHANNELS} channels, 1 {jammer_type}")
    # plt.show()

    if filepath is not None:
        plt.savefig(f"{filepath}/results_with_probabilities.png")

        plt.close()

#################################################################################
### Plotting the probability of selections
#################################################################################

def save_probability_selection(probability_tx_channel_selected, probability_rx_channel_selected, probability_jammer_channel_selected, jammer_type, filepath = None):
    plt.figure(figsize=(8, 4))

    plt.bar(np.arange(1, NUM_CHANNELS+1, 1), probability_tx_channel_selected)
    plt.xlabel("Channel")
    plt.ylabel("Probability of channel selection")
    plt.title("Probability of channel selection for Tx during testing")

    if filepath is not None:
        plt.savefig(f"{filepath}/probability_tx_channel_selected.png")

        plt.close()

    plt.figure(figsize=(8, 4))

    plt.bar(np.arange(1, NUM_CHANNELS+1, 1), probability_rx_channel_selected)
    plt.xlabel("Channel")
    plt.ylabel("Probability of channel selection")
    plt.title("Probability of channel selection for Rx during testing")

    if filepath is not None:
        plt.savefig(f"{filepath}/probability_rx_channel_selected.png")

        plt.close()

    for i in range(len(probability_jammer_channel_selected)):
        plt.figure(figsize=(8, 4))
        plt.bar(np.arange(1, NUM_CHANNELS+1, 1), probability_jammer_channel_selected[i])
        plt.xlabel("Channel")
        plt.ylabel("Probability of channel selection")
        plt.title(f"Probability of channel selection for Jammer {i+1} during testing")

        if filepath is not None:
            plt.savefig(f"{filepath}/probability_jammer_channel_selected_{i}.png")

            plt.close()

    # plt.show()

#################################################################################
### Plotting the results with the smart jammer
#################################################################################

def save_results_smart_jammer(tx_average_rewards, rx_average_rewards, jammer_average_rewards, 
                              probability_tx_channel_selected, probability_rx_channel_selected, probability_jammer_channel_selected,
                              filepath = None):
    # Normalize the rewards
    # tx_average_rewards = (np.array(tx_average_rewards) + 1)/2
    # rx_average_rewards = (np.array(rx_average_rewards) + 1)/2
    # jammer_average_rewards = (np.array(jammer_average_rewards) + 1)/2
    
    plt.figure(figsize=(12, 8))

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

    # plt.show()

    if filepath is not None:
        plt.savefig(f"{filepath}/results_with_probabilities.png")

        plt.close()

#################################################################################
### Plotting the actor and critic losses
#################################################################################

def save_results_losses(tx_actor_losses, tx_critic_losses, rx_actor_losses, rx_critic_losses, filepath = None):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(tx_actor_losses)
    plt.xlabel("Episode")
    plt.ylabel("Actor Loss")
    plt.title("Actor Loss over episodes during training for Tx")

    plt.subplot(2, 2, 2)
    plt.plot(tx_critic_losses)
    plt.xlabel("Episode")
    plt.ylabel("Critic Loss")
    plt.title("Critic Loss over episodes during training for Tx")

    plt.subplot(2, 2, 3)
    plt.plot(rx_actor_losses)
    plt.xlabel("Episode")
    plt.ylabel("Actor Loss")
    plt.title("Actor Loss over episodes during training for Rx")

    plt.subplot(2, 2, 4)
    plt.plot(rx_critic_losses)
    plt.xlabel("Episode")
    plt.ylabel("Critic Loss")
    plt.title("Critic Loss over episodes during training for Rx")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f"PPO algorithm")
    # plt.show()

    if filepath is not None:
        plt.savefig(f"{filepath}/training_losses.png")

        plt.close()

def save_channel_selection_training(tx_channel_selection, rx_channel_selection, jammer_channel_selection, filepath = None):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(tx_channel_selection)
    plt.xlabel("Episode")
    plt.ylabel("Channel")
    plt.title("Tx channel selection over episodes during training")

    plt.subplot(2, 2, 2)
    plt.plot(rx_channel_selection)
    plt.xlabel("Episode")
    plt.ylabel("Channel")
    plt.title("Rx channel selection over episodes during training")

    plt.subplot(2, 2, 3)
    plt.plot(jammer_channel_selection)
    plt.xlabel("Episode")
    plt.ylabel("Channel")
    plt.title("Jammer channel selection over episodes during training")

    plt.subplot(2, 2, 4)
    plt.plot(tx_channel_selection, label = "Tx")
    plt.plot(rx_channel_selection, label = "Rx")
    plt.plot(jammer_channel_selection, label = "Jammer")
    plt.xlabel("Episode")
    plt.ylabel("Channel")
    plt.title("Channel selection over episodes during training")
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f"PPO algorithm")
    # plt.show()

    if filepath is not None:
        plt.savefig(f"{filepath}/channel_selection_training.png")

        plt.close()