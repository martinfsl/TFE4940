import numpy as np
import matplotlib.pyplot as plt

# NUM_EPISODES = 50000
NUM_EPISODES = 100000
NUM_CHANNELS = 100

#################################################################################

# relative_path = f"Comparison/PPO_tests/update_random"

# rewards_tx_1 = np.loadtxt(f'{relative_path}/test_1/average_reward_both_tx.txt', dtype=float)
# rewards_rx_1 = np.loadtxt(f'{relative_path}/test_1/average_reward_both_rx.txt', dtype=float)

# rewards_tx_2 = np.loadtxt(f'{relative_path}/test_2/average_reward_both_tx.txt', dtype=float)
# rewards_rx_2 = np.loadtxt(f'{relative_path}/test_2/average_reward_both_rx.txt', dtype=float)

# rewards_tx_3 = np.loadtxt(f'{relative_path}/test_3/average_reward_both_tx.txt', dtype=float)
# rewards_rx_3 = np.loadtxt(f'{relative_path}/test_3/average_reward_both_rx.txt', dtype=float)

#################################################################################

#################################################################################

# relative_path = f"Comparison/PPO_tests/update_random/test_4"
# relative_path = f"Comparison/PPO_tests/update_random/further_tests/0_75_gamma"

# rewards_tx_1 = np.loadtxt(f'{relative_path}/1/average_reward_both_tx.txt', dtype=float)
# rewards_tx_2 = np.loadtxt(f'{relative_path}/2/average_reward_both_tx.txt', dtype=float)
# rewards_tx_3 = np.loadtxt(f'{relative_path}/3/average_reward_both_tx.txt', dtype=float)
# rewards_tx_4 = np.loadtxt(f'{relative_path}/4/average_reward_both_tx.txt', dtype=float)
# rewards_tx_5 = np.loadtxt(f'{relative_path}/5/average_reward_both_tx.txt', dtype=float)

# rewards_rx_1 = np.loadtxt(f'{relative_path}/1/average_reward_both_rx.txt', dtype=float)
# rewards_rx_2 = np.loadtxt(f'{relative_path}/2/average_reward_both_rx.txt', dtype=float)
# rewards_rx_3 = np.loadtxt(f'{relative_path}/3/average_reward_both_rx.txt', dtype=float)
# rewards_rx_4 = np.loadtxt(f'{relative_path}/4/average_reward_both_rx.txt', dtype=float)
# rewards_rx_5 = np.loadtxt(f'{relative_path}/5/average_reward_both_rx.txt', dtype=float)

# rewards_tx_1 = (rewards_tx_1 + 1)/2
# rewards_tx_2 = (rewards_tx_2 + 1)/2
# rewards_tx_3 = (rewards_tx_3 + 1)/2
# rewards_tx_4 = (rewards_tx_4 + 1)/2
# rewards_tx_5 = (rewards_tx_5 + 1)/2

# rewards_rx_1 = (rewards_rx_1 + 1)/2
# rewards_rx_2 = (rewards_rx_2 + 1)/2
# rewards_rx_3 = (rewards_rx_3 + 1)/2
# rewards_rx_4 = (rewards_rx_4 + 1)/2
# rewards_rx_5 = (rewards_rx_5 + 1)/2

#################################################################################

#################################################################################

relative_path = f"Comparison/PPO_tests/update_random/further_tests/gamma"

rewards_tx_0_50 = np.loadtxt(f'{relative_path}/0_5/3/average_reward_both_tx.txt', dtype=float)
rewards_tx_0_60 = np.loadtxt(f'{relative_path}/0_6/1/average_reward_both_tx.txt', dtype=float)
rewards_tx_0_75 = np.loadtxt(f'{relative_path}/0_75/5/average_reward_both_tx.txt', dtype=float)
rewards_tx_0_85 = np.loadtxt(f'{relative_path}/0_85/4/average_reward_both_tx.txt', dtype=float)
rewards_tx_0_95 = np.loadtxt(f'{relative_path}/0_95/3/average_reward_both_tx.txt', dtype=float)
rewards_tx_0_99 = np.loadtxt(f'{relative_path}/0_99/3/average_reward_both_tx.txt', dtype=float)

rewards_rx_0_50 = np.loadtxt(f'{relative_path}/0_5/3/average_reward_both_rx.txt', dtype=float)
rewards_rx_0_60 = np.loadtxt(f'{relative_path}/0_6/1/average_reward_both_rx.txt', dtype=float)
rewards_rx_0_75 = np.loadtxt(f'{relative_path}/0_75/5/average_reward_both_rx.txt', dtype=float)
rewards_rx_0_85 = np.loadtxt(f'{relative_path}/0_85/4/average_reward_both_rx.txt', dtype=float)
rewards_rx_0_95 = np.loadtxt(f'{relative_path}/0_95/3/average_reward_both_rx.txt', dtype=float)
rewards_rx_0_99 = np.loadtxt(f'{relative_path}/0_99/3/average_reward_both_rx.txt', dtype=float)

rewards_tx_0_50 = (rewards_tx_0_50 + 1)/2
rewards_tx_0_60 = (rewards_tx_0_60 + 1)/2
rewards_tx_0_75 = (rewards_tx_0_75 + 1)/2
rewards_tx_0_85 = (rewards_tx_0_85 + 1)/2
rewards_tx_0_95 = (rewards_tx_0_95 + 1)/2
rewards_tx_0_99 = (rewards_tx_0_99 + 1)/2

#################################################################################




plt.figure(1)

plt.plot(rewards_tx_0_50, label='$\gamma$ = 0.5')
plt.plot(rewards_tx_0_60, label='$\gamma$ = 0.6')
plt.plot(rewards_tx_0_75, label='$\gamma$ = 0.75')
plt.plot(rewards_tx_0_85, label='$\gamma$ = 0.85')
plt.plot(rewards_tx_0_95, label='$\gamma$ = 0.95')
plt.plot(rewards_tx_0_99, label='$\gamma$ = 0.99')

plt.xlabel('Episode')
plt.ylabel('Normalized average reward')
plt.title(f'Normalized average reward during training agent using PPO')
plt.legend()
plt.show()
