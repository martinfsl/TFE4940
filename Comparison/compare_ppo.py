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

# relative_path = f"Comparison/PPO_tests/update_random/further_tests/gamma"

# rewards_tx_0_50 = np.loadtxt(f'{relative_path}/0_5/3/average_reward_both_tx.txt', dtype=float)
# rewards_tx_0_60 = np.loadtxt(f'{relative_path}/0_6/1/average_reward_both_tx.txt', dtype=float)
# rewards_tx_0_75 = np.loadtxt(f'{relative_path}/0_75/5/average_reward_both_tx.txt', dtype=float)
# rewards_tx_0_85 = np.loadtxt(f'{relative_path}/0_85/4/average_reward_both_tx.txt', dtype=float)
# rewards_tx_0_95 = np.loadtxt(f'{relative_path}/0_95/3/average_reward_both_tx.txt', dtype=float)
# rewards_tx_0_99 = np.loadtxt(f'{relative_path}/0_99/3/average_reward_both_tx.txt', dtype=float)

# rewards_rx_0_50 = np.loadtxt(f'{relative_path}/0_5/3/average_reward_both_rx.txt', dtype=float)
# rewards_rx_0_60 = np.loadtxt(f'{relative_path}/0_6/1/average_reward_both_rx.txt', dtype=float)
# rewards_rx_0_75 = np.loadtxt(f'{relative_path}/0_75/5/average_reward_both_rx.txt', dtype=float)
# rewards_rx_0_85 = np.loadtxt(f'{relative_path}/0_85/4/average_reward_both_rx.txt', dtype=float)
# rewards_rx_0_95 = np.loadtxt(f'{relative_path}/0_95/3/average_reward_both_rx.txt', dtype=float)
# rewards_rx_0_99 = np.loadtxt(f'{relative_path}/0_99/3/average_reward_both_rx.txt', dtype=float)

# rewards_tx_0_50 = (rewards_tx_0_50 + 1)/2
# rewards_tx_0_60 = (rewards_tx_0_60 + 1)/2
# rewards_tx_0_75 = (rewards_tx_0_75 + 1)/2
# rewards_tx_0_85 = (rewards_tx_0_85 + 1)/2
# rewards_tx_0_95 = (rewards_tx_0_95 + 1)/2
# rewards_tx_0_99 = (rewards_tx_0_99 + 1)/2

#################################################################################

#################################################################################

# relative_path = f"Comparison/PPO_tests/update_random/further_tests/lambda_param"

# rewards_tx_0_40 = np.loadtxt(f'{relative_path}/0_40/2/average_reward_both_tx.txt', dtype=float)
# rewards_tx_0_50 = np.loadtxt(f'{relative_path}/0_50/2/average_reward_both_tx.txt', dtype=float)
# rewards_tx_0_60 = np.loadtxt(f'{relative_path}/0_60/3/average_reward_both_tx.txt', dtype=float)
# rewards_tx_0_70 = np.loadtxt(f'{relative_path}/0_70/4/average_reward_both_tx.txt', dtype=float)
# rewards_tx_0_75 = np.loadtxt(f'{relative_path}/0_75/1/average_reward_both_tx.txt', dtype=float)
# rewards_tx_0_80 = np.loadtxt(f'{relative_path}/0_80/4/average_reward_both_tx.txt', dtype=float)
# rewards_tx_0_85 = np.loadtxt(f'{relative_path}/0_85/5/average_reward_both_tx.txt', dtype=float)
# rewards_tx_0_90 = np.loadtxt(f'{relative_path}/0_90/3/average_reward_both_tx.txt', dtype=float)
# rewards_tx_0_95 = np.loadtxt(f'{relative_path}/0_95/5/average_reward_both_tx.txt', dtype=float)

# rewards_tx_0_40 = (rewards_tx_0_40 + 1)/2
# rewards_tx_0_50 = (rewards_tx_0_50 + 1)/2
# rewards_tx_0_60 = (rewards_tx_0_60 + 1)/2
# rewards_tx_0_70 = (rewards_tx_0_70 + 1)/2
# rewards_tx_0_75 = (rewards_tx_0_75 + 1)/2
# rewards_tx_0_80 = (rewards_tx_0_80 + 1)/2
# rewards_tx_0_85 = (rewards_tx_0_85 + 1)/2
# rewards_tx_0_90 = (rewards_tx_0_90 + 1)/2
# rewards_tx_0_95 = (rewards_tx_0_95 + 1)/2

#################################################################################

#################################################################################

# relative_path = f"Comparison/march_tests/PPO/ppo_smart_jammer/12_03"

# rewards_tx = np.loadtxt(f'{relative_path}/average_reward_both_tx.txt', dtype=float)
# rewards_rx = np.loadtxt(f'{relative_path}/average_reward_both_rx.txt', dtype=float)
# rewards_jammer = np.loadtxt(f'{relative_path}/average_reward_jammer.txt', dtype=float)

# rewards_tx = (rewards_tx + 1)/2
# rewards_rx = (rewards_rx + 1)/2
# rewards_jammer = (rewards_jammer + 1)/2

#################################################################################

#################################################################################

relative_path = f"Comparison/march_tests/PPO/ppo_smart_jammer/12_03/param_testing/lambda_param"
type = "both_tx"
run = 1

rewards_0_40 = np.loadtxt(f'{relative_path}/0_4/{run}/average_reward_{type}.txt', dtype=float)
rewards_0_50 = np.loadtxt(f'{relative_path}/0_5/{run}/average_reward_{type}.txt', dtype=float)
rewards_0_60 = np.loadtxt(f'{relative_path}/0_6/{run}/average_reward_{type}.txt', dtype=float)
rewards_0_80 = np.loadtxt(f'{relative_path}/0_8/{run}/average_reward_{type}.txt', dtype=float)
rewards_0_90 = np.loadtxt(f'{relative_path}/0_9/{run}/average_reward_{type}.txt', dtype=float)

size = "0_8"

rewards_1 = np.loadtxt(f'{relative_path}/{size}/1/average_reward_{type}.txt', dtype=float)
rewards_2 = np.loadtxt(f'{relative_path}/{size}/2/average_reward_{type}.txt', dtype=float)
rewards_3 = np.loadtxt(f'{relative_path}/{size}/3/average_reward_{type}.txt', dtype=float)
rewards_4 = np.loadtxt(f'{relative_path}/{size}/4/average_reward_{type}.txt', dtype=float)
rewards_5 = np.loadtxt(f'{relative_path}/{size}/5/average_reward_{type}.txt', dtype=float)

rewards_1 = (rewards_1 + 1)/2
rewards_2 = (rewards_2 + 1)/2
rewards_3 = (rewards_3 + 1)/2
rewards_4 = (rewards_4 + 1)/2
rewards_5 = (rewards_5 + 1)/2

#################################################################################



plt.figure(1)

# plt.plot(rewards_0_40, label='Lambda = 0.40')
# plt.plot(rewards_0_50, label='Lambda = 0.50')
# plt.plot(rewards_0_60, label='Lambda = 0.60')
# plt.plot(rewards_0_80, label='Lambda = 0.80')
# plt.plot(rewards_0_90, label='Lambda = 0.90')

plt.plot(rewards_1, label='Run 1')
plt.plot(rewards_2, label='Run 2')
plt.plot(rewards_3, label='Run 3')
plt.plot(rewards_4, label='Run 4')
plt.plot(rewards_5, label='Run 5')

plt.xlabel('Episode')
plt.ylabel('Normalized average reward')
plt.title(f'Normalized average reward during training agent using PPO')
plt.legend()
plt.show()
