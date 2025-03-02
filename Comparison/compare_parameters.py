import numpy as np
import matplotlib.pyplot as plt

start_path = 'Comparison/parameter_testing'
algorithm = 'IDDQN'
parameter = 'discount_factor'

relative_path = f'{start_path}/{algorithm}_{parameter}'

symbol = {'learning_rate': 'alpha', 
          'exploration_rate': '\epsilon',
          'batch_size': 'B',
          'memory_size': 'L',
          'exploration_decay': '\epsilon_{red}',
          'discount_factor': '\gamma'}

parameter_values = {'alpha': ['0_0025', '0_005', '0_0075', '0_01', '0_05'],
                    '\epsilon': ['0_1', '0_3', '0_5', '0_7', '0_9'],
                    'B': ['8', '16', '32', '48', '64'],
                    'L': ['25', '50', '100', '200', '400'],
                    '\epsilon_{red}': ['0_9', '0_95', '0_99', '0_995', '0_999'],
                    '\gamma': ['0_1', '0_25', '0_35', '0_5', '0_7', '0_8', '0_9']}

param_rewards_tx = []
param_rewards_rx = []

for value in parameter_values[symbol[parameter]]:
    rewards_tx = np.loadtxt(f'{relative_path}/{value}/average_reward_both_tx.txt', dtype=float)
    rewards_rx = np.loadtxt(f'{relative_path}/{value}/average_reward_both_rx.txt', dtype=float)

    rewards_tx_normalized = (rewards_tx + 1)/2
    rewards_rx_normalized = (rewards_rx + 1)/2

    # param_rewards_tx.append(rewards_tx)
    # param_rewards_rx.append(rewards_rx)

    param_rewards_tx.append(rewards_tx_normalized)
    param_rewards_rx.append(rewards_rx_normalized)

plt.figure(1)
for i, value in enumerate(parameter_values[symbol[parameter]]):
    if parameter == 'learning_rate':
        label = f'$\{symbol[parameter]}$ = {value.replace("_", ".")}'
    else:
        label = f'${symbol[parameter]}$ = {value.replace("_", ".")}'
    plt.plot(param_rewards_rx[i], label=label)
plt.xlabel('Episode')
plt.ylabel('Average reward')
plt.title('Average reward during training, Tx agent')
plt.legend()

plt.figure(2)
for i, value in enumerate(parameter_values[symbol[parameter]]):
    if parameter == 'learning_rate':
        label = f'$\{symbol[parameter]}$ = {value.replace("_", ".")}'
    else:
        label = f'${symbol[parameter]}$ = {value.replace("_", ".")}'
    plt.plot(param_rewards_rx[i], label=label)
plt.xlabel('Episode')
plt.ylabel('Average reward')
plt.title('Average reward during training, Rx agent')
plt.legend()

plt.show()