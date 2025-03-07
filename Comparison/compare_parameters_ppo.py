import numpy as np
import matplotlib.pyplot as plt

start_path = 'Comparison/february_tests'
algorithm = 'PPO'
parameter = 'trajectory_length'


symbol = {'learning_rate': 'alpha',
          'discount_factor': '\gamma',
          'clipping': '\epsilon_{clip}',
          'truncate': '\lambda',
          'trajectory_length': 'T',}

parameter_values = {'\gamma': ['0_20', '0_35', '0_50', '0_75', '0_90', '0_95'],
                    '\lambda': ['0_3', '0_4', '0_5', '0_7', '0_8', '0_95', '1_0'],
                    'T': ['5', '10', '25', '50', '100', '200', '500', '1000']}

param_rewards_tx = []
param_rewards_rx = []

relative_path = f'{start_path}/{algorithm}_parameter_tuning/{parameter}'

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