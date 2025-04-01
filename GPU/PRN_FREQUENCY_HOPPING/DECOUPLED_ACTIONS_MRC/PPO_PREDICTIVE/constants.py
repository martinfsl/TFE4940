import numpy as np

#################################################################################
### Defining constants
#################################################################################

NUM_CHANNELS = 20 # Number of channels in the system
NUM_EXTRA_ACTIONS = 0 # Number of extra channels that the Tx and Rx can sense
NUM_EXTRA_RECEIVE = 2 # Number of extra channels that the Rx can receive on

# Hyperparameters
# LEARNING_RATE = 0.001
LEARNING_RATE = 0.0005
# LEARNING_RATE = 0.0001
GAMMA = 0.85
LAMBDA = 0.80
EPSILON_CLIP = 0.2
EPSILON = 0.3
EPSILON_MIN = 0
EPSILON_MIN_JAMMER = 0.1
EPSILON_REDUCTION = 0.99
EPSILON_REDUCTION_JAMMER = 0.999
T = 100 # Number of steps between each update, i.e. length of the trajectory
M = 20 # Size of mini-batch during training
K = 15 # Number of epochs
C1 = 0.5 # Coefficient for the value loss
C2 = 0.05 # Coefficient for the entropy loss
W = 0.5 # Weight for the Compound Action Loss

# Parameters
BATCH_SIZE = 2
MAXIMUM_MEMORY_SIZE = 100
MEMORY_SIZE_BEFORE_TRAINING = 2*BATCH_SIZE

# Rewards
REWARD_SUCCESSFUL = 1 # Reward
REWARD_INTERFERENCE = -1 # Penalty
REWARD_UNSUCCESSFUL = -1 # Penalty
REWARD_MISS = -1 # Penalty

# Number of episodes for training
NUM_EPISODES = 40000

# Number of runs for testing
NUM_TEST_RUNS = 40000

# Bool variable to decide whether fading is to be considered
CONSIDER_FADING = True
SINR_THRESHOLD = 5
USE_SINR_AS_REWARD = False

# User transmit power and variance
TX_USER_TRANSMIT_POWER = 6
RX_USER_TRANSMIT_POWER = 6
H_TR_VARIANCE = 1 # Variance of the Rayleigh distribution for the Rayleigh fading from transmitter to receiver

# Jammer transmit power and variance
JAMMER_TRANSMIT_POWER = 10
H_JT_VARIANCE = 1 # Variance of the Rayleigh distribution for the Rayleigh fading from jammer to transmitter
H_JR_VARIANCE = 1 # Variance of the Rayleigh distribution for the Rayleigh fading from jammer to receiver
# Note: If you want the receiver to see the jammer but not the transmitter, reduce H_JT_VARIANCE, or vice versa if otherwise

# Noise power at the receiver
NOISE_VARIANCE = 0.1 # Normalized power, everything is relative to this

# Tracking jammer parameters
TRANSITION_STAY = 0.8
TRANSITION_1 = 0.05
TRANSITION_2 = 0.03
TRANSITION_3 = 0.02

# Number of sensed channels
NUM_SENSE_CHANNELS = 10 # Number of channels the Tx and Rx can sense, including the channel they are on
NUM_JAMMER_SENSE_CHANNELS = NUM_CHANNELS # Number of channels the jammer can sense

REWARD_SENSE = 0.5 # Additional reward for being able to sense the other agent's action even though it did receive the message
PENALTY_NONDIVERSE = 0 # Penalty for staying on the same pattern for NUM_PREV_PATTERN episodes
REWARD_DIVERSE = 0 # Reward for choosing a pattern that has not been used in the past NUM_PREV_PATTERN episodes
NUM_PREV_PATTERNS = 2 # Number of previous patterns to consider for the penalty

# Frequency-Hopping parameters
NUM_HOPS = 5
NUM_SEEDS = 32

# Determining the state space size
TX_STATE_SPACE_SIZE = NUM_HOPS*(NUM_SENSE_CHANNELS + 1) + 1
RX_STATE_SPACE_SIZE = NUM_HOPS*(NUM_SENSE_CHANNELS + 1) + 1 + NUM_EXTRA_RECEIVE
#################################################################################
### Defining inputs and outputs for the neural networks
TX_PREDICTION_NETWORK_INPUT_SIZE = TX_STATE_SPACE_SIZE
RX_PREDICTION_NETWORK_INPUT_SIZE = RX_STATE_SPACE_SIZE
USE_PREDICTION = True
if USE_PREDICTION:
    TX_PPO_NETWORK_INPUT_SIZE = TX_STATE_SPACE_SIZE + NUM_SEEDS
    RX_PPO_NETWORK_INPUT_SIZE = RX_STATE_SPACE_SIZE + NUM_SEEDS
else:
    TX_PPO_NETWORK_INPUT_SIZE = TX_STATE_SPACE_SIZE
    RX_PPO_NETWORK_INPUT_SIZE = RX_STATE_SPACE_SIZE

JAMMER_PPO_NETWORK_INPUT_SIZE = NUM_JAMMER_SENSE_CHANNELS + 1
#################################################################################