import numpy as np

#################################################################################
### Defining constants
#################################################################################

NUM_CHANNELS = 10
NUM_USERS = 1
ACTIONS = np.arange(0, NUM_CHANNELS, 1)

channel = [0]*NUM_CHANNELS

# Hyperparameters
GAMMA = 0.35
EPSILON = 0.3
EPSILON_MIN = 0.01
EPSILON_REDUCTION = 0.995
TAU = 0.01

# Hyperparameters for DQN
LEARNING_RATE = 0.005
DQN_BATCH_SIZE = 16
MAXIMUM_MEMORY_SIZE = 100
MEMORY_SIZE_BEFORE_TRAINING = 2*DQN_BATCH_SIZE

# Determining the state space size
STATE_SPACE_SIZE = NUM_CHANNELS + 1

# Rewards
REWARD_SUCCESSFUL = 1
REWARD_INTERFERENCE = -1

# Number of episodes for training
NUM_EPISODES = 5000

# Number of runs for testing
NUM_TEST_RUNS = 10000

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