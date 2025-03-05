import numpy as np

#################################################################################
### Defining constants
#################################################################################

NUM_CHANNELS = 20
NUM_USERS = 1
ACTIONS = np.arange(0, NUM_CHANNELS, 1)
NUM_EXTRA_ACTIONS = 5 # Number of extra channels that the Tx and Rx can sense

channel = [0]*NUM_CHANNELS

# Hyperparameters
LEARNING_RATE = 0.005
GAMMA = 0.35
EPSILON = 0.3
EPSILON_MIN = 0
EPSILON_MIN_JAMMER = 0.1
EPSILON_REDUCTION = 0.99
EPSILON_REDUCTION_JAMMER = 0.999

# Parameters
BATCH_SIZE = 2
MAXIMUM_MEMORY_SIZE = 100
MEMORY_SIZE_BEFORE_TRAINING = 2*BATCH_SIZE

# Determining the state space size
STATE_SPACE_SIZE = NUM_CHANNELS + 1

# Rewards
REWARD_SUCCESSFUL = 1
REWARD_INTERFERENCE = -1
REWARD_UNSUCCESSFUL = -1

# Number of episodes for training
# NUM_EPISODES = 500
# NUM_EPISODES = 20000
# NUM_EPISODES = 50000
NUM_EPISODES = 100000

# Number of runs for testing
NUM_TEST_RUNS = 10000
# NUM_TEST_RUNS = 50000
# NUM_TEST_RUNS = 100000

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

# Spectrum sensing jammer parameters
TRANSITION_STAY = 0.8
TRANSITION_1 = 0.05
TRANSITION_2 = 0.03
TRANSITION_3 = 0.02

# Number of sensed channels
# How many channels the Tx and Rx can sense at a time (including the channel they are on)
# NUM_SENSE_CHANNELS = 15
NUM_SENSE_CHANNELS = NUM_CHANNELS # This allows the Tx and Rx to sense all channels
NUM_JAMMER_SENSE_CHANNELS = 5 # Number of channels the jammer can sense