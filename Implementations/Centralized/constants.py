import numpy as np

#################################################################################
### Defining constants
#################################################################################

NUM_CHANNELS = 20 # Number of channels in the system
NUM_EXTRA_ACTIONS = 0 # Number of extra channels that the Tx and Rx can sense
NUM_EXTRA_RECEIVE = 0 # Number of extra channels that the Rx can receive on

# Hyperparameters
LEARNING_RATE = 0.0005
GAMMA = 0.85
LAMBDA = 0.80
EPSILON_CLIP = 0.2
T = 50 # Number of steps between each update, i.e. length of the trajectory
M = 30 # Size of mini-batch during training
K = 5 # Number of epochs
C1 = 0.7 # Coefficient for the value loss
C2 = 0.1 # Coefficient for the entropy loss
C3 = 0.2 # Coefficient for the belief loss
P = 20 # Number of time steps that the prediction observation contains

# Smart jammer hyperparameters
JAMMER_LEARNING_RATE = LEARNING_RATE
JAMMER_GAMMA = GAMMA
JAMMER_LAMBDA = LAMBDA
JAMMER_EPSILON_CLIP = EPSILON_CLIP
JAMMER_C1 = C1 # Coefficient for the value loss
JAMMER_C2 = C2 # Coefficient for the entropy loss

BELIEF_MODULE_LEARNING_RATE = 0.005
BELIEF_MODULE_BATCH_SIZE = 10
BELIEF_MODULE_MEMORY_SIZE = 100
BELIEF_MODULE_EPOCHS = 5

# Rewards
REWARD_SUCCESSFUL = 1 # Reward
REWARD_INTERFERENCE = -1 # Penalty
REWARD_UNSUCCESSFUL = -1 # Penalty
REWARD_MISS = -1 # Penalty

# Number of episodes for training
# NUM_EPISODES = 1000
NUM_EPISODES = 100000
# NUM_EPISODES = 40000

# Number of runs for testing
# NUM_TEST_RUNS = 1000
NUM_TEST_RUNS = 100000
# NUM_TEST_RUNS = 40000

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
# NUM_SENSE_CHANNELS = NUM_CHANNELS # Number of channels the Tx and Rx can sense, including the channel they are on
NUM_JAMMER_SENSE_CHANNELS = NUM_CHANNELS # Number of channels the jammer can sense

REWARD_SENSE = 0.5 # Additional reward for being able to sense the other agent's action even though it did receive the message
PENALTY_NONDIVERSE = 0 # Penalty for staying on the same pattern for NUM_PREV_PATTERN episodes
REWARD_DIVERSE = 0 # Reward for choosing a pattern that has not been used in the past NUM_PREV_PATTERN episodes
NUM_PREV_PATTERNS = 2 # Number of previous patterns to consider for the penalty

# Frequency-Hopping parameters
NUM_HOPS = 1
if NUM_HOPS == 1:
    NUM_SEEDS = NUM_CHANNELS
else:
    NUM_SEEDS = 32

# Determining the state space size
STATE_SPACE_SIZE = NUM_HOPS*(NUM_SENSE_CHANNELS + 1) + 1

#################################################################################
### Defining input and output sizes for the neural networks
USE_PREDICTION = True
# USE_PREDICTION = False

USE_STANDALONE_BELIEF_MODULE = True
# USE_STANDALONE_BELIEF_MODULE = False

PPO_NETWORK_INPUT_SIZE = STATE_SPACE_SIZE
if NUM_HOPS == 1:
    PPO_NETWORK_OUTPUT_SIZE = NUM_CHANNELS
else:
    PPO_NETWORK_OUTPUT_SIZE = NUM_SEEDS

SENSING_NETWORK_INPUT_SIZE = STATE_SPACE_SIZE
# PREDICTION_NETWORK_INPUT_SIZE = STATE_SPACE_SIZE + 1
if USE_PREDICTION:
    PREDICTION_NETWORK_INPUT_SIZE = P
    
    if NUM_HOPS == 1:
        PREDICTION_NETWORK_OUTPUT_SIZE = NUM_CHANNELS
    else:
        PREDICTION_NETWORK_OUTPUT_SIZE = NUM_SEEDS

else:
    PREDICTION_NETWORK_INPUT_SIZE = 0
    PREDICTION_NETWORK_OUTPUT_SIZE = 0

JAMMER_PPO_NETWORK_INPUT_SIZE = NUM_JAMMER_SENSE_CHANNELS + 1
#################################################################################