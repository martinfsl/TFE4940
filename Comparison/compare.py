import numpy as np
import matplotlib.pyplot as plt

# NUM_EPISODES = 50000
NUM_EPISODES = 100000
NUM_CHANNELS = 100

#################################################################################
# only_tx = np.loadtxt(f'research_ting/old_simulation/aa_performance/{NUM_EPISODES}_episodes/{NUM_CHANNELS}_channels/average_reward_only_tx.txt', dtype=float)
# iddqn_tx = np.loadtxt(f'IDDQN/aa_performance/{NUM_EPISODES}_episodes/{NUM_CHANNELS}_channels/average_reward_both_tx.txt', dtype=float)
# iddqn_rx = np.loadtxt(f'IDDQN/aa_performance/{NUM_EPISODES}_episodes/{NUM_CHANNELS}_channels/average_reward_both_rx.txt', dtype=float)
#################################################################################

#################################################################################
# relative_path = f"Comparison/11_02"

# iddqn_tx_with_min = np.loadtxt(f"{relative_path}/IDDQN_with_min/average_reward_both_tx.txt", dtype=float)
# iddqn_rx_with_min = np.loadtxt(f"{relative_path}/IDDQN_with_min/average_reward_both_rx.txt", dtype=float)
# iddqn_tx_without_min = np.loadtxt(f"{relative_path}/IDDQN_without_min/average_reward_both_tx.txt", dtype=float)
# iddqn_rx_without_min = np.loadtxt(f"{relative_path}/IDDQN_without_min/average_reward_both_rx.txt", dtype=float)
#################################################################################

#################################################################################
# jammer = "tracking"
# jammer = "sweeping"
# relative_path = f"Comparison/Basic_vs_Realistic_Sensing/{jammer}"

# iddqn_tx_basic = np.loadtxt(f"{relative_path}/Basic/average_reward_both_tx.txt", dtype=float)
# iddqn_rx_basic = np.loadtxt(f"{relative_path}/Basic/average_reward_both_rx.txt", dtype=float)
# iddqn_tx_realistic_5 = np.loadtxt(f"{relative_path}/Realistic_Sensing_5/average_reward_both_tx.txt", dtype=float)
# iddqn_rx_realistic_5 = np.loadtxt(f"{relative_path}/Realistic_Sensing_5/average_reward_both_rx.txt", dtype=float)
# iddqn_tx_realistic_15 = np.loadtxt(f"{relative_path}/Realistic_Sensing_15/average_reward_both_tx.txt", dtype=float)
# iddqn_rx_realistic_15 = np.loadtxt(f"{relative_path}/Realistic_Sensing_15/average_reward_both_rx.txt", dtype=float)

# iddqn_tx_basic_normalized = (iddqn_tx_basic + 1)/2
# iddqn_rx_basic_normalized = (iddqn_rx_basic + 1)/2
# iddqn_tx_realistic_5_normalized = (iddqn_tx_realistic_5 + 1)/2
# iddqn_rx_realistic_5_normalized = (iddqn_rx_realistic_5 + 1)/2
# iddqn_tx_realistic_15_normalized = (iddqn_tx_realistic_15 + 1)/2
# iddqn_rx_realistic_15_normalized = (iddqn_rx_realistic_15 + 1)/2
#################################################################################

#################################################################################
# relative_path_tracking = f"Comparison/Basic_vs_Realistic_Sensing/Tracking"
# relative_path_1_sweep = f"Comparison/Basic_vs_Realistic_Sensing/1-sweep"

# iddqn_tx_basic_tracking = np.loadtxt(f"{relative_path_tracking}/Basic/average_reward_both_tx.txt", dtype=float)
# iddqn_rx_basic_tracking = np.loadtxt(f"{relative_path_tracking}/Basic/average_reward_both_rx.txt", dtype=float)
# iddqn_tx_realistic_5_tracking = np.loadtxt(f"{relative_path_tracking}/Realistic_Sensing_5/average_reward_both_tx.txt", dtype=float)
# iddqn_rx_realistic_5_tracking = np.loadtxt(f"{relative_path_tracking}/Realistic_Sensing_5/average_reward_both_rx.txt", dtype=float)
# iddqn_tx_realistic_15_tracking = np.loadtxt(f"{relative_path_tracking}/Realistic_Sensing_15/average_reward_both_tx.txt", dtype=float)
# iddqn_rx_realistic_15_tracking = np.loadtxt(f"{relative_path_tracking}/Realistic_Sensing_15/average_reward_both_rx.txt", dtype=float)

# iddqn_tx_basic_1_sweep = np.loadtxt(f"{relative_path_1_sweep}/Basic/average_reward_both_tx.txt", dtype=float)
# iddqn_rx_basic_1_sweep = np.loadtxt(f"{relative_path_1_sweep}/Basic/average_reward_both_rx.txt", dtype=float)
# iddqn_tx_realistic_5_1_sweep = np.loadtxt(f"{relative_path_1_sweep}/Realistic_Sensing_5/average_reward_both_tx.txt", dtype=float)
# iddqn_rx_realistic_5_1_sweep = np.loadtxt(f"{relative_path_1_sweep}/Realistic_Sensing_5/average_reward_both_rx.txt", dtype=float)
# iddqn_tx_realistic_15_1_sweep = np.loadtxt(f"{relative_path_1_sweep}/Realistic_Sensing_15/average_reward_both_tx.txt", dtype=float)
# iddqn_rx_realistic_15_1_sweep = np.loadtxt(f"{relative_path_1_sweep}/Realistic_Sensing_15/average_reward_both_rx.txt", dtype=float)

# # Normalize the average rewards
# iddqn_tx_basic_tracking_normalized = (iddqn_tx_basic_tracking + 1)/2
# iddqn_tx_basic_1_sweep_normalized = (iddqn_tx_basic_1_sweep + 1)/2
# iddqn_tx_realistic_5_tracking_normalized = (iddqn_tx_realistic_5_tracking + 1)/2
# iddqn_tx_realistic_5_1_sweep_normalized = (iddqn_tx_realistic_5_1_sweep + 1)/2
# iddqn_tx_realistic_15_tracking_normalized = (iddqn_tx_realistic_15_tracking + 1)/2
# iddqn_tx_realistic_15_1_sweep_normalized = (iddqn_tx_realistic_15_1_sweep + 1)/2
#################################################################################

#################################################################################
# relative_path = f"Comparison/Basic_vs_Realistic_Sensing/tracking"

# iddqn_rx_realistic_15_v1 = np.loadtxt(f"{relative_path}/Realistic_Sensing_15/average_reward_both_rx.txt", dtype=float)
# iddqn_tx_realistic_15_v1 = np.loadtxt(f"{relative_path}/Realistic_Sensing_15/average_reward_both_tx.txt", dtype=float)
# iddqn_tx_realistic_15_v2 = np.loadtxt(f"{relative_path}/Realistic_Sensing_15/average_reward_both_tx_v2.txt", dtype=float)
# iddqn_rx_realistic_15_v2 = np.loadtxt(f"{relative_path}/Realistic_Sensing_15/average_reward_both_rx_v2.txt", dtype=float)
# iddqn_tx_realistic_15_v3 = np.loadtxt(f"{relative_path}/Realistic_Sensing_15/average_reward_both_tx_v3.txt", dtype=float)
# iddqn_rx_realistic_15_v3 = np.loadtxt(f"{relative_path}/Realistic_Sensing_15/average_reward_both_rx_v3.txt", dtype=float)
#################################################################################

#################################################################################
# relative_path = f"Comparison/15_02/observation_vs_no_observation/100_channels"

# iddqn_tx_observation = np.loadtxt(f"{relative_path}/average_reward_both_tx_observation_15.txt", dtype=float)
# iddqn_rx_observation = np.loadtxt(f"{relative_path}/average_reward_both_rx_observation_15.txt", dtype=float)
# iddqn_tx_observation_v2 = np.loadtxt(f"{relative_path}/average_reward_both_tx_observation_15_v2.txt", dtype=float)
# iddqn_rx_observation_v2 = np.loadtxt(f"{relative_path}/average_reward_both_rx_observation_15_v2.txt", dtype=float)
# iddqn_tx_no_observation = np.loadtxt(f"{relative_path}/average_reward_both_tx_no_observation_15.txt", dtype=float)
# iddqn_rx_no_observation = np.loadtxt(f"{relative_path}/average_reward_both_rx_no_observation_15.txt", dtype=float)

# iddqn_tx_observation = np.loadtxt(f"{relative_path}/average_reward_both_tx_observation.txt", dtype=float)
# iddqn_rx_observation = np.loadtxt(f"{relative_path}/average_reward_both_rx_observation.txt", dtype=float)
# iddqn_tx_no_observation = np.loadtxt(f"{relative_path}/average_reward_both_tx_no_observation.txt", dtype=float)
# iddqn_rx_no_observation = np.loadtxt(f"{relative_path}/average_reward_both_rx_no_observation.txt", dtype=float)
#################################################################################

#################################################################################
# relative_path = f"Comparison/16_02/100_channels/20000_episodes"

# smart_tx = np.loadtxt(f"{relative_path}/smart/average_reward_both_tx_15.txt", dtype=float)
# smart_rx = np.loadtxt(f"{relative_path}/smart/average_reward_both_rx_15.txt", dtype=float)
# tracking_tx = np.loadtxt(f"{relative_path}/tracking/average_reward_both_tx_15.txt", dtype=float)
# tracking_rx = np.loadtxt(f"{relative_path}/tracking/average_reward_both_rx_15.txt", dtype=float)
# tracking_full_obs_tx = np.loadtxt(f"{relative_path}/tracking/average_reward_both_tx_full_obs.txt", dtype=float)
# tracking_full_obs_rx = np.loadtxt(f"{relative_path}/tracking/average_reward_both_rx_full_obs.txt", dtype=float)
# sweeping_tx = np.loadtxt(f"{relative_path}/sweeping/average_reward_both_tx_15.txt", dtype=float)
# sweeping_rx = np.loadtxt(f"{relative_path}/sweeping/average_reward_both_rx_15.txt", dtype=float)

#################################################################################

#################################################################################
# relative_path = f"Comparison/17_02/smart_full_class_vs_general"

# smart_tx_full_class = np.loadtxt(f"{relative_path}/full_class/average_reward_both_tx.txt", dtype=float)
# smart_rx_full_class = np.loadtxt(f"{relative_path}/full_class/average_reward_both_rx.txt", dtype=float)
# smart_tx_general = np.loadtxt(f"{relative_path}/general/average_reward_both_tx.txt", dtype=float)
# smart_rx_general = np.loadtxt(f"{relative_path}/general/average_reward_both_rx.txt", dtype=float)

# smart_tx_full_class_normalized = (smart_tx_full_class + 1)/2
# smart_rx_full_class_normalized = (smart_rx_full_class + 1)/2
# smart_tx_general_normalized = (smart_tx_general + 1)/2
# smart_rx_general_normalized = (smart_rx_general + 1)/2



#################################################################################

#################################################################################
# relative_path = f"Comparison/17_02/drqn_vs_ddqn"

# drqn_tx = np.loadtxt(f"{relative_path}/drqn/average_reward_both_tx.txt", dtype=float)
# drqn_rx = np.loadtxt(f"{relative_path}/drqn/average_reward_both_rx.txt", dtype=float)
# drqn_jammer = np.loadtxt(f"{relative_path}/drqn/average_reward_jammer.txt", dtype=float)
# ddqn_tx = np.loadtxt(f"{relative_path}/ddqn/average_reward_both_tx.txt", dtype=float)
# ddqn_rx = np.loadtxt(f"{relative_path}/ddqn/average_reward_both_rx.txt", dtype=float)
# ddqn_jammer = np.loadtxt(f"{relative_path}/ddqn/average_reward_jammer.txt", dtype=float)

# drqn_tx_normalized = (drqn_tx + 1)/2
# drqn_rx_normalized = (drqn_rx + 1)/2
# drqn_jammer_normalized = (drqn_jammer + 1)/2
# ddqn_tx_normalized = (ddqn_tx + 1)/2
# ddqn_rx_normalized = (ddqn_rx + 1)/2
# ddqn_jammer_normalized = (ddqn_jammer + 1)/2

#################################################################################

#################################################################################
# relative_path = f"Comparison/february_tests/23_02/directive_jamming_100000"

# iddqn_both_tx = np.loadtxt(f"{relative_path}/both_v2/average_reward_both_tx.txt", dtype=float)
# iddqn_both_rx = np.loadtxt(f"{relative_path}/both_v2/average_reward_both_rx.txt", dtype=float)

# iddqn_only_rx_tx = np.loadtxt(f"{relative_path}/only_rx/average_reward_both_tx.txt", dtype=float)
# iddqn_only_rx_rx = np.loadtxt(f"{relative_path}/only_rx/average_reward_both_rx.txt", dtype=float)

# iddqn_only_tx_tx = np.loadtxt(f"{relative_path}/only_tx/average_reward_both_tx.txt", dtype=float)
# iddqn_only_tx_rx = np.loadtxt(f"{relative_path}/only_tx/average_reward_both_rx.txt", dtype=float)

# iddqn_both_tx_normalized = (iddqn_both_tx + 1)/2
# iddqn_both_rx_normalized = (iddqn_both_rx + 1)/2
# iddqn_only_rx_tx_normalized = (iddqn_only_rx_tx + 1)/2
# iddqn_only_rx_rx_normalized = (iddqn_only_rx_rx + 1)/2
# iddqn_only_tx_tx_normalized = (iddqn_only_tx_tx + 1)/2
# iddqn_only_tx_rx_normalized = (iddqn_only_tx_rx + 1)/2

#################################################################################

#################################################################################
# relative_path = f"Comparison/february_tests/pred_vs_non_pred/no_pred"

# no_pred_sweeping_tx_20000 = np.loadtxt(f"{relative_path}/20000_episodes/sweeping/average_reward_both_tx.txt", dtype=float)
# no_pred_sweeping_rx_20000 = np.loadtxt(f"{relative_path}/20000_episodes/sweeping/average_reward_both_rx.txt", dtype=float)
# no_pred_tracking_tx_20000 = np.loadtxt(f"{relative_path}/20000_episodes/tracking/average_reward_both_tx.txt", dtype=float)
# no_pred_tracking_rx_20000 = np.loadtxt(f"{relative_path}/20000_episodes/tracking/average_reward_both_rx.txt", dtype=float)
# no_pred_smart_rnn_tx_20000 = np.loadtxt(f"{relative_path}/20000_episodes/smart_rnn/average_reward_both_tx.txt", dtype=float)
# no_pred_smart_rnn_rx_20000 = np.loadtxt(f"{relative_path}/20000_episodes/smart_rnn/average_reward_both_rx.txt", dtype=float)
# no_pred_smart_fnn_tx_20000 = np.loadtxt(f"{relative_path}/20000_episodes/smart_fnn/average_reward_both_tx.txt", dtype=float)
# no_pred_smart_fnn_rx_20000 = np.loadtxt(f"{relative_path}/20000_episodes/smart_fnn/average_reward_both_rx.txt", dtype=float)

# no_pred_sweeping_tx_20000_normalized = (no_pred_sweeping_tx_20000 + 1)/2
# no_pred_sweeping_rx_20000_normalized = (no_pred_sweeping_rx_20000 + 1)/2
# no_pred_tracking_tx_20000_normalized = (no_pred_tracking_tx_20000 + 1)/2
# no_pred_tracking_rx_20000_normalized = (no_pred_tracking_rx_20000 + 1)/2
# no_pred_smart_rnn_tx_20000_normalized = (no_pred_smart_rnn_tx_20000 + 1)/2
# no_pred_smart_rnn_rx_20000_normalized = (no_pred_smart_rnn_rx_20000 + 1)/2
# no_pred_smart_fnn_tx_20000_normalized = (no_pred_smart_fnn_tx_20000 + 1)/2
# no_pred_smart_fnn_rx_20000_normalized = (no_pred_smart_fnn_rx_20000 + 1)/2

# no_pred_sweeping_tx_100000 = np.loadtxt(f"{relative_path}/100000_episodes/sweeping/average_reward_both_tx.txt", dtype=float)
# no_pred_sweeping_rx_100000 = np.loadtxt(f"{relative_path}/100000_episodes/sweeping/average_reward_both_rx.txt", dtype=float)
# no_pred_tracking_tx_100000 = np.loadtxt(f"{relative_path}/100000_episodes/tracking/average_reward_both_tx.txt", dtype=float)
# no_pred_tracking_rx_100000 = np.loadtxt(f"{relative_path}/100000_episodes/tracking/average_reward_both_rx.txt", dtype=float)
# no_pred_smart_rnn_tx_100000 = np.loadtxt(f"{relative_path}/100000_episodes/smart_rnn/average_reward_both_tx.txt", dtype=float)
# no_pred_smart_rnn_rx_100000 = np.loadtxt(f"{relative_path}/100000_episodes/smart_rnn/average_reward_both_rx.txt", dtype=float)

# no_pred_sweeping_tx_100000_normalized = (no_pred_sweeping_tx_100000 + 1)/2
# no_pred_sweeping_rx_100000_normalized = (no_pred_sweeping_rx_100000 + 1)/2
# no_pred_tracking_tx_100000_normalized = (no_pred_tracking_tx_100000 + 1)/2
# no_pred_tracking_rx_100000_normalized = (no_pred_tracking_rx_100000 + 1)/2
# no_pred_smart_rnn_tx_100000_normalized = (no_pred_smart_rnn_tx_100000 + 1)/2
# no_pred_smart_rnn_rx_100000_normalized = (no_pred_smart_rnn_rx_100000 + 1)/2

#################################################################################


#################################################################################
relative_path = f"Comparison/february_tests/pred_vs_non_pred/"

no_pred_sweeping_tx_100000 = np.loadtxt(f"{relative_path}/no_pred/100000_episodes/sweeping/average_reward_both_tx.txt", dtype=float)
no_pred_sweeping_rx_100000 = np.loadtxt(f"{relative_path}/no_pred/100000_episodes/sweeping/average_reward_both_rx.txt", dtype=float)
no_pred_tracking_tx_100000 = np.loadtxt(f"{relative_path}/no_pred/100000_episodes/tracking/average_reward_both_tx.txt", dtype=float)
no_pred_tracking_rx_100000 = np.loadtxt(f"{relative_path}/no_pred/100000_episodes/tracking/average_reward_both_rx.txt", dtype=float)
no_pred_smart_rnn_tx_100000 = np.loadtxt(f"{relative_path}/no_pred/100000_episodes/smart_rnn/average_reward_both_tx.txt", dtype=float)
no_pred_smart_rnn_rx_100000 = np.loadtxt(f"{relative_path}/no_pred/100000_episodes/smart_rnn/average_reward_both_rx.txt", dtype=float)
no_pred_smart_fnn_tx_100000 = np.loadtxt(f"{relative_path}/no_pred/100000_episodes/smart_fnn/average_reward_both_tx.txt", dtype=float)
no_pred_smart_fnn_rx_100000 = np.loadtxt(f"{relative_path}/no_pred/100000_episodes/smart_fnn/average_reward_both_rx.txt", dtype=float)

no_pred_sweeping_tx_100000_normalized = (no_pred_sweeping_tx_100000 + 1)/2
no_pred_sweeping_rx_100000_normalized = (no_pred_sweeping_rx_100000 + 1)/2
no_pred_tracking_tx_100000_normalized = (no_pred_tracking_tx_100000 + 1)/2
no_pred_tracking_rx_100000_normalized = (no_pred_tracking_rx_100000 + 1)/2
no_pred_smart_rnn_tx_100000_normalized = (no_pred_smart_rnn_tx_100000 + 1)/2
no_pred_smart_rnn_rx_100000_normalized = (no_pred_smart_rnn_rx_100000 + 1)/2
no_pred_smart_fnn_tx_100000_normalized = (no_pred_smart_fnn_tx_100000 + 1)/2
no_pred_smart_fnn_rx_100000_normalized = (no_pred_smart_fnn_rx_100000 + 1)/2

rx_pred_sweeping_tx_100000 = np.loadtxt(f"{relative_path}/rx_pred/100000_episodes/sweeping/average_reward_both_tx.txt", dtype=float)
rx_pred_sweeping_rx_100000 = np.loadtxt(f"{relative_path}/rx_pred/100000_episodes/sweeping/average_reward_both_rx.txt", dtype=float)
rx_pred_tracking_tx_100000 = np.loadtxt(f"{relative_path}/rx_pred/100000_episodes/tracking/average_reward_both_tx.txt", dtype=float)
rx_pred_tracking_rx_100000 = np.loadtxt(f"{relative_path}/rx_pred/100000_episodes/tracking/average_reward_both_rx.txt", dtype=float)
# rx_pred_smart_rnn_tx_100000 = np.loadtxt(f"{relative_path}/rx_pred/100000_episodes/smart_rnn/average_reward_both_tx.txt", dtype=float)
# rx_pred_smart_rnn_rx_100000 = np.loadtxt(f"{relative_path}/rx_pred/100000_episodes/smart_rnn/average_reward_both_rx.txt", dtype=float)
rx_pred_smart_rnn_tx_100000 = np.loadtxt(f"{relative_path}/rx_pred/100000_episodes/smart_rnn_v2/average_reward_both_tx.txt", dtype=float)
rx_pred_smart_rnn_rx_100000 = np.loadtxt(f"{relative_path}/rx_pred/100000_episodes/smart_rnn_v2/average_reward_both_rx.txt", dtype=float)
rx_pred_smart_fnn_tx_100000 = np.loadtxt(f"{relative_path}/rx_pred/100000_episodes/smart_fnn/average_reward_both_tx.txt", dtype=float)
rx_pred_smart_fnn_rx_100000 = np.loadtxt(f"{relative_path}/rx_pred/100000_episodes/smart_fnn/average_reward_both_rx.txt", dtype=float)

rx_pred_sweeping_tx_100000_normalized = (rx_pred_sweeping_tx_100000 + 1)/2
rx_pred_sweeping_rx_100000_normalized = (rx_pred_sweeping_rx_100000 + 1)/2
rx_pred_tracking_tx_100000_normalized = (rx_pred_tracking_tx_100000 + 1)/2
rx_pred_tracking_rx_100000_normalized = (rx_pred_tracking_rx_100000 + 1)/2
rx_pred_smart_rnn_tx_100000_normalized = (rx_pred_smart_rnn_tx_100000 + 1)/2
rx_pred_smart_rnn_rx_100000_normalized = (rx_pred_smart_rnn_rx_100000 + 1)/2
rx_pred_smart_fnn_tx_100000_normalized = (rx_pred_smart_fnn_tx_100000 + 1)/2
rx_pred_smart_fnn_rx_100000_normalized = (rx_pred_smart_fnn_rx_100000 + 1)/2

both_pred_sweeping_tx_100000 = np.loadtxt(f"{relative_path}/both_pred/100000_episodes/sweeping/average_reward_both_tx.txt", dtype=float)
both_pred_sweeping_rx_100000 = np.loadtxt(f"{relative_path}/both_pred/100000_episodes/sweeping/average_reward_both_rx.txt", dtype=float)
both_pred_tracking_tx_100000 = np.loadtxt(f"{relative_path}/both_pred/100000_episodes/tracking/average_reward_both_tx.txt", dtype=float)
both_pred_tracking_rx_100000 = np.loadtxt(f"{relative_path}/both_pred/100000_episodes/tracking/average_reward_both_rx.txt", dtype=float)
# both_pred_smart_rnn_tx_100000 = np.loadtxt(f"{relative_path}/both_pred/100000_episodes/smart_rnn/average_reward_both_tx.txt", dtype=float)
# both_pred_smart_rnn_rx_100000 = np.loadtxt(f"{relative_path}/both_pred/100000_episodes/smart_rnn/average_reward_both_rx.txt", dtype=float)
both_pred_smart_rnn_tx_100000 = np.loadtxt(f"{relative_path}/both_pred/100000_episodes/smart_rnn_v2/average_reward_both_tx.txt", dtype=float)
both_pred_smart_rnn_rx_100000 = np.loadtxt(f"{relative_path}/both_pred/100000_episodes/smart_rnn_v2/average_reward_both_rx.txt", dtype=float)
both_pred_smart_fnn_tx_100000 = np.loadtxt(f"{relative_path}/both_pred/100000_episodes/smart_fnn/average_reward_both_tx.txt", dtype=float)
both_pred_smart_fnn_rx_100000 = np.loadtxt(f"{relative_path}/both_pred/100000_episodes/smart_fnn/average_reward_both_rx.txt", dtype=float)

both_pred_sweeping_tx_100000_normalized = (both_pred_sweeping_tx_100000 + 1)/2
both_pred_sweeping_rx_100000_normalized = (both_pred_sweeping_rx_100000 + 1)/2
both_pred_tracking_tx_100000_normalized = (both_pred_tracking_tx_100000 + 1)/2
both_pred_tracking_rx_100000_normalized = (both_pred_tracking_rx_100000 + 1)/2
both_pred_smart_rnn_tx_100000_normalized = (both_pred_smart_rnn_tx_100000 + 1)/2
both_pred_smart_rnn_rx_100000_normalized = (both_pred_smart_rnn_rx_100000 + 1)/2
both_pred_smart_fnn_tx_100000_normalized = (both_pred_smart_fnn_tx_100000 + 1)/2
both_pred_smart_fnn_rx_100000_normalized = (both_pred_smart_fnn_rx_100000 + 1)/2

no_pred_tx = {"sweeping": no_pred_sweeping_tx_100000_normalized, 
              "tracking": no_pred_tracking_tx_100000_normalized, 
              "smart_rnn": no_pred_smart_rnn_tx_100000_normalized, 
              "smart_fnn": no_pred_smart_fnn_tx_100000_normalized}

no_pred_rx = {"sweeping": no_pred_sweeping_rx_100000_normalized,
              "tracking": no_pred_tracking_rx_100000_normalized,
              "smart_rnn": no_pred_smart_rnn_rx_100000_normalized,
              "smart_fnn": no_pred_smart_fnn_rx_100000_normalized}

rx_pred_tx = {"sweeping": rx_pred_sweeping_tx_100000_normalized,
              "tracking": rx_pred_tracking_tx_100000_normalized,
              "smart_rnn": rx_pred_smart_rnn_tx_100000_normalized,
              "smart_fnn": rx_pred_smart_fnn_tx_100000_normalized}

rx_pred_rx = {"sweeping": rx_pred_sweeping_rx_100000_normalized,
              "tracking": rx_pred_tracking_rx_100000_normalized,
              "smart_rnn": rx_pred_smart_rnn_rx_100000_normalized,
              "smart_fnn": rx_pred_smart_fnn_rx_100000_normalized}

both_pred_tx = {"sweeping": both_pred_sweeping_tx_100000_normalized,
                "tracking": both_pred_tracking_tx_100000_normalized,
                "smart_rnn": both_pred_smart_rnn_tx_100000_normalized,
                "smart_fnn": both_pred_smart_fnn_tx_100000_normalized}

both_pred_rx = {"sweeping": both_pred_sweeping_rx_100000_normalized,
                "tracking": both_pred_tracking_rx_100000_normalized,
                "smart_rnn": both_pred_smart_rnn_rx_100000_normalized,
                "smart_fnn": both_pred_smart_fnn_rx_100000_normalized}

# For plotting:
jammer = "smart_rnn"

#################################################################################




plt.figure(1)

plt.plot(no_pred_tx[jammer], label = f"No prediction - Tx - {jammer}")
# plt.plot(no_pred_rx[jammer], label = f"No prediction - Rx - {jammer}")

plt.plot(rx_pred_tx[jammer], label = f"Rx prediction - Tx - {jammer}")
# plt.plot(rx_pred_rx[jammer], label = f"Rx prediction - Rx - {jammer}")

plt.plot(both_pred_tx[jammer], label = f"Both prediction - Tx - {jammer}")
# plt.plot(both_pred_rx[jammer], label = f"Both prediction - Rx - {jammer}")

plt.xlabel('Episode')
plt.ylabel('Normalized average reward')
plt.title(f'Normalized average reward during training')
plt.legend()
plt.show()
