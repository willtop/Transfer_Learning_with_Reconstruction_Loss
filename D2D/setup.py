import os
import random
import numpy as np
import torch
# For windows specific error
os.environ['KMP_DUPLICATE_LIB_OK']='True'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE)

# Source task trained agnositic to target task
LAYOUT_SETTING = 'A'
SourceTask_HarmonicMean = {'Type': 'Source-Task',
        'Task': 'Harmonic',
        'Fullname': 'Harmonic-Mean-Rate',
        'Train': int(1e6),
        'Valid': 5000,
        'Minibatch_Size': 2000,
        'Learning_Rate': 1e-3,
        'Epochs': 100,
        'Loss_Combine_Weight': 0.2}
TargetTask_Min = {'Type': 'Target-Task',
        'Task': 'Min',
        'Fullname': 'Min-Rate',
        'Train': 1000,
        'Valid': 5000,
        'Minibatch_Size': 100,
        'Learning_Rate': 2e-5,
        'Epochs': 15000}
TargetTask_Sum = {'Type': 'Target-Task',
        'Task': 'Sum',
        'Fullname': 'Sum-Rate',
        'Train': 1000,
        'Valid': 5000,
        'Minibatch_Size': 100,
        'Learning_Rate': 2e-5,
        'Epochs': 15000}

# Transfer Configuration on Task Specifications
TRANSFER_CONFIGURE = 'I'

if TRANSFER_CONFIGURE == 'I':
    SOURCETASK = SourceTask_HarmonicMean
    TARGETTASK = TargetTask_Min
elif TRANSFER_CONFIGURE == 'II':
    SOURCETASK = SourceTask_HarmonicMean
    TARGETTASK = TargetTask_Sum
else:
    print(f"Invalid Transfer Configuration Option: {TRANSFER_CONFIGURE}! Exiting...")
    exit(1)
N_TEST_SAMPLES = 2000

assert SOURCETASK['Train'] % SOURCETASK['Minibatch_Size'] == 0 and \
       TARGETTASK['Train'] % TARGETTASK['Minibatch_Size'] == 0

# millimeter wave environment settings
BANDWIDTH = 5e6
CARRIER_FREQUENCY = 50e9 # 50GHz carrier frequency for millimeter wave
WAVELENGTH = 2.998e8/CARRIER_FREQUENCY
_NOISE_dBm_Hz = -150
NOISE_POWER = np.power(10, ((_NOISE_dBm_Hz-30)/10)) * BANDWIDTH
if LAYOUT_SETTING=='A':
    N_LINKS = 10
    FIELD_LENGTH = 150
    SHORTEST_DIRECTLINK = 5
    LONGEST_DIRECTLINK = 30
elif LAYOUT_SETTING=='B':
    N_LINKS = 15
    FIELD_LENGTH = 250
    SHORTEST_DIRECTLINK = 10
    LONGEST_DIRECTLINK = 35
else:
    print(f"Wrong Layout Setting {LAYOUT_SETTING}!")
    exit(1)
SHORTEST_CROSSLINK = 5
TX_HEIGHT = 1.5
RX_HEIGHT = 1.5
_TX_POWER_dBm = 30
TX_POWER = np.power(10, (_TX_POWER_dBm - 30) / 10)
SETTING_STRING = "N{}_L{}_{}-{}m".format(N_LINKS, FIELD_LENGTH, SHORTEST_DIRECTLINK, LONGEST_DIRECTLINK)
SINR_GAP_dB = 0
SINR_GAP = np.power(10, SINR_GAP_dB/10)
ANTENNA_GAIN_DB = 6


# set random seed
RANDOM_SEED = 123
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
