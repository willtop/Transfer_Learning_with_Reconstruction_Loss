import os
import random
import numpy as np
import torch
# For windows specific error
os.environ['KMP_DUPLICATE_LIB_OK']='True'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE)

# Transfer Configuration on Task Specifications
TRANSFER_CONFIGURE = 'IV'

if TRANSFER_CONFIGURE == 'I':
    SOURCETASK = {'Type': 'Source-Task',
        'Task': 'Sum',
        'Fullname': 'Sum-Rate',
        'Train': int(5e5),
        'Valid': 5000}
    TARGETTASK = {'Type': 'Target-Task',
        'Task': 'Min',
        'Fullname': 'Min-Rate',
        'Train': int(1000),
        'Valid': 5000} 
    LAYOUT_SETTING = 'A'
    COMBINE_WEIGHT_RECONSTRUCT = 3
    LEARNING_RATE_SOURCETASK = 1e-3
    N_EPOCHES_SOURCETASK = 300
    LEARNING_RATE_TARGETTASK = 1e-5
    N_EPOCHES_TARGETTASK = 10000
elif TRANSFER_CONFIGURE == 'II':
    SOURCETASK = {'Type': 'Source-Task',
        'Task': 'Sum',
        'Fullname': 'Sum-Rate',
        'Train': int(5e5),
        'Valid': 5000}
    TARGETTASK = {'Type': 'Target-Task',
        'Task': 'Harmonic',
        'Fullname': 'Harmonic-Mean-Rate',
        'Train': int(1000),
        'Valid': 5000}
    LAYOUT_SETTING = 'A'
    COMBINE_WEIGHT_RECONSTRUCT = 4
    LEARNING_RATE_SOURCETASK = 1e-3
    N_EPOCHES_SOURCETASK = 300
    LEARNING_RATE_TARGETTASK = 1e-5
    N_EPOCHES_TARGETTASK = 20000
elif TRANSFER_CONFIGURE == 'III':
    SOURCETASK = {'Type': 'Source-Task',
        'Task': 'Harmonic',
        'Fullname': 'Harmonic-Mean-Rate',
        'Train': int(5e5),
        'Valid': 5000}
    TARGETTASK = {'Type': 'Target-Task',
        'Task': 'Min',
        'Fullname': 'Min-Rate',
        'Train': int(1000),
        'Valid': 5000} 
    LAYOUT_SETTING = 'A'
    COMBINE_WEIGHT_RECONSTRUCT = 0.1
    LEARNING_RATE_SOURCETASK = 1e-3
    N_EPOCHES_SOURCETASK = 150
    LEARNING_RATE_TARGETTASK = 2e-5
    N_EPOCHES_TARGETTASK = 30000
elif TRANSFER_CONFIGURE == 'IV':
    SOURCETASK = {'Type': 'Source-Task',
        'Task': 'Harmonic',
        'Fullname': 'Harmonic-Mean-Rate',
        'Train': int(5e5),
        'Valid': 5000}
    TARGETTASK = {'Type': 'Target-Task',
        'Task': 'Sum',
        'Fullname': 'Sum-Rate',
        'Train': int(1000),
        'Valid': 5000} 
    LAYOUT_SETTING = 'A'
    COMBINE_WEIGHT_RECONSTRUCT = 1
    LEARNING_RATE_SOURCETASK = 1e-3
    N_EPOCHES_SOURCETASK = 150
    LEARNING_RATE_TARGETTASK = 2e-5
    N_EPOCHES_TARGETTASK = 30000
else:
    print(f"Invalid Transfer Configuration Option: {TRANSFER_CONFIGURE}! Exiting...")
    exit(1)
N_TEST_SAMPLES = 2000


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
