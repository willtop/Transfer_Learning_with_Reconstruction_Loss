import os
import random
import numpy as np
import torch
# For windows specific error
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# millimeter wave environment settings
SETTING = 'A'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE)
BANDWIDTH = 5e6
CARRIER_FREQUENCY = 50e9 # 50GHz carrier frequency for millimeter wave
WAVELENGTH = 2.998e8/CARRIER_FREQUENCY
_NOISE_dBm_Hz = -150
NOISE_POWER = np.power(10, ((_NOISE_dBm_Hz-30)/10)) * BANDWIDTH
if SETTING=='A':
    N_LINKS = 10
    FIELD_LENGTH = 150
    SHORTEST_DIRECTLINK = 5
    LONGEST_DIRECTLINK = 30
elif SETTING=='B':
    N_LINKS = 15
    FIELD_LENGTH = 200
    SHORTEST_DIRECTLINK = 20
    LONGEST_DIRECTLINK = 30
else:
    print(f"Wrong Setting {SETTING}!")
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

# Transfer Configuration on Task Specifications
TRANSFER_CONFIGURE = 'A'

if TRANSFER_CONFIGURE == 'A':
    SOURCETASK = {'Type': 'Source-Task',
        'Task': 'Sum-Rate',
        'Train': int(5e5),
        'Valid': 5000}
    TARGETTASK = {'Type': 'Target-Task',
        'Task': 'Min-Rate',
        'Train': int(1000),
        'Valid': 2000} 
elif TRANSFER_CONFIGURE == 'B':
    SOURCETASK = {'Type': 'Source-Task',
        'Task': 'Sum-Rate',
        'Train': int(5e5),
        'Valid': 5000}
    TARGETTASK = {'Type': 'Target-Task',
        'Task': 'Jain-Fairness',
        'Train': int(1000),
        'Valid': 2000}
elif TRANSFER_CONFIGURE == 'C':
    SOURCETASK = {'Type': 'Source-Task',
        'Task': 'Sum-Rate',
        'Train': int(5e5),
        'Valid': 5000}
    TARGETTASK = {'Type': 'Target-Task',
        'Task': 'Harmonic',
        'Train': int(1000),
        'Valid': 2000}
else:
    print(f"Invalid Transfer Configuration Option: {TRANSFER_CONFIGURE}! Exiting...")
    exit(1)

N_TEST_SAMPLES = 2000


# set random seed
RANDOM_SEED = 123
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
