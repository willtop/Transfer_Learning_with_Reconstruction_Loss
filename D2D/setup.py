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
_NOISE_dBm_Hz = -169
NOISE_POWER = np.power(10, ((_NOISE_dBm_Hz-30)/10)) * BANDWIDTH
if SETTING=='A':
    N_LINKS = 10
    FIELD_LENGTH = 100
    SHORTEST_DIRECTLINK = 10
    LONGEST_DIRECTLINK = 20
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

# number of samples 
# Note: the testing layouts generated in the "MinRate" is used to test both sum rate and min rate
N_SAMPLES = {'SourceTask':{
    'Train': int(1e6),
    'Valid': 5000
}, 'TargetTask': {
    'Train': int(1000),
    'Valid': 2000
}, 'Test': 1000
}

# set random seed
RANDOM_SEED = 1234
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
