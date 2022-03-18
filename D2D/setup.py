import os
import random
import numpy as np
import torch

# millimeter wave environment settings
SETTING = 'A'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BANDWIDTH = 5e6
CARRIER_FREQUENCY = 50e9 # 50GHz carrier frequency for millimeter wave
WAVELENGTH = 2.998e8/CARRIER_FREQUENCY
_NOISE_dBm_Hz = -169
NOISE_POWER = np.power(10, ((_NOISE_dBm_Hz-30)/10)) * BANDWIDTH
if SETTING=='A':
    N_LINKS = 10
    FIELD_LENGTH = 150
    SHORTEST_DIRECTLINK = 5
    LONGEST_DIRECTLINK = 15
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
ANTENNA_GAIN_DB = 2.5

# number of samples 
# Note: the testing layouts generated in the "MinRate" is used to test both sum rate and min rate
N_SAMPLES = {'SumRate':{
    'Train': int(1e6),
    'Valid': 5000
}, 'MinRate': {
    'Train': int(15e3),
    'Valid': 0,#2000,
    'Test': 0#2000
}}

# set random seed
RANDOM_SEED = 123
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
