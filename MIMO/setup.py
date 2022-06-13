import numpy as np
import torch
import os
import random
# For windows specific error
os.environ['KMP_DUPLICATE_LIB_OK']='True'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device for torch: ", DEVICE)

"""
Environment Setting
"""
N_BS = 3
N_BS_ANTENNAS = 16
ANTENNA_SPACING_PHASE_SHIFT = 1
N_PILOTS = 4
FIELD_LENGTH = 40
FIELD_HEIGHT = 20
BS_LOCATIONS = np.array([[0,0,20], [0,40,20], [40,0,20]])
assert np.shape(BS_LOCATIONS) == (N_BS, 3)
BANDWIDTH = 5e6
CARRIER_FREQUENCY = 50e9 
WAVELENGTH = 2.998e8/CARRIER_FREQUENCY
_NOISE_dBm_Hz = -150
NOISE_POWER = np.power(10, ((_NOISE_dBm_Hz-30)/10)) * BANDWIDTH
RICIAN_FACTOR = 10
# design choice: number of factors to be reconstructed
N_FACTORS = 3   # distance to BS, sin_theta_cos_phi, cos_theta_cos_phi

# ensure the number of antennas is a perfect square, for 2D array
assert np.floor(np.sqrt(N_BS_ANTENNAS))**2 == N_BS_ANTENNAS

"""
Training Setting
"""
SOURCETASK = {'Type': 'Source-Task',
        'Task': 'Beamforming',
        'Train': int(5e5),
        'Valid': 5000,
        'Minibatch_Size': 2000,
        'Learning_Rate': 1e-3,
        'Epochs': 200,
        'Loss_Combine_Weight': 3}
TARGETTASK = {'Type': 'Target-Task',
        'Task': 'Localization',
        'Train': 1000,
        'Valid': 5000,
        'Minibatch_Size': 100,
        'Learning_Rate': 2e-5,
        'Epochs': 25000}
N_TEST_SAMPLES = 2000
assert SOURCETASK['Task'] in ['Localization', 'Beamforming'] and \
       TARGETTASK['Task'] in ['Localization', 'Beamforming']
assert SOURCETASK['Train'] % SOURCETASK['Minibatch_Size'] == 0 and \
       TARGETTASK['Train'] % TARGETTASK['Minibatch_Size'] == 0

"""
Reproduciability
"""
RANDOM_SEED = 123
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)