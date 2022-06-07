import numpy as np
import torch

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
BS_LOCATIONS = np.array([[0,0,0], [0,40,20], [40,0,20]])
assert np.shape(BS_LOCATIONS) == (N_BS, 3)
BANDWIDTH = 5e6
_NOISE_dBm_Hz = -150
NOISE_POWER = np.power(10, ((_NOISE_dBm_Hz-30)/10)) * BANDWIDTH
RICIAN_FACTOR = 10

# ensure the number of antennas is a perfect square, for 2D array
assert np.floor(np.sqrt(N_BS_ANTENNAS))**2 == N_BS_ANTENNAS

"""
Training Setting
"""
