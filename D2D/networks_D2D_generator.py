# This script contains the generator code for producing wireless network channels and sum-rate/min-rate targets

import numpy as np
import matplotlib.pyplot as plt
from benchmarks import *
from scipy.io import savemat
from utils import *
from setup import *

GENERATE_SUMRATE = True
GENERATE_MINRATE = True

if __name__ == '__main__':
    print(f"Generating channels for: sum rate: {GENERATE_SUMRATE}; min rate: {GENERATE_MINRATE}")
    if GENERATE_SUMRATE:
        print("Generate wireless channels for sum rate training, including statistics for input normalization")
        g, _ = generate_D2D_channelGains(N_SAMPLES['SumRate']['Train']+N_SAMPLES['SumRate']['Valid'])
        fp = FP_power_control(g)
        np.save("Data/g_sumRate_{}.npy".format(SETTING_STRING), g)
        # Use the sum rate train data for input normalization stats
        np.save("Trained_Models/g_train_mean_{}.npy".format(SETTING_STRING), np.mean(g[:N_SAMPLES['SumRate']['Train']], axis=0))
        np.save("Trained_Models/g_train_std_{}.npy".format(SETTING_STRING), np.std(g[:N_SAMPLES['SumRate']['Train']], axis=0))

    if GENERATE_MINRATE:
        print("Generate wireless channels for min rate training, as well as testing samples")
        g, _ = generate_D2D_channelGains(N_SAMPLES['MinRate']['Train']+N_SAMPLES['MinRate']['Valid']+N_SAMPLES['MinRate']['Test'])
        np.save("Data/g_minRate_{}.npy".format(SETTING_STRING), g)
        print("Save wireless channels for matlab script to compute min-rate GP solutions")
        savemat("Data/g_minRate_{}.mat".format(SETTING_STRING), {'g': g})

    print("Script Completed!")