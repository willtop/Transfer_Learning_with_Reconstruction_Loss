# This script contains the generator code for producing wireless network channels and sum-rate/min-rate targets

import numpy as np
import matplotlib.pyplot as plt
from benchmarks import *
from scipy.io import savemat
from utils import *
from setup import *


if __name__ == '__main__':
    # Generate and save for sum rate
    g, _ = generate_D2D_channelGains(N_SAMPLES['SumRate']['Train']+N_SAMPLES['SumRate']['Valid'])
    fp = FP_power_control(g)
    np.save("Data/g_sumRate_{}.npy".format(SETTING_STRING), g)
    # Use the sum rate train data for input normalization stats
    np.save("Trained_Models/g_train_mean_{}.npy".format(SETTING_STRING), np.mean(g[:N_SAMPLES['SumRate']['Train']], axis=0))
    np.save("Trained_Models/g_train_std_{}.npy".format(SETTING_STRING), np.std(g[:N_SAMPLES['SumRate']['Train']], axis=0))

    # Generate and save for min rate
    g, _ = generate_D2D_channelGains(N_SAMPLES['MinRate']['Train']+N_SAMPLES['MinRate']['Valid']+N_SAMPLES['MinRate']['Test'])
    np.save("Data/g_minRate_{}.npy".format(SETTING_STRING), g)
    # save for matlab script to compute GP solutions
    savemat("Data/g_minRate_{}.mat".format(SETTING_STRING), {'g': g})

    print("Script Completed!")
