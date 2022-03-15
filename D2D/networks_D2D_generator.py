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
    np.save("Data/fp_{}.npy".format(SETTING_STRING), fp)

    # Generate and save for min rate
    g, _ = generate_D2D_channelGains(N_SAMPLES['MinRate']['Train']+N_SAMPLES['MinRate']['Valid'])
    dink = 
    np.save("Data/g_minRate_{}.npy".format(SETTING_STRING), g)
    savemat("Data/h_est_test_{}.mat".format(SETTING_STRING), {'h_est': csi_dl_pl_cl[-N_TEST:]})

    # Save input normalization stats
    np.save("Trained_Models_D2D/h_est_train_mean_{}.npy".format(SETTING_STRING), np.mean(csi_dl_pl_cl[:N_TRAIN], axis=0))
    np.save("Trained_Models_D2D/h_est_train_std_{}.npy".format(SETTING_STRING), np.std(csi_dl_pl_cl[:N_TRAIN], axis=0))

    print("Script Completed!")
