# This script contains the generator code for producing wireless network channels and sum-rate/min-rate targets

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from utils import *
from setup import *

CHECK_SETTING = False
GENERATE_CHANNELS_SOURCETASK = True
GENERATE_CHANNELS_TARGETTASK = True
GENERATE_CHANNELS_TEST = True

if __name__ == '__main__':
    if CHECK_SETTING:
        # Generate layouts for FP and save as testing data for GP
        g, _ = generate_D2D_channelGains(100)
        # plot some layouts
        for i in range(5):
            plt.plot(g[i].flatten())
            plot_label_direct_channels(plt.gca())
            plt.show()
        fp = FP_power_control(g)
        sumrates_fp = np.sum(compute_rates(compute_SINRs(fp, g)),axis=1)
        print("FP sum rates: {:.3f} Mbps".format(np.mean(sumrates_fp)/1e6))
        sumrates_allactive = np.sum(compute_rates(compute_SINRs(np.ones_like(fp), g)),axis=1)
        print("Full Power sum rates: {:.3f}% of FP".format(np.mean(sumrates_allactive/sumrates_fp)*100))
        np.save("Data/g_test_{}.npy".format(SETTING_STRING), g)
        savemat("Data/g_test_{}.mat".format(SETTING_STRING), {'g': g})
        input("Invoke matlab script and compute GP solutions. Press any key once finished")
        GP_power_control()
        exit(0)

    if GENERATE_CHANNELS_SOURCETASK:
        print(f"Generate wireless channels for {SOURCETASK['Type']} {SOURCETASK['Task']} training, including statistics for input normalization......")
        g, _ = generate_D2D_channelGains(SOURCETASK['Train']+SOURCETASK['Valid'])
        np.save("Data/g_sourceTask_{}.npy".format(SETTING_STRING), g)
        # Use the sum rate train data for input normalization stats
        np.save("Trained_Models/Input_Normalization_Stats/g_train_mean_{}.npy".format(SETTING_STRING), np.mean(g[:SOURCETASK['Train']], axis=0))
        np.save("Trained_Models/Input_Normalization_Stats/g_train_std_{}.npy".format(SETTING_STRING), np.std(g[:SOURCETASK['Train']], axis=0))

    if GENERATE_CHANNELS_TARGETTASK:
        print("Generate wireless channels for target task training......")
        g, _ = generate_D2D_channelGains(TARGETTASK['Train']+TARGETTASK['Valid'])
        np.save("Data/g_targetTask_{}.npy".format(SETTING_STRING), g)

    if GENERATE_CHANNELS_TEST:
        print("Generate wireless channels for testing......")
        g, _ = generate_D2D_channelGains(N_TEST_SAMPLES)
        np.save("Data/g_test_{}.npy".format(SETTING_STRING), g)
        print("Save test channels to matlab file for GP on Min-Rate optimization...")
        savemat("Data/g_test_{}.mat".format(SETTING_STRING), {'g': g})

    print("Script Completed!")
