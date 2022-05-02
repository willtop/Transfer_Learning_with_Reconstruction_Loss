# This script contains the generator code for producing wireless network channels and sum-rate/min-rate targets

import numpy as np
from scipy.io import savemat
from utils import *
from setup import *

CHECK_SETTING = True
GENERATE_CHANNELS_SOURCETASK = False
GENERATE_CHANNELS_TARGETTASK = False
GENERATE_CHANNELS_TEST = True

if __name__ == '__main__':
    if CHECK_SETTING:
        # Generate layouts for FP and save as testing data for GP
        g, _ = generate_D2D_channelGains(100)
        fp = FP_power_control(g)
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
        print("Save test channels to matlab file for GP on min-rate optimization...")
        savemat("Data/g_test_{}.mat".format(SETTING_STRING), {'g': g})

    print("Script Completed!")
