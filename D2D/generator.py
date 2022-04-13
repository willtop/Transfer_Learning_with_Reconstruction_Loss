# This script contains the generator code for producing wireless network channels and sum-rate/min-rate targets

import numpy as np
from scipy.io import savemat
from utils import *
from setup import *

FP_CHECK = False
GENERATE_CHANNELS_SOURCETASK = True
GENERATE_CHANNELS_TARGETTASK = True
GENERATE_CHANNELS_TEST = True

if __name__ == '__main__':
    if FP_CHECK:
        # Only generate a few layouts and see FP allocation average
        g, _ = generate_D2D_channelGains(1000)
        fp = FP_power_control(g)
        print("[{}] Avg FP allocation power on sum rate: {}%".format(SETTING_STRING, np.mean(fp)*100))
        exit(0)

    if GENERATE_CHANNELS_SOURCETASK:
        print("Generate wireless channels for source weights sum rate training, including statistics for input normalization......")
        g, _ = generate_D2D_channelGains(N_SAMPLES['SourceTask']['Train']+N_SAMPLES['SourceTask']['Valid'])
        np.save("Data/g_sourceTask_{}.npy".format(SETTING_STRING), g)
        # Use the sum rate train data for input normalization stats
        np.save("Trained_Models/Input_Normalization_Stats/g_train_mean_{}.npy".format(SETTING_STRING), np.mean(g[:N_SAMPLES['SourceTask']['Train']], axis=0))
        np.save("Trained_Models/Input_Normalization_Stats/g_train_std_{}.npy".format(SETTING_STRING), np.std(g[:N_SAMPLES['SourceTask']['Train']], axis=0))

    if GENERATE_CHANNELS_TARGETTASK:
        print("Generate wireless channels for target weights sum rate training......")
        g, _ = generate_D2D_channelGains(N_SAMPLES['TargetTask']['Train']+N_SAMPLES['TargetTask']['Valid'])
        np.save("Data/g_targetTask_{}.npy".format(SETTING_STRING), g)

    if GENERATE_CHANNELS_TEST:
        print("Generate wireless channels for testing......")
        g, _ = generate_D2D_channelGains(N_SAMPLES['Test'])
        np.save("Data/g_test_{}.npy".format(SETTING_STRING), g)
        print("Save test channels to matlab file for GP on min-rate optimization...")
        savemat("Data/g_test_{}.mat".format(SETTING_STRING), {'g': g})

    print("Script Completed!")
