# This script contains the generator code for producing wireless network channels and sum-rate/min-rate targets

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from setup import *

FP_CHECK = False
GENERATE_WEIGHTS = False
GENERATE_CHANNELS_SOURCETASK = False
GENERATE_CHANNELS_TARGETTASK = True
GENERATE_CHANNELS_TEST = False

if __name__ == '__main__':
    if FP_CHECK:
        weights1 = np.load("Trained_Models/Importance_Weights/sourceTask_weights.npy")
        weights2 = np.load("Trained_Models/Importance_Weights/targetTask_weights.npy")
        visualize_importance_weights(weights1, weights2)
        # Only generate a few layouts and see FP allocation average
        g, _ = generate_D2D_channelGains(1000)
        fp1 = FP_power_control(g, weights1)
        fp2 = FP_power_control(g, weights2)
        print("Avg FP allocation power [{}]: source task: {:.3f}%; target task: {:.3f}%".format(SETTING_STRING, np.mean(fp1)*100, np.mean(fp2)*100))
        exit(0)

    print(f"[Generate] weights: {GENERATE_WEIGHTS}; source task channels: {GENERATE_CHANNELS_SOURCETASK}; target task channels: {GENERATE_CHANNELS_TARGETTASK}; test channels: {GENERATE_CHANNELS_TEST}")
    if GENERATE_WEIGHTS:
        print(f"Generate two sets of weights, each with {N_LINKS} components...")
        weights1 = np.random.sample(size=N_LINKS) 
        weights1 = weights1/np.sum(weights1)
        weights2 = np.random.sample(size=N_LINKS) 
        weights2 = weights2/np.sum(weights2)
        np.save("Trained_Models/Importance_Weights/sourceTask_weights.npy", weights1)
        np.save("Trained_Models/Importance_Weights/targetTask_weights.npy", weights2) 

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

    print("Script Completed!")
