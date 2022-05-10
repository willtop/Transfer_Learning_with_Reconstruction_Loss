# Script for evaluating D2D network objectives: sum rate or min rate

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from neural_nets import Regular_Net, Transfer_Net, Autoencoder_Transfer_Net
from setup import *
from utils import *

VISUALIZE_SOURCETASK = False
VISUALIZE_TARGETTASK = True

if(__name__ =='__main__'):
    print(f"[D2D] Evaluate {TASK_DESCR} over {N_TEST_SAMPLES} test data.")

    regular_net, transfer_net, ae_transfer_net = Regular_Net().to(DEVICE), Transfer_Net().to(DEVICE), Autoencoder_Transfer_Net().to(DEVICE)
    for task in [SOURCETASK, TARGETTASK]:
        print(f"Evaluating {task['Type']}: identifying {task['Task']}...")
        datatype = "SourceTaskTest" if task['Type']=="Source_Task" else "TargetTaskTest"
        source_data = load_source_data(datatype)
        assert len(source_data) == N_TEST_SAMPLES
        data_loader = DataLoader(source_data, batch_size=len(source_data), shuffle=False)
        for data, targets in data_loader: # only one batch
            accuracies = {}
            if task['Type'] == "Source-Task":
                predictions = regular_net.sourcetask(data.to(DEVICE))
                accuracies["Regular Learning"] = np.mean(predictions.round().detach().cpu().numpy()==targets.numpy())
                predictions = transfer_net.sourcetask(data.to(DEVICE))
                accuracies["Conventional Transfer Learning"] = np.mean(predictions.round().detach().cpu().numpy()==targets.numpy())
                predictions = ae_transfer_net.sourcetask(data.to(DEVICE))
                accuracies["Transfer Learning with Reconstruction"] = np.mean(predictions.round().detach().cpu().numpy()==targets.numpy())
            else:
                predictions = regular_net.targettask(data.to(DEVICE))
                accuracies["Regular Learning"] = np.mean(predictions.round().detach().cpu().numpy()==targets.numpy())
                predictions = transfer_net.targettask(data.to(DEVICE))
                accuracies["Conventional Transfer Learning"] = np.mean(predictions.round().detach().cpu().numpy()==targets.numpy())
                predictions = ae_transfer_net.targettask(data.to(DEVICE))
                accuracies["Transfer Learning with Reconstruction"] = np.mean(predictions.round().detach().cpu().numpy()==targets.numpy())
            
        print(f"{TASK_DESCR} {task['Type']} Accuracies on identifying {task['Task']}: ")
        for method_key, accuracy in accuracies.items():
            print("[{}]: {:.2f}%;".format(method_key, accuracy*100), end="")
        print("\n")
        
    print("Evaluation Finished Successfully!")
