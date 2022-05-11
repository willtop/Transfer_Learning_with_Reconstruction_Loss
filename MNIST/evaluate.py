# Script for evaluating D2D network objectives: sum rate or min rate

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
from neural_nets import Regular_Net, Transfer_Net, Autoencoder_Transfer_Net
from setup import *
import utils

VISUALIZE_SOURCETASK = False
VISUALIZE_TARGETTASK = True

if(__name__ =='__main__'):
    print(f"[D2D] Evaluate {TASK_DESCR} over {N_TEST_SAMPLES} test data.")

    test_data = MNIST(root='Data/', train=False, download=True, 
            transform=transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize(size=(IMAGE_LENGTH, IMAGE_LENGTH)),
                                          transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                                          transforms.Lambda(lambda x: x.flatten())]))
    assert len(test_data) == N_TEST_SAMPLES
    data_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    regular_net, transfer_net, ae_transfer_net = Regular_Net().to(DEVICE), Transfer_Net().to(DEVICE), Autoencoder_Transfer_Net().to(DEVICE)
    for task in [SOURCETASK, TARGETTASK]:
        print(f"Evaluating {task['Type']}: identifying {task['Task']}...")
        for data, targets in data_loader: # only one batch
            assert data.size() == (N_TEST_SAMPLES, INPUT_SIZE) and \
                   targets.size() == (N_TEST_SAMPLES, )
            targets = utils.convert_targets(targets, task)                    
            accuracies = {}
            if "Source" in task['Type']:
                predictions = regular_net.sourcetask(data.to(DEVICE))
                accuracies["Regular Learning"] = utils.compute_accuracy(predictions.detach().cpu(), targets)
                predictions = transfer_net.sourcetask(data.to(DEVICE))
                accuracies["Conventional Transfer Learning"] = utils.compute_accuracy(predictions.detach().cpu(), targets)
                predictions, _ = ae_transfer_net.sourcetask(data.to(DEVICE))
                accuracies["Transfer Learning with Reconstruction"] = utils.compute_accuracy(predictions.detach().cpu(), targets)
            else:
                predictions = regular_net.targettask(data.to(DEVICE))
                accuracies["Regular Learning"] = utils.compute_accuracy(predictions.detach().cpu(), targets)
                predictions = transfer_net.targettask(data.to(DEVICE))
                accuracies["Conventional Transfer Learning"] = utils.compute_accuracy(predictions.detach().cpu(), targets)
                predictions = ae_transfer_net.targettask(data.to(DEVICE))
                accuracies["Transfer Learning with Reconstruction"] = utils.compute_accuracy(predictions.detach().cpu(), targets)
            
        print(f"{TASK_DESCR} {task['Type']} Accuracies on identifying {task['Task']}: ")
        for method_key, accuracy in accuracies.items():
            print("[{}]: {:.2f}%;".format(method_key, accuracy*100), end="")
        print("\n")
        
    print("Evaluation Finished Successfully!")
