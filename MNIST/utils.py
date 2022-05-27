import numpy as np
import torch
from setup import *
from torch.utils.data.dataset import Subset

def get_subclass_data(original_data):
    idx = [target in CLASSES for target in original_data.targets]
    original_data.data = original_data.data[idx]
    original_data.targets = original_data.targets[idx]
    return original_data

# currently quite slow implementation for Subset class as it doesn't have .targets attribute
def get_class_distribution(data_to_show, desc_str):
    print(f"Class distribution for {desc_str}")
    for i in CLASSES:
        if not isinstance(data_to_show, Subset):
            n_samples = (data_to_show.targets == i).sum().item()
        else:
            n_samples = 0
            for _, j in data_to_show:
                if j==i:
                    n_samples +=1
        print(f"Class {i} has {n_samples}/{len(data_to_show)} samples")
    return

def convert_targets(targets, task_obj):
    return torch.tensor(targets==task_obj['Task'], dtype=torch.float32)
    
# require targets to be already-converted one-hot encoding
def compute_accuracy(predictions, targets):
    assert targets.ndim == 1
    assert predictions.size() == targets.size()
    return np.mean(predictions.round().numpy() == targets.numpy())
