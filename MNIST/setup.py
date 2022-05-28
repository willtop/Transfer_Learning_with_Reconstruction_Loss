import os
import random
import numpy as np
import torch
# For windows specific error
os.environ['KMP_DUPLICATE_LIB_OK']='True'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE)

APPLICATION = 'MNIST'
# APPLICATION = 'FashionMNIST'
assert APPLICATION in ['MNIST', 'FashionMNIST'], f"Invalid application: {APPLICATION}"

# MNIST
SourceTask_MNIST = {'Type': 'Source_Task',
  'Task': 1,
  'Minibatch_Size': 500,
  'Epochs': 100,
  'Learning_Rate': 1e-3,
  'Loss_Combine_Weight': 5}
TargetTask_MNIST = {'Type': 'Target_Task',
  'Task': 5,
  'Minibatch_Size': 32,
  'Epochs': 8000,
  'Learning_Rate': 2e-4}

# Fashion-MNIST
SourceTask_FashionMNIST = {'Type': 'Source_Task',
  'Task': 0, 
  'Minibatch_Size': 500,
  'Epochs': 100,
  'Learning_Rate': 1e-3,
  'Loss_Combine_Weight': 3}
TargetTask_FashionMNIST = {'Type': 'Target_Task',
  'Task': 5, 
  'Minibatch_Size': 50,
  'Epochs': 7000,
  'Learning_Rate': 1e-3}

if APPLICATION == 'MNIST':
    SOURCETASK = SourceTask_MNIST
    TARGETTASK = TargetTask_MNIST
    CLASSES = [1,3,5]
    TASK_DESCR = f"MNIST_{SOURCETASK['Task']}-{TARGETTASK['Task']}_with{3}"
    IMAGE_LENGTH = 10
    INPUT_SIZE = IMAGE_LENGTH**2
elif APPLICATION == 'FashionMNIST':
    SOURCETASK = SourceTask_FashionMNIST
    TARGETTASK = TargetTask_FashionMNIST
    CLASSES = [0,5,7]
    TASK_DESCR = f"FashionMNIST_{SOURCETASK['Task']}-{TARGETTASK['Task']}"
    IMAGE_LENGTH = 10
    INPUT_SIZE = IMAGE_LENGTH**2
else:
    print(f"Invalid application {APPLICATION}!")
    exit(1)
assert SOURCETASK['Task'] in CLASSES and TARGETTASK['Task'] in CLASSES

# set random seed
RANDOM_SEED = 123
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
