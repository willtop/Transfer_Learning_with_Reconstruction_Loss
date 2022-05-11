import os
import random
import numpy as np
import torch
# For windows specific error
os.environ['KMP_DUPLICATE_LIB_OK']='True'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE)

# APPLICATION = 'MNIST'
APPLICATION = 'FashionMNIST'
assert APPLICATION in ['MNIST', 'FashionMNIST'], f"Invalid application: {APPLICATION}"

# MNIST
SourceTask_MNIST = {'Type': 'Source_Task',
  'Task': [0, 4, 6],
  'Output_Dim': 4, # Numbers to be identified plus the rest
  'Train': 49950,
  'Valid': 5000,
  'Minibatch_Size': 500,
  'Epochs': 150,
  'Learning_Rate': 1e-3,
  'Loss_Combine_Weight': 3}
TargetTask_MNIST = {'Type': 'Target_Task',
  'Task': [5, 8, 9],
  'Output_Dim': 4, # Numbers to be identified plus the rest
  'Train': 50,
  'Valid': 5000,
  'Minibatch_Size': 50,
  'Epochs': 2000,
  'Learning_Rate': 2e-4}

# Fashion-MNIST
SourceTask_FashionMNIST = {'Type': 'Source_Task',
  'Task': [0, 3, 4, 5], 
  'Output_Dim': 5, # Classes to be identified plus the rest
  'Train': 48500,
  'Valid': 5000,
  'Minibatch_Size': 500,
  'Epochs': 100,
  'Learning_Rate': 1e-3,
  'Loss_Combine_Weight': 3}
TargetTask_FashionMNIST = {'Type': 'Target_Task',
  'Task': [2, 6, 7, 9], 
  'Output_Dim': 5, # Classes to be identified plus the rest
  'Train': 1500,
  'Valid': 5000,
  'Minibatch_Size': 50,
  'Epochs': 2000,
  'Learning_Rate': 5e-4}

if APPLICATION == 'MNIST':
    SOURCETASK = SourceTask_MNIST
    TARGETTASK = TargetTask_MNIST
    TASK_DESCR = f"MNIST_{SOURCETASK['Task']}-{TARGETTASK['Task']}"
    IMAGE_LENGTH = 14
    INPUT_SIZE = IMAGE_LENGTH**2
    N_TEST_SAMPLES = 10000
    assert SOURCETASK['Train']+SOURCETASK['Valid']+TARGETTASK['Train']+TARGETTASK['Valid'] == 60000
elif APPLICATION == 'FashionMNIST':
    SOURCETASK = SourceTask_FashionMNIST
    TARGETTASK = TargetTask_FashionMNIST
    TASK_DESCR = f"FashionMNIST_{SOURCETASK['Task']}-{TARGETTASK['Task']}"
    IMAGE_LENGTH = 14
    INPUT_SIZE = IMAGE_LENGTH**2
    N_TEST_SAMPLES = 10000
    assert SOURCETASK['Train']+SOURCETASK['Valid']+TARGETTASK['Train']+TARGETTASK['Valid'] == 60000
else:
    print(f"Invalid application {APPLICATION}!")
    exit(1)
assert (TARGETTASK['Train']+TARGETTASK['Valid'])<(SOURCETASK['Train']+SOURCETASK['Valid'])

# set random seed
RANDOM_SEED = 123
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
