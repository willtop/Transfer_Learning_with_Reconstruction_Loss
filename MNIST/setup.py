import os
import random
import numpy as np
import torch
# For windows specific error
os.environ['KMP_DUPLICATE_LIB_OK']='True'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE)

APPLICATION = 'MNIST'

SourceTask_MNIST = {'Type': 'Source-Task',
  'Task': 1,
  'Fullname': 'MNIST Identify 1',
  'Train': 56000,
  'Valid': 4000,
  'Minibatch_Size': 100,
  'Epochs': 30,
  'Learning_Rate': 1e-3,
  'Loss_Combine_Weight': 1}
TargetTask_MNIST = {'Type': 'Target_Task',
  'Task': 5,
  'Fullname': 'MNIST Identify 5',
  'Train': 1000,
  'Valid': 4000,
  'Minibatch_Size': 100,
  'Epochs': 100,
  'Learning_Rate': 1e-4}

if APPLICATION == 'MNIST':
    SOURCETASK = SourceTask_MNIST
    TARGETTASK = TargetTask_MNIST
else:
    print(f"Invalid application {APPLICATION}!")
    exit(1)
assert SOURCETASK['Train'] % SOURCETASK['Minibatch_Size'] == 0 and \
       TARGETTASK['Train'] % TARGETTASK['Minibatch_Size'] == 0

# set random seed
RANDOM_SEED = 123
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
