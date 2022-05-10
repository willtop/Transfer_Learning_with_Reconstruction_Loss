import os
import random
import numpy as np
import torch
# For windows specific error
os.environ['KMP_DUPLICATE_LIB_OK']='True'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE)

APPLICATION = 'MNIST'

SourceTask_MNIST = {'Type': 'Source_Task',
  'Task': 1,
  'Fullname': 'MNIST Identify 1',
  'Train': 50000,
  'Valid': 10000,
  'Minibatch_Size': 500,
  'Epochs': 10,
  'Learning_Rate': 1e-4,
  'Loss_Combine_Weight': 1}
TargetTask_MNIST = {'Type': 'Target_Task',
  'Task': 5,
  'Fullname': 'MNIST Identify 5',
  'Train': 100,
  'Valid': 10000,
  'Minibatch_Size': 50,
  'Epochs': 500,
  'Learning_Rate': 1e-4}
N_TEST_SAMPLES_MNIST = 10000

if APPLICATION == 'MNIST':
    SOURCETASK = SourceTask_MNIST
    TARGETTASK = TargetTask_MNIST
    TASK_DESCR = f"MNIST_{SOURCETASK['Task']}-{TARGETTASK['Task']}"
    INPUT_SIZE = 28*28
    N_TEST_SAMPLES = N_TEST_SAMPLES_MNIST
else:
    print(f"Invalid application {APPLICATION}!")
    exit(1)
assert SOURCETASK['Train'] % SOURCETASK['Minibatch_Size'] == 0 and \
       TARGETTASK['Train'] % TARGETTASK['Minibatch_Size'] == 0
assert (TARGETTASK['Train']+TARGETTASK['Valid'])<(SOURCETASK['Train']+SOURCETASK['Valid'])

# set random seed
RANDOM_SEED = 123
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
