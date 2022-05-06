import os
import random
import numpy as np
import torch
# For windows specific error
os.environ['KMP_DUPLICATE_LIB_OK']='True'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE)

SOURCETASK = {'Number': 1,
  'Train': 56000,
  'Valid': 4000,
  'Learning_Rate': 1e-3}

TARGETTASK = {'Number': 5,
  'Train': 5000,
  'Valid': 4000,
  'Learning_Rate': 1e-4}

# set random seed
RANDOM_SEED = 123
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
