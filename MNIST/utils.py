from setup import *
from torchvision.datasets import MNIST
from torchvision import transforms

def load_source_data(datatype):
    whether_train = "Train" in datatype
    task = SOURCETASK['Task'] if "Source" in datatype else TARGETTASK['Task']
    source_data = MNIST(root=f'{TASK_DESCR}_Data_{datatype}/', train=whether_train, download=True, 
            transform=transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                                            transforms.Lambda(lambda x: x.flatten())]), 
            target_transform=transforms.Lambda(lambda y: torch.tensor(y==task, dtype=torch.float32)))
    return source_data