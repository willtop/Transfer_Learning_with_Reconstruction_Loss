# script to visualize downsampling results on both MNIST and FashionMNIST

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

side_lengths = [28, 20, 14]
if __name__=="__main__":
    mnist_data = {}
    for side_length in side_lengths:
        if side_length == 28:
            mnist_data[side_length] = FashionMNIST(root="Data_Checks/", train=True, download=True,
                transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=(0.1307,),std=(0.3081,))]))
        else:
            mnist_data[side_length] = FashionMNIST(root="Data_Checks/", train=True, download=True,
                transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=(0.1307,),std=(0.3081,)),
                                              transforms.Resize(size=(side_length, side_length))]))
    for idx in range(5):
        fig, axes = plt.subplots(1, len(side_lengths))
        for i, side_length in enumerate(side_lengths):
            ax = axes[i]
            ax.set_title(f"{side_length}X{side_length} figure")
            img, _ = mnist_data[side_length][idx]
            assert img.size() == (1, side_length, side_length), "Wrong size at side length {}: {}".format(side_length, img.size())
            img = np.squeeze(np.array(img))
            ax.imshow(img)
        plt.show()
    print("Script completed!")