# script to visualize downsampling results on both MNIST and FashionMNIST

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

side_lengths = [28, 20, 14, 10, 7]
if __name__=="__main__":
    mnist_data = {}
    for side_length in side_lengths:
        if side_length == 28:
            mnist_data[side_length] = MNIST(root="Data_Checks/", train=True, download=True,
                transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=(0.1307,),std=(0.3081,))]))
        else:
            mnist_data[side_length] = MNIST(root="Data_Checks/", train=True, download=True,
                transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=(0.1307,),std=(0.3081,)),
                                              transforms.Resize(size=(side_length, side_length))]))
    # Find index of images on class of interests
    classes = [0,1,8]
    idxes = {}
    for cls in classes:
        idxes_tmp = (mnist_data[side_lengths[0]].targets == cls).int()
        idxes[cls] = torch.nonzero(idxes_tmp, as_tuple=True)[0]
    for k in range(5):
        fig, axes = plt.subplots(len(classes), len(side_lengths))
        for i, cls in enumerate(classes):
            for j, side_length in enumerate(side_lengths):
                idx = idxes[cls][k].item()
                img, img_cls = mnist_data[side_length][idx]
                assert img_cls == cls
                ax = axes[i][j]
                ax.set_title(f"{side_length}X{side_length} figure on Class {cls}")
                assert img.size() == (1, side_length, side_length), "Wrong size at side length {}: {}".format(side_length, img.size())
                img = np.squeeze(np.array(img))
                ax.imshow(img)
        plt.show()
    print("Script completed!")