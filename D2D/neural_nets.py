import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from setup import *
import os

class Neural_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # model architecture attribute
        self.feature_length = int(N_LINKS * N_LINKS /2)
        # attributes to be overridden by subclasses
        self.model_type = None
        self.model_path = None
        # general attributes
        self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Trained_Models")
        self.input_mean = torch.tensor(np.load(os.path.join(self.base_dir, "Input_Normalization_Stats", "g_train_mean_{}.npy".format(SETTING_STRING))),dtype=torch.float32).to(DEVICE)
        self.input_std = torch.tensor(np.load(os.path.join(self.base_dir, "Input_Normalization_Stats", "g_train_std_{}.npy".format(SETTING_STRING))),dtype=torch.float32).to(DEVICE)

    def preprocess_input(self, g):
        assert g.ndim == 3
        g = g.view(-1, N_LINKS*N_LINKS)
        x = (g-self.input_mean.view(1, N_LINKS*N_LINKS))/self.input_std.view(1, N_LINKS*N_LINKS)
        return x

    def sumRate_power_control(self):
        raise NotImplementedError

    def minRate_power_control(self):
        raise NotImplementedError

    def load_model(self):
        if os.path.exists(self.model_path):
            self.load_state_dict(torch.load(self.model_path))
            print("[{}] Load trained model from: {}".format(self.model_type, self.model_path))
        else:
            print("[{}] Train from scratch.".format(self.model_type))

    def save_model(self):
        torch.save(self.state_dict(), self.model_path)
        print("[{}] Model saved at {}".format(self.model_type, self.model_path))
        return

    # Modules to compose different types of neural net
    def construct_feature_module(self):
        feature_module = nn.ModuleList()
        feature_module.append(nn.Linear(N_LINKS*N_LINKS, 3*N_LINKS*N_LINKS))
        feature_module.append(nn.ReLU())
        feature_module.append(nn.Linear(3*N_LINKS*N_LINKS, 3*N_LINKS*N_LINKS))
        feature_module.append(nn.ReLU())
        feature_module.append(nn.Linear(3*N_LINKS*N_LINKS, 3*N_LINKS*N_LINKS))
        feature_module.append(nn.ReLU())
        feature_module.append(nn.Linear(3*N_LINKS*N_LINKS, self.feature_length))
        feature_module.append(nn.ReLU())
        return feature_module
    
    def construct_optimizer_module(self):
        optimizer_module = nn.ModuleList()
        optimizer_module.append(nn.Linear(self.feature_length, N_LINKS*N_LINKS))
        optimizer_module.append(nn.ReLU())
        optimizer_module.append(nn.Linear(N_LINKS*N_LINKS, N_LINKS*N_LINKS))
        optimizer_module.append(nn.ReLU())
        optimizer_module.append(nn.Linear(N_LINKS*N_LINKS, 4*N_LINKS))
        optimizer_module.append(nn.ReLU())
        optimizer_module.append(nn.Linear(4*N_LINKS, N_LINKS))
        # with power control output being 0~1
        optimizer_module.append(nn.Sigmoid())
        return optimizer_module

class Regular_Net(Neural_Net):
    def __init__(self):
        super().__init__()
        self.model_type = "Regular"
        self.model_path = os.path.join(self.base_dir, "{}_{}.ckpt".format(self.model_type, SETTING_STRING))
        self.sumRate_feature_module = self.construct_feature_module()
        self.sumRate_optimizer_module = self.construct_optimizer_module()
        self.minRate_feature_module = self.construct_feature_module()
        self.minRate_optimizer_module = self.construct_optimizer_module()
        self.load_model()

    def sumRate_power_control(self, g):
        x = self.preprocess_input(g)
        for lyr in self.sumRate_feature_module:
            x = lyr(x)
        for lyr  in self.sumRate_optimizer_module:
            x = lyr(x)
        return x

    def minRate_power_control(self, g):
        x = self.preprocess_input(g)
        for lyr in self.minRate_feature_module:
            x = lyr(x)
        for lyr  in self.minRate_optimizer_module:
            x = lyr(x)
        return x


class Transfer_Net(Neural_Net):
    def __init__(self):
        super().__init__()
        self.model_type = "Transfer"
        self.model_path = os.path.join(self.base_dir, "{}_{}.ckpt".format(self.model_type, SETTING_STRING))
        self.feature_module = self.construct_feature_module()
        self.sumRate_optimizer_module = self.construct_optimizer_module()
        self.minRate_optimizer_module = self.construct_optimizer_module()
        self.load_model()

    def sumRate_power_control(self, g):
        x = self.preprocess_input(g)
        for lyr in self.feature_module:
            x = lyr(x)
        for lyr  in self.sumRate_optimizer_module:
            x = lyr(x)
        return x

    # freeze parameters for transfer learning
    def freeze_parameters(self):
        for lyr in self.feature_module:
            for para in lyr.parameters():
                para.requires_grad = False
        return

    def minRate_power_control(self, g):
        x = self.preprocess_input(g)
        for lyr in self.feature_module:
            x = lyr(x)
        for lyr  in self.minRate_optimizer_module:
            x = lyr(x)
        return x

class Autoencoder_Transfer_Net(Neural_Net):
    def __init__(self):
        super().__init__()
        self.model_type = "Autoencoder_Transfer"
        self.model_path = os.path.join(self.base_dir, "{}_{}.ckpt".format(self.model_type, SETTING_STRING))
        self.feature_module = self.construct_feature_module()
        self.decoder_module = self.construct_decoder_module()
        self.sumRate_optimizer_module = self.construct_optimizer_module()
        self.minRate_optimizer_module = self.construct_optimizer_module()
        self.load_model()

    def construct_decoder_module(self):
        decoder_module = nn.ModuleList()
        decoder_module.append(nn.Linear(self.feature_length, 3*N_LINKS*N_LINKS))
        decoder_module.append(nn.ReLU())
        decoder_module.append(nn.Linear(3*N_LINKS*N_LINKS, 3*N_LINKS*N_LINKS))
        decoder_module.append(nn.ReLU())
        decoder_module.append(nn.Linear(3*N_LINKS*N_LINKS, N_LINKS*N_LINKS))
        return decoder_module

    def sumRate_power_control(self, g):
        x = self.preprocess_input(g)
        inputs = torch.clone(x)
        for lyr in self.feature_module:
            x = lyr(x)
        features = torch.clone(x)
        # try to reconstruct inputs
        for lyr in self.decoder_module:
            x = lyr(x)
        inputs_reconstructed = x
        for lyr in self.sumRate_optimizer_module:
            features = lyr(features)
        outputs = features
        return outputs, inputs, inputs_reconstructed
    
    # freeze parameters for transfer learning
    def freeze_parameters(self):
        for lyr in self.feature_module:
            for para in lyr.parameters():
                para.requires_grad = False
        return

    def minRate_power_control(self, g):
        x = self.preprocess_input(g)
        for lyr in self.feature_module:
            x = lyr(x)
        for lyr in self.minRate_optimizer_module:
            x = lyr(x)
        return x

    # freeze parameters for transfer learning
    def freeze_parameters(self):
        for lyr in self.feature_module:
            for para in lyr.parameters():
                para.requires_grad = False
        return

if __name__ == "__main__":
    regular_net = Regular_Net()
    n_parameters = sum(p.numel() for p in regular_net.parameters())
    print("Regular Net number of parameters (both sum rate and min rate net combined): ", n_parameters) 
    n_parameters = sum(p.numel() for p in regular_net.sumRate_feature_module.parameters())
    print("The feature module number of parameters: ", n_parameters)
    n_parameters = sum(p.numel() for p in regular_net.sumRate_optimizer_module.parameters())
    print("The optimizer module number of parameters: ", n_parameters)
