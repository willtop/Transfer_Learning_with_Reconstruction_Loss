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
        self.feature_length = 75
        # attributes to be overridden by subclasses
        self.model_type = None
        self.model_path = None
        # general attributes
        self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Trained_Models")

    def sourcetask(self):
        raise NotImplementedError

    def targettask(self):
        raise NotImplementedError

    def load_model(self):
        if os.path.exists(self.model_path):
            if not torch.cuda.is_available():
                print("Working on a CPU! Loading neural nets while mapping storages on CPU...")
                self.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            else:
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
        feature_module.append(nn.Linear(INPUT_SIZE, 100))
        feature_module.append(nn.ReLU())
        feature_module.append(nn.Linear(100, self.feature_length))
        feature_module.append(nn.ReLU())
        return feature_module
    
    def construct_optimizer_module(self, output_dim):
        optimizer_module = nn.ModuleList()
        optimizer_module.append(nn.Linear(self.feature_length, output_dim))
        # predicting the probability of the output_dim classes
        optimizer_module.append(nn.Softmax(dim=1))
        return optimizer_module

    def _construct_model_path(self, model_type):
        return os.path.join(self.base_dir, f"{TASK_DESCR}", f"{model_type}.ckpt")

class Regular_Net(Neural_Net):
    def __init__(self):
        super().__init__()
        self.model_type = "Regular"
        self.model_path = self._construct_model_path(self.model_type)
        self.sourcetask_feature_module = self.construct_feature_module()
        self.sourcetask_optimizer_module = self.construct_optimizer_module(SOURCETASK['Output_Dim'])
        self.targettask_feature_module = self.construct_feature_module()
        self.targettask_optimizer_module = self.construct_optimizer_module(TARGETTASK['Output_Dim'])
        self.load_model()

    def sourcetask(self, x):
        for lyr in self.sourcetask_feature_module:
            x = lyr(x)
        for lyr  in self.sourcetask_optimizer_module:
            x = lyr(x)
        return x

    def targettask(self, x):
        for lyr in self.targettask_feature_module:
            x = lyr(x)
        for lyr  in self.targettask_optimizer_module:
            x = lyr(x)
        return x


class Transfer_Net(Neural_Net):
    def __init__(self):
        super().__init__()
        self.model_type = "Transfer"
        self.model_path = self._construct_model_path(self.model_type)
        self.feature_module = self.construct_feature_module()
        self.sourcetask_optimizer_module = self.construct_optimizer_module(SOURCETASK['Output_Dim'])
        self.targettask_optimizer_module = self.construct_optimizer_module(TARGETTASK['Output_Dim'])
        self.load_model()

    def sourcetask(self, x):
        for lyr in self.feature_module:
            x = lyr(x)
        for lyr  in self.sourcetask_optimizer_module:
            x = lyr(x)
        return x

    # freeze parameters for transfer learning
    def freeze_parameters(self):
        for lyr in self.feature_module:
            for para in lyr.parameters():
                para.requires_grad = False
        return

    def targettask(self, x):
        for lyr in self.feature_module:
            x = lyr(x)
        for lyr  in self.targettask_optimizer_module:
            x = lyr(x)
        return x

class Autoencoder_Transfer_Net(Neural_Net):
    def __init__(self):
        super().__init__()
        self.model_type = "Autoencoder_Transfer"
        self.model_path = self._construct_model_path(self.model_type)
        self.feature_module = self.construct_feature_module()
        self.decoder_module = self.construct_decoder_module()
        self.sourcetask_optimizer_module = self.construct_optimizer_module(SOURCETASK['Output_Dim'])
        self.targettask_optimizer_module = self.construct_optimizer_module(TARGETTASK['Output_Dim'])
        # for auto-encoder reconstruction loss
        self.reconstruct_loss_func = nn.MSELoss(reduction='mean')
        self.load_model()

    def construct_decoder_module(self):
        decoder_module = nn.ModuleList()
        decoder_module.append(nn.Linear(self.feature_length, 100))
        decoder_module.append(nn.ReLU())
        decoder_module.append(nn.Linear(100, INPUT_SIZE))
        return decoder_module

    def sourcetask(self, x):
        inputs = torch.clone(x)
        for lyr in self.feature_module:
            x = lyr(x)
        features = torch.clone(x)
        # try to reconstruct inputs
        for lyr in self.decoder_module:
            x = lyr(x)
        inputs_reconstructed = x
        for lyr in self.sourcetask_optimizer_module:
            features = lyr(features)
        outputs = features
        return outputs, self.reconstruct_loss_func(inputs, inputs_reconstructed)
    
    # freeze parameters for transfer learning
    def freeze_parameters(self):
        for lyr in self.feature_module:
            for para in lyr.parameters():
                para.requires_grad = False
        return

    def targettask(self, x):
        for lyr in self.feature_module:
            x = lyr(x)
        for lyr in self.targettask_optimizer_module:
            x = lyr(x)
        return x

if __name__ == "__main__":
    ae_transfer_net = Autoencoder_Transfer_Net()
    n_parameters = sum(p.numel() for p in ae_transfer_net.feature_module.parameters())
    print("The feature module number of parameters: ", n_parameters)
    n_parameters = sum(p.numel() for p in ae_transfer_net.sourcetask_optimizer_module.parameters())
    print("The optimizer module number of parameters: ", n_parameters)
    n_parameters = sum(p.numel() for p in ae_transfer_net.decoder_module.parameters())
    print("The decoder module number of parameters: ", n_parameters)
