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
        self.feature_length = int(N_LINKS * N_LINKS * 0.5)
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

    def sourceTask_powerControl(self):
        raise NotImplementedError

    def targetTask_powerControl(self):
        raise NotImplementedError

    def compute_rates(self, pc, channels):
        dl = torch.diagonal(channels, dim1=1, dim2=2)
        cl = channels * (1.0-torch.eye(N_LINKS, dtype=torch.float).to(DEVICE))
        sinrs_numerators = pc * dl
        sinrs_denominators = torch.squeeze(torch.matmul(cl, torch.unsqueeze(pc,-1)), -1) + NOISE_POWER/TX_POWER
        sinrs = sinrs_numerators / (sinrs_denominators * SINR_GAP)
        return torch.log(1+sinrs) # Un-normalized for better scaled gradients

    def compute_objective(self, rates, task):
        n_layouts = rates.size(dim=0)
        if task == 'Sum':
            obj = torch.sum(rates, dim=1)
        elif task == 'Min':
            obj, _ = torch.min(rates, dim=1)
        elif task == 'Harmonic':
            obj = N_LINKS / torch.sum(1/rates, dim=1)
        else:
            print(f"{task} objective computation unimplemented yet!")
            exit(1)
        assert obj.size() == (n_layouts,)
        return obj

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
        feature_module.append(nn.Linear(N_LINKS*N_LINKS, 2*N_LINKS*N_LINKS))
        feature_module.append(nn.ReLU())
        feature_module.append(nn.Linear(2*N_LINKS*N_LINKS, 2*N_LINKS*N_LINKS))
        feature_module.append(nn.ReLU())
        feature_module.append(nn.Linear(2*N_LINKS*N_LINKS, self.feature_length))
        feature_module.append(nn.ReLU())
        return feature_module
    
    def construct_optimizer_module(self):
        optimizer_module = nn.ModuleList()
        optimizer_module.append(nn.Linear(self.feature_length, 2*N_LINKS))
        optimizer_module.append(nn.ReLU())
        optimizer_module.append(nn.Linear(2*N_LINKS, N_LINKS))
        optimizer_module.append(nn.ReLU())
        optimizer_module.append(nn.Linear(N_LINKS, N_LINKS))
        # with power control output being 0~1
        optimizer_module.append(nn.Sigmoid())
        return optimizer_module

    def _construct_model_path(self, model_type):
        return os.path.join(self.base_dir, f"{SOURCETASK['Task']}-{TARGETTASK['Task']}", f"{model_type}_{SETTING_STRING}.ckpt")

class Regular_Net(Neural_Net):
    def __init__(self):
        super().__init__()
        self.model_type = "Regular"
        self.model_path = self._construct_model_path(self.model_type)
        self.sourceTask_feature_module = self.construct_feature_module()
        self.sourceTask_optimizer_module = self.construct_optimizer_module()
        self.targetTask_feature_module = self.construct_feature_module()
        self.targetTask_optimizer_module = self.construct_optimizer_module()
        self.load_model()

    def sourceTask_powerControl(self, g):
        x = self.preprocess_input(g)
        for lyr in self.sourceTask_feature_module:
            x = lyr(x)
        for lyr  in self.sourceTask_optimizer_module:
            x = lyr(x)
        rates = self.compute_rates(x, g)
        obj = self.compute_objective(rates, SOURCETASK['Task'])
        obj = torch.mean(obj)
        return x, obj

    def targetTask_powerControl(self, g):
        x = self.preprocess_input(g)
        for lyr in self.targetTask_feature_module:
            x = lyr(x)
        for lyr  in self.targetTask_optimizer_module:
            x = lyr(x)
        rates = self.compute_rates(x, g)
        obj = self.compute_objective(rates, TARGETTASK['Task'])
        obj = torch.mean(obj)
        return x, obj


class Transfer_Net(Neural_Net):
    def __init__(self):
        super().__init__()
        self.model_type = "Transfer"
        self.model_path = self._construct_model_path(self.model_type)
        self.feature_module = self.construct_feature_module()
        self.sourceTask_optimizer_module = self.construct_optimizer_module()
        self.targetTask_optimizer_module = self.construct_optimizer_module()
        self.load_model()

    def sourceTask_powerControl(self, g):
        x = self.preprocess_input(g)
        for lyr in self.feature_module:
            x = lyr(x)
        for lyr  in self.sourceTask_optimizer_module:
            x = lyr(x)
        rates = self.compute_rates(x, g)
        obj = self.compute_objective(rates, SOURCETASK['Task'])
        obj = torch.mean(obj)
        return x, obj

    # freeze parameters for transfer learning
    def freeze_parameters(self):
        for lyr in self.feature_module:
            for para in lyr.parameters():
                para.requires_grad = False
        return

    def targetTask_powerControl(self, g):
        x = self.preprocess_input(g)
        for lyr in self.feature_module:
            x = lyr(x)
        for lyr  in self.targetTask_optimizer_module:
            x = lyr(x)
        rates = self.compute_rates(x, g)
        obj = self.compute_objective(rates, TARGETTASK['Task'])
        obj = torch.mean(obj)
        return x, obj

class Autoencoder_Transfer_Net(Neural_Net):
    def __init__(self):
        super().__init__()
        self.model_type = "Autoencoder_Transfer"
        self.model_path = self._construct_model_path(self.model_type)
        self.feature_module = self.construct_feature_module()
        self.decoder_module = self.construct_decoder_module()
        self.sourceTask_optimizer_module = self.construct_optimizer_module()
        self.targetTask_optimizer_module = self.construct_optimizer_module()
        # for auto-encoder reconstruction loss
        self.reconstruct_loss_func = nn.MSELoss(reduction='mean')
        self.load_model()

    def construct_decoder_module(self):
        decoder_module = nn.ModuleList()
        decoder_module.append(nn.Linear(self.feature_length, 2*N_LINKS*N_LINKS))
        decoder_module.append(nn.ReLU())
        decoder_module.append(nn.Linear(2*N_LINKS*N_LINKS, N_LINKS*N_LINKS))
        return decoder_module

    def sourceTask_powerControl(self, g):
        x = self.preprocess_input(g)
        inputs = torch.clone(x)
        for lyr in self.feature_module:
            x = lyr(x)
        features = torch.clone(x)
        # try to reconstruct inputs
        for lyr in self.decoder_module:
            x = lyr(x)
        inputs_reconstructed = x
        for lyr in self.sourceTask_optimizer_module:
            features = lyr(features)
        pc = features
        rates = self.compute_rates(pc, g)
        obj = self.compute_objective(rates, SOURCETASK['Task'])
        obj = torch.mean(obj)
        return pc, obj, self.reconstruct_loss_func(inputs, inputs_reconstructed)
    
    # freeze parameters for transfer learning
    def freeze_parameters(self):
        for lyr in self.feature_module:
            for para in lyr.parameters():
                para.requires_grad = False
        return

    def targetTask_powerControl(self, g):
        x = self.preprocess_input(g)
        for lyr in self.feature_module:
            x = lyr(x)
        for lyr in self.targetTask_optimizer_module:
            x = lyr(x)
        rates = self.compute_rates(x, g)
        obj = self.compute_objective(rates, TARGETTASK['Task'])
        obj = torch.mean(obj)
        return x, obj

if __name__ == "__main__":
    ae_transfer_net = Autoencoder_Transfer_Net()
    n_parameters = sum(p.numel() for p in ae_transfer_net.feature_module.parameters())
    print("The feature module number of parameters: ", n_parameters)
    n_parameters = sum(p.numel() for p in ae_transfer_net.sourceTask_optimizer_module.parameters())
    print("The optimizer module number of parameters: ", n_parameters)
    n_parameters = sum(p.numel() for p in ae_transfer_net.decoder_module.parameters())
    print("The decoder module number of parameters: ", n_parameters)
