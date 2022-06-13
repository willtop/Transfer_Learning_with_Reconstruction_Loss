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
        self.feature_length = int(N_BS * N_BS_ANTENNAS * 0.5)
        # attributes to be overridden by subclasses
        self.model_type = None
        self.model_path = None
        self.model_path_noEarlyStop = None
        # general attributes
        self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Trained_Models")
        self.channels_mean = torch.tensor(np.load(os.path.join(self.base_dir, "Channels_Stats", "channels_train_mean.npy")),dtype=torch.float32).to(DEVICE)
        self.channels_std = torch.tensor(np.load(os.path.join(self.base_dir, "Channels_Stats", "channels_train_std.npy")),dtype=torch.float32).to(DEVICE)

    def _preprocess_inputs(self, inputs):
        assert inputs.ndim == 3
        inputs = inputs.view(-1, N_BS*N_PILOTS)
        inputs = torch.view_as_real(inputs)
        inputs = inputs.view(-1, N_BS*N_PILOTS*2)
        return inputs
    
    def _compute_beamformer_gains(self, beamformers_raw, channels):
        n_networks = beamformers_raw.size(0)
        assert beamformers_raw.size() == (n_networks, N_BS*N_BS_ANTENNAS*2)
        assert channels.size() == (n_networks, N_BS, N_BS_ANTENNAS)
        beamformers_raw = torch.view_as_complex(beamformers_raw).view(n_networks, N_BS, N_BS_ANTENNAS)
        # normalize beamformers to unit power
        beamformers = beamformers_raw / beamformers_raw.norm(dim=-1, keepdim=True)
        # compute beamformer gain across all BSs
        channel_gains_tmp = torch.sum(beamformers * channels.conj(), dim=-1)
        channel_gains = channel_gains.abs().pow(2)
        assert channel_gains.size() == (n_networks, N_BS)
        return channel_gains

    def sourcetask(self):
        raise NotImplementedError

    def targettask(self):
        raise NotImplementedError

    def _load_model(self, early_stop):
        model_path_to_load = self.model_path if early_stop else self.model_path_noEarlyStop
        if os.path.exists(model_path_to_load):
            if not torch.cuda.is_available():
                print("Working on a CPU! Loading neural nets while mapping storages on CPU...")
                self.load_state_dict(torch.load(model_path_to_load, map_location=torch.device('cpu')))
            else:
                self.load_state_dict(torch.load(model_path_to_load))
            print("[{}] Model loaded from {}".format(self.model_type, model_path_to_load))
        else:
            print("[{}] Train from scratch!".format(self.model_type))
        return

    def save_model(self, early_stop):
        model_path_to_save = self.model_path if early_stop else self.model_path_noEarlyStop
        torch.save(self.state_dict(), model_path_to_save)
        print("[{}] Model saved at {}".format(self.model_type, model_path_to_save))
        return

    # Modules to compose different types of neural net
    def _construct_feature_module(self):
        new_module = nn.ModuleList()
        new_module.append(nn.Linear(N_BS*N_PILOTS*2, int(1.5*N_BS*N_BS_ANTENNAS)))
        new_module.append(nn.ReLU())
        new_module.append(nn.Linear(int(1.5*N_BS*N_BS_ANTENNAS), int(1.5*N_BS*N_BS_ANTENNAS)))
        new_module.append(nn.ReLU())
        new_module.append(nn.Linear(int(1.5*N_BS*N_BS_ANTENNAS), self.feature_length))
        new_module.append(nn.ReLU())
        return new_module
    
    def _construct_localization_optimizer_module(self):
        new_module = nn.ModuleList()
        new_module.append(nn.Linear(self.feature_length, 30))
        new_module.append(nn.ReLU())
        new_module.append(nn.Linear(30, 15))
        new_module.append(nn.ReLU())
        new_module.append(nn.Linear(15, 3))
        # regard each coordinate predicted as normalized by the side-length of the region
        new_module.append(nn.Sigmoid())
        return new_module
    
    def _construct_beamforming_optimizer_module(self):
        new_module = nn.ModuleList()
        new_module.append(nn.Linear(self.feature_length, 150))
        new_module.append(nn.ReLU())
        new_module.append(nn.Linear(150, 100))
        new_module.append(nn.ReLU())
        new_module.append(nn.Linear(100, 2*N_BS*N_BS_ANTENNAS))
        return new_module
    
    def _construct_optimizer_module(self, task_obj):
        if task_obj['Task'] == 'Localization':
            return self._construct_localization_optimizer_module()
        return self._construct_beamforming_optimizer_module()

    def _construct_model_path(self, model_type):
        model_path = os.path.join(self.base_dir, f"{SOURCETASK['Task']}-to-{TARGETTASK['Task']}", f"{model_type}.ckpt")
        return model_path
    
    def _construct_model_path_noEarlyStop(self, model_type):
        model_path_noEarlyStop = os.path.join(self.base_dir, f"{SOURCETASK['Task']}-to-{TARGETTASK['Task']}", f"{model_type}_noEarlyStop.ckpt")
        return model_path_noEarlyStop

class Regular_Net(Neural_Net):
    def __init__(self, early_stop=True):
        super().__init__()
        self.model_type = "Regular"
        self.model_path = self._construct_model_path(self.model_type)
        self.model_path_noEarlyStop = self._construct_model_path_noEarlyStop(self.model_type)
        self.sourcetask_feature_module = self._construct_feature_module()
        self.targettask_feature_module = self._construct_feature_module()
        self.sourcetask_optimizer_module = self._construct_optimizer_module(SOURCETASK)
        self.targettask_optimizer_module = self._construct_optimizer_module(TARGETTASK)
        self._load_model(early_stop)

    def sourcetask(self, x, channels):
        x = self._preprocess_input(x)
        for lyr in self.sourcetask_feature_module:
            x = lyr(x)
        for lyr  in self.sourcetask_optimizer_module:
            x = lyr(x)
        if SOURCETASK['Task'] == "Beamforming":
            channel_gains = self._compute_beamformer_gains(x, channels)
            return channel_gains.sum(dim=1).mean()
        return x # for localization

    def targettask(self, x, channels):
        x = self._preprocess_input(x)
        for lyr in self.targettask_feature_module:
            x = lyr(x)
        for lyr  in self.targettask_optimizer_module:
            x = lyr(x)
        if TARGETTASK['Task'] == "Beamforming":
            channel_gains = self._compute_beamformer_gains(x, channels)
            return channel_gains.sum(dim=1).mean()
        return x # for localization


class Transfer_Net(Neural_Net):
    def __init__(self, early_stop=True):
        super().__init__()
        self.model_type = "Transfer"
        self.model_path = self._construct_model_path(self.model_type)
        self.model_path_noEarlyStop = self._construct_model_path_noEarlyStop(self.model_type)
        self.feature_module = self._construct_feature_module()
        self.sourcetask_optimizer_module = self._construct_optimizer_module(SOURCETASK)
        self.targettask_optimizer_module = self._construct_optimizer_module(TARGETTASK)
        self._load_model(early_stop)

    def sourcetask(self, x, channels):
        x = self.preprocess_input(x)
        for lyr in self.feature_module:
            x = lyr(x)
        for lyr  in self.sourcetask_optimizer_module:
            x = lyr(x)
        if SOURCETASK['Task'] == "Beamforming":
            channel_gains = self._compute_beamformer_gains(x, channels)
            return channel_gains.sum(dim=1).mean()
        return x # for localization

    # freeze parameters for transfer learning
    def freeze_parameters(self):
        for lyr in self.feature_module:
            for para in lyr.parameters():
                para.requires_grad = False
        return

    def targettask(self, x, channels):
        x = self.preprocess_input(x)
        for lyr in self.feature_module:
            x = lyr(x)
        for lyr  in self.targettask_optimizer_module:
            x = lyr(x)
        if TARGETTASK['Task'] == "Beamforming":
            channel_gains = self._compute_beamformer_gains(x, channels)
            return channel_gains.sum(dim=1).mean()
        return x # for localization

class Autoencoder_Transfer_Net(Neural_Net):
    def __init__(self, early_stop=True):
        super().__init__()
        self.model_type = "Autoencoder_Transfer"
        self.model_path = self._construct_model_path(self.model_type)
        self.model_path_noEarlyStop = self._construct_model_path_noEarlyStop(self.model_type)
        self.feature_module = self._construct_feature_module()
        self.decoder_module = self._construct_decoder_module()
        self.sourcetask_optimizer_module = self._construct_optimizer_module(SOURCETASK)
        self.targettask_optimizer_module = self._construct_optimizer_module(TARGETTASK)
        self._load_model(early_stop)

    def _construct_decoder_module(self):
        decoder_module = nn.ModuleList()
        decoder_module.append(nn.Linear(self.feature_length, 30))
        decoder_module.append(nn.ReLU())
        decoder_module.append(nn.Linear(30, 20))
        decoder_module.append(nn.ReLU())
        # factors to be reconstructed aggregated over all BSs
        decoder_module.append(nn.Linear(20, N_BS*N_FACTORS))
        return decoder_module

    def sourcetask(self, x, channels):
        x = self._preprocess_input(x)
        for lyr in self.feature_module:
            x = lyr(x)
        factors_reconstructed = torch.clone(x)
        # try to reconstruct factors
        for lyr in self.decoder_module:
            factors_reconstructed = lyr(factors_reconstructed)
        # optimize for the objective
        for lyr in self.sourcetask_optimizer_module:
            x = lyr(x)
        if SOURCETASK['Task'] == "Beamforming":
            channel_gains = self._compute_beamformer_gains(x, channels)
            x = channel_gains.sum(dim=1).mean()
        return x, factors_reconstructed
    
    # freeze parameters for transfer learning
    def freeze_parameters(self):
        for lyr in self.new_module:
            for para in lyr.parameters():
                para.requires_grad = False
        return

    def targettask(self, x, channels):
        x = self._preprocess_input(x)
        for lyr in self.feature_module:
            x = lyr(x)
        for lyr in self.targettask_optimizer_module:
            x = lyr(x)
        if TARGETTASK['Task'] == "Beamforming":
            channel_gains = self._compute_beamformer_gains(x, channels)
            return channel_gains.sum(dim=1).mean()
        return x # for localization

if __name__ == "__main__":
    ae_transfer_net = Autoencoder_Transfer_Net()
    print("The number of features in the feature layer: ", ae_transfer_net.feature_length)
    n_parameters = sum(p.numel() for p in ae_transfer_net.feature_module.parameters())
    print("The feature module number of parameters: ", n_parameters)
    n_parameters = sum(p.numel() for p in ae_transfer_net.sourcetask_optimizer_module.parameters())
    print(f"The optimizer module number of parameters on task {SOURCETASK['Task']}: ", n_parameters)
    n_parameters = sum(p.numel() for p in ae_transfer_net.targettask_optimizer_module.parameters())
    print(f"The optimizer module number of parameters on task {TARGETTASK['Task']}: ", n_parameters)
    n_parameters = sum(p.numel() for p in ae_transfer_net.decoder_module.parameters())
    print("The decoder module number of parameters: ", n_parameters)
    print("Script finished!")