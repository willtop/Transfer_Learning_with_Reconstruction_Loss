# Training script for all the models

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import argparse
import matplotlib.pyplot as plt
import utils 
from setup import *
from neural_nets import Regular_Net, Transfer_Net, Autoencoder_Transfer_Net

def plot_training_curves():
    legend_fontsize = 19
    label_fontsize = 20
    tick_fontsize = 15
    line_width = 1.4
    print("[D2D] Plotting training curves...")

    fig, axes = plt.subplots(2,2)
    # Plot for source task
    train_losses = np.load(f"Trained_Models/{SOURCETASK['Task']}-to-{TARGETTASK['Task']}/train_losses_sourcetask_{SOURCETASK['Task']}.npy")
    valid_losses = np.load(f"Trained_Models/{SOURCETASK['Task']}-to-{TARGETTASK['Task']}/valid_losses_sourcetask_{SOURCETASK['Task']}.npy")
    axes[0][0].set_xlabel("Training Steps", fontsize=label_fontsize)
    axes[0][0].set_ylabel(f"Training Loss (Source Task: {SOURCETASK['Task']})", fontsize=label_fontsize)
    axes[0][0].plot(train_losses[:,0], 'g', linewidth=line_width, label="Regular Learning")
    axes[0][0].plot(train_losses[:,1], 'b', linewidth=line_width, label="Conventional Transfer")
    axes[0][0].plot(train_losses[:,2], 'r', linewidth=line_width, label="Transfer with Reconstruct")
    axes[0][0].plot(train_losses[:,3], 'r--', linewidth=line_width, label="Transfer with Reconstruct (Total Loss)")
    axes[0][1].set_xlabel("Training Steps", fontsize=label_fontsize)
    axes[0][1].set_ylabel(f"Validation Loss (Source Task: {SOURCETASK['Task']})", fontsize=label_fontsize)
    axes[0][1].plot(valid_losses[:,0], 'g', linewidth=line_width, label="Regular Learning")
    axes[0][1].plot(valid_losses[:,1], 'b', linewidth=line_width, label="Conventional Transfer")
    axes[0][1].plot(valid_losses[:,2], 'r', linewidth=line_width, label="Transfer with Reconstruct")
    axes[0][1].plot(valid_losses[:,3], 'r--', linewidth=line_width, label="Transfer with Reconstruct (Total Loss)")
    # Plot for target task
    train_losses = np.load(f"Trained_Models/{SOURCETASK['Task']}-to-{TARGETTASK['Task']}/train_losses_targettask_{TARGETTASK['Task']}.npy")
    valid_losses = np.load(f"Trained_Models/{SOURCETASK['Task']}-to-{TARGETTASK['Task']}/valid_losses_targettask_{TARGETTASK['Task']}.npy")
    axes[1][0].set_xlabel("Training Steps", fontsize=label_fontsize)
    axes[1][0].set_ylabel(f"Training Loss (Target Task: {TARGETTASK['Task']})", fontsize=label_fontsize)
    axes[1][0].plot(train_losses[:,0], 'g', linewidth=line_width, label="Regular Learning")
    axes[1][0].plot(train_losses[:,1], 'b', linewidth=line_width, label="Conventional Transfer")
    axes[1][0].plot(train_losses[:,2], 'r', linewidth=line_width, label="Transfer with Reconstruct")
    axes[1][1].set_xlabel("Training Steps", fontsize=label_fontsize)
    axes[1][1].set_ylabel(f"Validation Loss (Target Task: {TARGETTASK['Task']})", fontsize=label_fontsize)
    axes[1][1].plot(valid_losses[:,0], 'g', linewidth=line_width, label="Regular Learning")
    axes[1][1].plot(valid_losses[:,1], 'b', linewidth=line_width, label="Conventional Transfer")
    axes[1][1].plot(valid_losses[:,2], 'r', linewidth=line_width, label="Transfer with Reconstruct")
    for ax in axes.flatten():
        ax.legend(prop={'size':legend_fontsize}, loc='upper right')
        ax.tick_params(axis='both', labelsize=tick_fontsize)
    plt.subplots_adjust(left=0.075,right=0.95,bottom=0.1,top=0.95)
    plt.show()
    print("Finished plotting.")
    return

# Pytorch computation
def compute_beamformer_gains(beamformers, channels):
    n_networks = beamformers.size(0)
    assert channels.size() == (n_networks, N_BS, N_BS_ANTENNAS) and \
           beamformers.size() == (n_networks, N_BS, N_BS_ANTENNAS)
    # ensure beamformers are normalized to unit power
    beamformer_powers = beamformers.norm(dim=-1).flatten()
    assert torch.all(beamformer_powers<1.01) and \
            torch.all(beamformer_powers>0.99)
    # compute beamformer gain across all BSs
    channel_gains = torch.sum(beamformers * channels.conj(), dim=-1)
    channel_gains = channel_gains.abs().pow(2)
    assert channel_gains.size() == (n_networks, N_BS)
    return channel_gains.sum(dim=1).mean()

def shuffle_divide_batches(n_batches, *args):
    outputs = []
    n_layouts = np.shape(args[0])[0]
    perm = np.arange(n_layouts)
    np.random.shuffle(perm)
    for arg in args:
        assert np.shape(arg)[0] == n_layouts
        data_batches = np.split(arg[perm], n_batches, axis=0)
        outputs.append(data_batches)
    return outputs

LOCALIZATION_LOSS_FUNC = nn.MSELoss(reduction="mean")
RECONSTRUCTION_LOSS_FUNC = nn.MSELoss(reduction="mean")

if(__name__=="__main__"):
    parser = argparse.ArgumentParser(description="main script argument parser")
    parser.add_argument("--plot", default=False, help="Whether plot the training curves (executed after training)")
    args = parser.parse_args()
    if args.plot:
        plot_training_curves()
        exit(0)

    print(f"<<<<<<<<<<<<<<<<<<<<<<<[{SOURCETASK['Task']}]->[{TARGETTASK['Task']}]>>>>>>>>>>>>>>>>>>>>>>")
    """ 
    Source-Task Training 
    """
    regular_net, transfer_net, ae_transfer_net = \
            Regular_Net().to(DEVICE), Transfer_Net().to(DEVICE), Autoencoder_Transfer_Net().to(DEVICE)
    print(f"[Source Task {SOURCETASK['Task']}] Loading data...")
    uelocs = np.load("Data/uelocs_sourcetask.npy")
    channels = np.load("Data/channels_sourcetask.npy")
    factors = np.load("Data/factors_sourcetask.npy")
    measures = utils.obtain_measured_uplink_signals(channels)
    assert np.shape(uelocs)[0] == np.shape(channels)[0] == np.shape(factors)[0] == \
                SOURCETASK['Train'] + SOURCETASK['Valid']
    uelocs_train, uelocs_valid = uelocs[:SOURCETASK['Train']], uelocs[-SOURCETASK['Valid']:]
    channels_train, channels_valid = channels[:SOURCETASK['Train']], channels[-SOURCETASK['Valid']:]
    factors_train, factors_valid = factors[:SOURCETASK['Train']], factors[-SOURCETASK['Valid']:]
    measures_train, measures_valid = measures[:SOURCETASK['Train']], measures[-SOURCETASK['Valid']:]
    n_minibatches = int(SOURCETASK['Train'] / SOURCETASK['Minibatch_Size'])
    print(f"[Source Task on {SOURCETASK['Task']}] Data Loaded! With {SOURCETASK['Train']} training samples ({n_minibatches} minibatches) and {SOURCETASK['Valid']} validation samples.")

    optimizer_regular, optimizer_transfer, optimizer_ae_transfer = \
            optim.Adam(regular_net.parameters(), lr=SOURCETASK['Learning_Rate']), \
            optim.Adam(transfer_net.parameters(), lr=SOURCETASK['Learning_Rate']), \
            optim.Adam(ae_transfer_net.parameters(), lr=SOURCETASK['Learning_Rate'])
    regular_loss_min, transfer_loss_min, ae_transfer_loss_combined_min = np.inf, np.inf, np.inf
    train_loss_eps, valid_loss_eps = [], []
    for i in trange(1, SOURCETASK['Epochs']+1):
        regular_loss_ep, transfer_loss_ep, ae_transfer_loss_ep, ae_transfer_loss_combined_ep = 0, 0, 0, 0
        uelocs_batches, channels_batches, factors_batches, measures_batches = \
            shuffle_divide_batches(n_minibatches, uelocs_train, channels_train, factors_train, measures_train)
        for j in range(n_minibatches):
            optimizer_regular.zero_grad()
            optimizer_transfer.zero_grad()
            optimizer_ae_transfer.zero_grad()
            # Regular Net
            outputs = regular_net.sourcetask(torch.tensor(measures_batches[j], dtype=torch.cfloat).to(DEVICE))
            regular_loss = -compute_beamformer_gains(outputs, channels) if SOURCETASK['Task']=="Beamforming" \
                            else LOCALIZATION_LOSS_FUNC(outputs, torch.tensor(uelocs_batches,dtype=torch.float32))
            # Transfer Net
            outputs = transfer_net.sourcetask(torch.tensor(measures_batches[j], dtype=torch.cfloat).to(DEVICE))
            transfer_loss = -compute_beamformer_gains(outputs, channels) if SOURCETASK['Task']=="Beamforming" \
                            else LOCALIZATION_LOSS_FUNC(outputs, torch.tensor(uelocs_batches,dtype=torch.float32))
            # AutoEncoder Transfer Net
            outputs, factors_reconstructed = ae_transfer_net.sourcetask(torch.tensor(measures_batches[j], dtype=torch.cfloat).to(DEVICE))
            ae_transfer_loss = -compute_beamformer_gains(outputs, channels) if SOURCETASK['Task']=="Beamforming" \
                            else LOCALIZATION_LOSS_FUNC(outputs, torch.tensor(uelocs_batches,dtype=torch.float32))
            ae_transfer_loss_combined = ae_transfer_loss + SOURCETASK['Loss_Combine_Weight'] * RECONSTRUCTION_LOSS_FUNC(factors_reconstructed, torch.tensor(factors_batches,dtype=torch.float32).view(-1, N_BS*N_FACTORS))
            # Training and recording loss
            regular_loss.backward(); optimizer_regular.step()
            transfer_loss.backward(); optimizer_transfer.step()
            ae_transfer_loss_combined.backward(); optimizer_ae_transfer.step()           
            regular_loss_ep += regular_loss.item()
            transfer_loss_ep += transfer_loss.item()
            ae_transfer_loss_ep += ae_transfer_loss.item()
            ae_transfer_loss_combined_ep += ae_transfer_loss_combined.item()
            if (j+1) % min(50,n_minibatches) == 0:
                # Validation
                with torch.no_grad():
                    outputs = regular_net.sourcetask(torch.tensor(measures_valid, dtype=torch.float32).to(DEVICE))
                    regular_loss = -compute_beamformer_gains(outputs, channels).item() if SOURCETASK['Task']=="Beamforming" \
                                    else LOCALIZATION_LOSS_FUNC(outputs, torch.tensor(uelocs_valid,dtype=torch.float32)).item()
                    outputs = transfer_net.sourcetask(torch.tensor(measures_valid, dtype=torch.float32).to(DEVICE))
                    transfer_loss = -compute_beamformer_gains(outputs, channels).item() if SOURCETASK['Task']=="Beamforming" \
                                    else LOCALIZATION_LOSS_FUNC(outputs, torch.tensor(uelocs_valid,dtype=torch.float32)).item()
                    outputs, factors_reconstructed = ae_transfer_net.sourcetask(torch.tensor(measures_valid, dtype=torch.float32).to(DEVICE))
                    ae_transfer_loss = -compute_beamformer_gains(outputs, channels).item() if SOURCETASK['Task']=="Beamforming" \
                                    else LOCALIZATION_LOSS_FUNC(outputs, torch.tensor(uelocs_valid,dtype=torch.float32)).item()
                    ae_transfer_loss_combined = ae_transfer_loss + SOURCETASK['Loss_Combine_Weight'] * RECONSTRUCTION_LOSS_FUNC(factors_reconstructed, torch.tensor(factors_valid,dtype=torch.float32).view(-1, N_BS*N_FACTORS)).item()
                train_loss_eps.append([regular_loss_ep/(j+1), transfer_loss_ep/(j+1), ae_transfer_loss_ep/(j+1), ae_transfer_loss_combined_ep/(j+1)])
                valid_loss_eps.append([regular_loss, transfer_loss, ae_transfer_loss, ae_transfer_loss_combined])
                print("[Source Task][Regular] Tr:{:6.3e}; Va:{:6.3e} [Transfer] Tr: {:6.3e}; Va:{:6.3e} [AE Transfer] Tr: {:6.3e}; Va: {:6.3e}".format(
                    regular_loss_ep/(j+1), regular_loss, transfer_loss_ep/(j+1), transfer_loss, ae_transfer_loss_ep/(j+1), ae_transfer_loss))
                # For source task, always do early stopping based on validation losses
                if (regular_loss < regular_loss_min):
                    regular_net.save_model(early_stop=True)
                    regular_loss_min = regular_loss
                if (transfer_loss < transfer_loss_min):
                    transfer_net.save_model(early_stop=True)
                    transfer_loss_min = transfer_loss
                if (ae_transfer_loss_combined < ae_transfer_loss_combined_min):
                    ae_transfer_net.save_model(early_stop=True)
                    ae_transfer_loss_combined_min = ae_transfer_loss_combined    
                np.save(f"Trained_Models/{SOURCETASK['Task']}-to-{TARGETTASK['Task']}/train_losses_sourcetask_{SOURCETASK['Task']}.npy", np.array(train_loss_eps))
                np.save(f"Trained_Models/{SOURCETASK['Task']}-to-{TARGETTASK['Task']}/valid_losses_sourceTask_{SOURCETASK['Task']}.npy", np.array(valid_loss_eps))

    """ 
    Target Task Training
    """
    # Create neural network objects again so they load weights from previous early stopping best checkpoint on source task
    regular_net, transfer_net, ae_transfer_net = \
            Regular_Net().to(DEVICE), Transfer_Net().to(DEVICE), Autoencoder_Transfer_Net().to(DEVICE)
    print(f"[Target Task {TARGETTASK['Task']}] Loading data...")
    uelocs = np.load("Data/uelocs_targettask.npy")
    channels = np.load("Data/channels_targettask.npy")
    factors = np.load("Data/factors_targettask.npy")
    measures = utils.obtain_measured_uplink_signals(channels)
    assert np.shape(uelocs)[0] == np.shape(channels)[0] == np.shape(factors)[0] == \
                TARGETTASK['Train'] + TARGETTASK['Valid']
    uelocs_train, uelocs_valid = uelocs[:TARGETTASK['Train']], uelocs[-TARGETTASK['Valid']:]
    channels_train, channels_valid = channels[:TARGETTASK['Train']], channels[-TARGETTASK['Valid']:]
    factors_train, factors_valid = factors[:TARGETTASK['Train']], factors[-TARGETTASK['Valid']:]
    measures_train, measures_valid = measures[:TARGETTASK['Train']], measures[-TARGETTASK['Valid']:]
    n_minibatches = int(TARGETTASK['Train'] / TARGETTASK['Minibatch_Size'])
    print(f"[Target Task on {TARGETTASK['Task']}] Data Loaded! With {TARGETTASK['Train']} training samples ({n_minibatches} minibatches) and {TARGETTASK['Valid']} validation samples.")
    
    print("[Target Task] Freeze the neural network parameters up to the factor layers...")
    transfer_net.freeze_parameters()
    ae_transfer_net.freeze_parameters()
    optimizer_regular, optimizer_transfer, optimizer_ae_transfer = \
            optim.Adam(filter(lambda para: para.requires_grad, regular_net.parameters()), lr=TARGETTASK['Learning_Rate']), \
            optim.Adam(filter(lambda para: para.requires_grad, transfer_net.parameters()), lr=TARGETTASK['Learning_Rate']), \
            optim.Adam(filter(lambda para: para.requires_grad, ae_transfer_net.parameters()), lr=TARGETTASK['Learning_Rate'])
    regular_loss_min, transfer_loss_min, ae_transfer_loss_min = np.inf, np.inf, np.inf
    train_loss_eps, valid_loss_eps = [], []
    for i in trange(1, TARGETTASK['Epochs']+1):
        regular_loss_ep, transfer_loss_ep, ae_transfer_loss_ep = 0, 0, 0
        uelocs_batches, channels_batches, factors_batches, measures_batches = \
            shuffle_divide_batches(n_minibatches, uelocs_train, channels_train, factors_train, measures_train)
        for j in range(n_minibatches):
            optimizer_regular.zero_grad()
            optimizer_transfer.zero_grad()
            optimizer_ae_transfer.zero_grad()
            # Regular Net
            outputs = regular_net.targettask(torch.tensor(measures_batches[j], dtype=torch.cfloat).to(DEVICE))
            regular_loss = -compute_beamformer_gains(outputs, channels) if TARGETTASK['Task']=="Beamforming" \
                            else LOCALIZATION_LOSS_FUNC(outputs, torch.tensor(uelocs_batches, dtype=torch.float32))
            # Transfer Net
            outputs = transfer_net.targettask(torch.tensor(measures_batches[j], dtype=torch.cfloat).to(DEVICE))
            transfer_loss = -compute_beamformer_gains(outputs, channels) if TARGETTASK['Task']=="Beamforming" \
                            else LOCALIZATION_LOSS_FUNC(outputs, torch.tensor(uelocs_batches, dtype=torch.float32))
            # AutoEncoder Transfer Net
            outputs = ae_transfer_net.targettask(torch.tensor(measures_batches[j], dtype=torch.cfloat).to(DEVICE))
            ae_transfer_loss = -compute_beamformer_gains(outputs, channels) if TARGETTASK['Task']=="Beamforming" \
                            else LOCALIZATION_LOSS_FUNC(outputs, torch.tensor(uelocs_batches, dtype=torch.float32))
            # Training and recording loss
            regular_loss.backward(); optimizer_regular.step()
            transfer_loss.backward(); optimizer_transfer.step()
            ae_transfer_loss.backward(); optimizer_ae_transfer.step()           
            regular_loss_ep += regular_loss.item()
            transfer_loss_ep += transfer_loss.item()
            ae_transfer_loss_ep += ae_transfer_loss.item()
        # Validation
        with torch.no_grad():
            outputs = regular_net.targettask(torch.tensor(measures_valid, dtype=torch.cfloat).to(DEVICE))
            regular_loss = -compute_beamformer_gains(outputs, channels).item() if TARGETTASK['Task']=="Beamforming" \
                            else LOCALIZATION_LOSS_FUNC(outputs, torch.tensor(uelocs_valid, dtype=torch.float32)).item()
            outputs = transfer_net.targettask(torch.tensor(measures_valid, dtype=torch.cfloat).to(DEVICE))
            transfer_loss = -compute_beamformer_gains(outputs, channels).item() if TARGETTASK['Task']=="Beamforming" \
                            else LOCALIZATION_LOSS_FUNC(outputs, torch.tensor(uelocs_valid, dtype=torch.float32)).item()
            outputs = ae_transfer_net.targettask(torch.tensor(measures_valid, dtype=torch.cfloat).to(DEVICE))
            ae_transfer_loss = -compute_beamformer_gains(outputs, channels).item() if TARGETTASK['Task']=="Beamforming" \
                            else LOCALIZATION_LOSS_FUNC(outputs, torch.tensor(uelocs_valid, dtype=torch.float32)).item()
        train_loss_eps.append([regular_loss_ep/(j+1), transfer_loss_ep/(j+1), ae_transfer_loss_ep/(j+1)])
        valid_loss_eps.append([regular_loss, transfer_loss, ae_transfer_loss])
        print("[Target Task][Regular] Tr:{:6.3e}; Va:{:6.3e} [Transfer] Tr: {:6.3e}; Va:{:6.3e} [AE Transfer] Tr: {:6.3e}; Va: {:6.3e}".format(
            regular_loss_ep/(j+1), regular_loss, transfer_loss_ep/(j+1), transfer_loss, ae_transfer_loss_ep/(j+1), ae_transfer_loss))
        # Early stopping based on validation losses
        if (regular_loss < regular_loss_min):
            regular_net.save_model(early_stop=True)
            regular_loss_min = regular_loss
        if (transfer_loss < transfer_loss_min):
            transfer_net.save_model(early_stop=True)
            transfer_loss_min = transfer_loss
        if (ae_transfer_loss < ae_transfer_loss_min):
            ae_transfer_net.save_model(early_stop=True)
            ae_transfer_loss_min = ae_transfer_loss   
        # Also track models trained without early stopping 
        regular_net.save_model(early_stop=False)
        transfer_net.save_model(early_stop=False)
        ae_transfer_net.save_model(early_stop=False)
        np.save(f"Trained_Models/{SOURCETASK['Task']}-to-{TARGETTASK['Task']}/train_losses_targettask_{TARGETTASK['Task']}.npy", np.array(train_loss_eps))
        np.save(f"Trained_Models/{SOURCETASK['Task']}-to-{TARGETTASK['Task']}/valid_losses_targettask_{TARGETTASK['Task']}.npy", np.array(valid_loss_eps))

    print(f"[{SOURCETASK['Task']}]->[{TARGETTASK['Task']}] Training finished!")
