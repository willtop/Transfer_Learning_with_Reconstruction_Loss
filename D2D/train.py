# Training script for all the models

from random import gammavariate
import numpy as np
from utils import FP_power_control, compute_SINRs, compute_rates
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import argparse
import matplotlib.pyplot as plt
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
    train_losses = np.load(f"Trained_Models/{SOURCETASK['Task']}-to-{TARGETTASK['Task']}/train_losses_sourceTask_{SETTING_STRING}.npy")
    valid_losses = np.load(f"Trained_Models/{SOURCETASK['Task']}-to-{TARGETTASK['Task']}/valid_losses_sourceTask_{SETTING_STRING}.npy")
    axes[0][0].set_xlabel("Training Steps", fontsize=label_fontsize)
    axes[0][0].set_ylabel("Training Loss (Source Task)", fontsize=label_fontsize)
    axes[0][0].plot(train_losses[:,0], 'g', linewidth=line_width, label="Regular Learning")
    axes[0][0].plot(train_losses[:,1], 'b', linewidth=line_width, label="Conventional Transfer")
    axes[0][0].plot(train_losses[:,2], 'r', linewidth=line_width, label="Transfer with Reconstruct")
    axes[0][0].plot(train_losses[:,3], 'r--', linewidth=line_width, label="Transfer with Reconstruct (Total Loss)")
    axes[0][1].set_xlabel("Training Steps", fontsize=label_fontsize)
    axes[0][1].set_ylabel("Validation Loss (Source Task)", fontsize=label_fontsize)
    axes[0][1].plot(valid_losses[:,0], 'g', linewidth=line_width, label="Regular Learning")
    axes[0][1].plot(valid_losses[:,1], 'b', linewidth=line_width, label="Conventional Transfer")
    axes[0][1].plot(valid_losses[:,2], 'r', linewidth=line_width, label="Transfer with Reconstruct")
    axes[0][1].plot(valid_losses[:,3], 'r--', linewidth=line_width, label="Transfer with Reconstruct (Total Loss)")
    # Plot for target task
    train_losses = np.load(f"Trained_Models/{SOURCETASK['Task']}-to-{TARGETTASK['Task']}/train_losses_targetTask_{SETTING_STRING}.npy")
    valid_losses = np.load(f"Trained_Models/{SOURCETASK['Task']}-to-{TARGETTASK['Task']}/valid_losses_targetTask_{SETTING_STRING}.npy")
    axes[1][0].set_xlabel("Training Steps", fontsize=label_fontsize)
    axes[1][0].set_ylabel("Training Loss (Target Task)", fontsize=label_fontsize)
    axes[1][0].plot(train_losses[:,0], 'g', linewidth=line_width, label="Regular Learning")
    axes[1][0].plot(train_losses[:,1], 'b', linewidth=line_width, label="Conventional Transfer")
    axes[1][0].plot(train_losses[:,2], 'r', linewidth=line_width, label="Transfer with Reconstruct")
    axes[1][1].set_xlabel("Training Steps", fontsize=label_fontsize)
    axes[1][1].set_ylabel("Validation Loss (Target Task)", fontsize=label_fontsize)
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

def shuffle_divide_batches(inputs, n_batches):
    n_layouts = np.shape(inputs)[0]
    perm = np.arange(n_layouts)
    np.random.shuffle(perm)
    inputs_batches = np.split(inputs[perm], n_batches, axis=0)
    return inputs_batches


if(__name__=="__main__"):
    parser = argparse.ArgumentParser(description="main script argument parser")
    parser.add_argument("--plot", default=False, help="Whether plot the training curves (executed after training)")
    args = parser.parse_args()
    if args.plot:
        plot_training_curves()
        exit(0)


    """ 
    Source-Task Training 
    """
    print(f"<<<<<<<<<<<<<<<<<<<<<<<[{SOURCETASK['Task']}]->[{TARGETTASK['Task']}]>>>>>>>>>>>>>>>>>>>>>>")
    regular_net, transfer_net, ae_transfer_net = \
            Regular_Net().to(DEVICE), Transfer_Net().to(DEVICE), Autoencoder_Transfer_Net().to(DEVICE)
    print("[Source Task] Loading data...")
    g = np.load(f"Data/g_sourceTask_{SETTING_STRING}.npy")
    assert np.shape(g)[0] == SOURCETASK['Train'] + SOURCETASK['Valid']
    g_train, g_valid = g[:SOURCETASK['Train']], g[-SOURCETASK['Valid']:]
    assert SOURCETASK['Train'] % SOURCETASK['Minibatch_Size'] == 0
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
        g_batches = shuffle_divide_batches(g_train, n_minibatches)
        for j in range(n_minibatches):
            optimizer_regular.zero_grad()
            optimizer_transfer.zero_grad()
            optimizer_ae_transfer.zero_grad()
            # Regular Net
            _, objAvg = regular_net.sourceTask_powerControl(torch.tensor(g_batches[j], dtype=torch.float32).to(DEVICE))
            regular_loss = -objAvg
            # Transfer Net
            _, objAvg = transfer_net.sourceTask_powerControl(torch.tensor(g_batches[j], dtype=torch.float32).to(DEVICE))
            transfer_loss = -objAvg
            # AutoEncoder Transfer Net
            _, objAvg, reconstruct_loss = ae_transfer_net.sourceTask_powerControl(torch.tensor(g_batches[j], dtype=torch.float32).to(DEVICE))
            ae_transfer_loss = -objAvg
            ae_transfer_loss_combined = ae_transfer_loss + SOURCETASK['Loss_Combine_Weight'] * reconstruct_loss
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
                    _, objAvg = regular_net.sourceTask_powerControl(torch.tensor(g_valid, dtype=torch.float32).to(DEVICE))
                    regular_loss = -objAvg.item()
                    _, objAvg = transfer_net.sourceTask_powerControl(torch.tensor(g_valid, dtype=torch.float32).to(DEVICE))
                    transfer_loss = -objAvg.item()
                    _, objAvg, reconstruct_loss = ae_transfer_net.sourceTask_powerControl(torch.tensor(g_valid, dtype=torch.float32).to(DEVICE))
                    ae_transfer_loss = -objAvg.item()
                    ae_transfer_loss_combined = ae_transfer_loss + SOURCETASK['Loss_Combine_Weight'] * reconstruct_loss.item()
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
                np.save(f"Trained_Models/{SOURCETASK['Task']}-to-{TARGETTASK['Task']}/train_losses_sourceTask_{SETTING_STRING}.npy", np.array(train_loss_eps))
                np.save(f"Trained_Models/{SOURCETASK['Task']}-to-{TARGETTASK['Task']}/valid_losses_sourceTask_{SETTING_STRING}.npy", np.array(valid_loss_eps))

    """ 
    Target Task Training
    """
    print("[Target Task] Loading data...")
    g = np.load(f"Data/g_targetTask_{SETTING_STRING}.npy")
    assert np.shape(g)[0] == TARGETTASK['Train'] + TARGETTASK['Valid']
    g_train, g_valid = g[:TARGETTASK['Train']], g[-TARGETTASK['Valid']:]
    assert TARGETTASK['Train'] % TARGETTASK['Minibatch_Size'] == 0
    n_minibatches = int(TARGETTASK['Train'] / TARGETTASK['Minibatch_Size'])
    print(f"[Target Task] Data Loaded! With {TARGETTASK['Train']} training samples ({n_minibatches} minibatches) and {TARGETTASK['Valid']} validation samples.")

    # Create neural network objects again so they load weights from previous early stopping best checkpoint on source task
    regular_net, transfer_net, ae_transfer_net = \
            Regular_Net().to(DEVICE), Transfer_Net().to(DEVICE), Autoencoder_Transfer_Net().to(DEVICE)
    print("[Target Task] Freeze the neural network parameters up to the feature layers...")
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
        g_batches = shuffle_divide_batches(g_train, n_minibatches)
        for j in range(n_minibatches):
            optimizer_regular.zero_grad()
            optimizer_transfer.zero_grad()
            optimizer_ae_transfer.zero_grad()
            # Regular Net
            _, objAvg = regular_net.targetTask_powerControl(torch.tensor(g_batches[j], dtype=torch.float32).to(DEVICE))
            regular_loss = -objAvg
            # Transfer Net
            _, objAvg = transfer_net.targetTask_powerControl(torch.tensor(g_batches[j], dtype=torch.float32).to(DEVICE))
            transfer_loss = -objAvg
            # AutoEncoder Transfer Net
            _, objAvg = ae_transfer_net.targetTask_powerControl(torch.tensor(g_batches[j], dtype=torch.float32).to(DEVICE))
            ae_transfer_loss = -objAvg
            # Training and recording loss
            regular_loss.backward(); optimizer_regular.step()
            transfer_loss.backward(); optimizer_transfer.step()
            ae_transfer_loss.backward(); optimizer_ae_transfer.step()           
            regular_loss_ep += regular_loss.item()
            transfer_loss_ep += transfer_loss.item()
            ae_transfer_loss_ep += ae_transfer_loss.item()
        # Validation
        with torch.no_grad():
            _, objAvg = regular_net.targetTask_powerControl(torch.tensor(g_valid, dtype=torch.float32).to(DEVICE))
            regular_loss = -objAvg.item()
            _, objAvg = transfer_net.targetTask_powerControl(torch.tensor(g_valid, dtype=torch.float32).to(DEVICE))
            transfer_loss = -objAvg.item()
            _, objAvg = ae_transfer_net.targetTask_powerControl(torch.tensor(g_valid, dtype=torch.float32).to(DEVICE))
            ae_transfer_loss = -objAvg.item()
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
        np.save(f"Trained_Models/{SOURCETASK['Task']}-to-{TARGETTASK['Task']}/train_losses_targetTask_{SETTING_STRING}.npy", np.array(train_loss_eps))
        np.save(f"Trained_Models/{SOURCETASK['Task']}-to-{TARGETTASK['Task']}/valid_losses_targetTask_{SETTING_STRING}.npy", np.array(valid_loss_eps))

    print(f"[{SOURCETASK['Task']}]->[{TARGETTASK['Task']}] Training finished!")
