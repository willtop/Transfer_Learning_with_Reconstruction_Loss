# Training script for all the models

from random import gammavariate
import numpy as np
from utils import FP_power_control, compute_SINRs, compute_rates
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import argparse
import matplotlib.pyplot as plt
from setup import *
from neural_nets import Regular_Net, Transfer_Net, Autoencoder_Transfer_Net


def plot_training_curves():
    print("[D2D] Plotting training curves...")
    fig, axes = plt.subplots(2,2)
    fig.suptitle(f"Loss over D2D networks {SETTING_STRING}")
    # Plot for source task
    train_losses = np.load(f"Trained_Models/{SOURCETASK['Task']}-{TARGETTASK['Task']}/train_losses_sourceTask_{SETTING_STRING}.npy")
    valid_losses = np.load(f"Trained_Models/{SOURCETASK['Task']}-{TARGETTASK['Task']}/valid_losses_sourceTask_{SETTING_STRING}.npy")
    axes[0][0].set_xlabel("Epoches")
    axes[0][0].set_ylabel("Training Losses (Source Task)")
    axes[0][0].plot(train_losses[:,0], 'g', label="Regular Network")
    axes[0][0].plot(train_losses[:,1], 'b', label="Transfer Network")
    axes[0][0].plot(train_losses[:,2], 'r', label="AE Transfer Network")
    axes[0][0].plot(train_losses[:,3], 'r--', label="AE Transfer Network Combined")
    axes[0][0].legend()
    axes[0][1].set_xlabel("Epoches")
    axes[0][1].set_ylabel("Validation Losses (Source Task)")
    axes[0][1].plot(valid_losses[:,0], 'g', label="Regular Network")
    axes[0][1].plot(valid_losses[:,1], 'b', label="Transfer Network")
    axes[0][1].plot(valid_losses[:,2], 'r', label="AE Transfer Network")
    axes[0][1].plot(valid_losses[:,3], 'r--', label="AE Transfer Network Combined")
    axes[0][1].legend()
    # Plot for target task
    train_losses = np.load(f"Trained_Models/{SOURCETASK['Task']}-{TARGETTASK['Task']}/train_losses_targetTask_{SETTING_STRING}.npy")
    valid_losses = np.load(f"Trained_Models/{SOURCETASK['Task']}-{TARGETTASK['Task']}/valid_losses_targetTask_{SETTING_STRING}.npy")
    axes[1][0].set_xlabel("Epoches")
    axes[1][0].set_ylabel("Training Losses (Target Task)")
    axes[1][0].plot(train_losses[:,0], 'g', label="Regular Network")
    axes[1][0].plot(train_losses[:,1], 'b', label="Transfer Network")
    axes[1][0].plot(train_losses[:,2], 'r', label="AE Transfer Network")
    axes[1][0].legend()
    axes[1][1].set_xlabel("Epoches")
    axes[1][1].set_ylabel("Validation Losses (Target Task)")
    axes[1][1].plot(valid_losses[:,0], 'g', label="Regular Network")
    axes[1][1].plot(valid_losses[:,1], 'b', label="Transfer Network")
    axes[1][1].plot(valid_losses[:,2], 'r', label="AE Transfer Network")
    axes[1][1].legend()
    plt.show()
    print("Finished plotting.")
    return

def shuffle_divide_batches(inputs, n_batches):
    n_layouts = np.shape(inputs)[0]
    perm = np.arange(n_layouts)
    np.random.shuffle(perm)
    inputs_batches = np.split(inputs[perm], n_batches, axis=0)
    return inputs_batches

EARLY_STOPPING = True

if(__name__=="__main__"):
    parser = argparse.ArgumentParser(description="main script argument parser")
    parser.add_argument("--plot", default=False, help="Whether plot the training curves (executed after training)")
    args = parser.parse_args()
    if args.plot:
        plot_training_curves()
        exit(0)

    print("Loading MNIST source data...")
    source_data = MNIST(root='MNIST_Data/', train=True, download=True, 
              transform=transforms.compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                                            transforms.Lambda(lambda x: x.flatten())], 
              target_transform=transforms.Lambda(lambda y: int(y==SOURCETASK['Task']))))

    """ 
    Source-Task Training 
    """
    print(f"<<<<<<<<<<<<<<<<<<<<<<<Learn to identify [{SOURCETASK['Task']}]->[{TARGETTASK['Task']}]>>>>>>>>>>>>>>>>>>>>>>")
    regular_net, transfer_net, ae_transfer_net = \
            Regular_Net().to(DEVICE), Transfer_Net().to(DEVICE), Autoencoder_Transfer_Net().to(DEVICE)
    # The split should be reproduciable with torch.manual_seed set in setup.py                                
    train_data, valid_data = random_split(source_data, [SOURCETASK['Train'], SOURCETASK['Valid']])
    train_loader = DataLoader(train_data, batch_size = SOURCETASK['Minibatch_Size'], shuffle=True)    
    valid_loader = DataLoader(valid_data, batch_size = len(valid_data), shuffle=False)
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
        for j, data, targets in enumerate(train_loader):
            optimizer_regular.zero_grad()
            optimizer_transfer.zero_grad()
            optimizer_ae_transfer.zero_grad()
            # Regular Net
            predictions = regular_net.sourcetask(data.to(DEVICE))
            regular_loss = -objAvg
            # Transfer Net
            predictions = transfer_net.sourcetask(data.to(DEVICE))
            transfer_loss = -objAvg
            # AutoEncoder Transfer Net
            _, objAvg, reconstruct_loss = ae_transfer_net.sourceTask_powerControl(torch.tensor(g_batches[j], dtype=torch.float32).to(DEVICE))
            ae_transfer_loss = -objAvg
            ae_transfer_loss_combined = ae_transfer_loss + COMBINE_WEIGHT_RECONSTRUCT * reconstruct_loss
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
                    ae_transfer_loss_combined = ae_transfer_loss + COMBINE_WEIGHT_RECONSTRUCT * reconstruct_loss.item()
                train_loss_eps.append([regular_loss_ep/(j+1), transfer_loss_ep/(j+1), ae_transfer_loss_ep/(j+1), ae_transfer_loss_combined_ep/(j+1)])
                valid_loss_eps.append([regular_loss, transfer_loss, ae_transfer_loss, ae_transfer_loss_combined])
                print("[Source Task][Regular] Tr:{:6.3e}; Va:{:6.3e} [Transfer] Tr: {:6.3e}; Va:{:6.3e} [AE Transfer] Tr: {:6.3e}; Va: {:6.3e}".format(
                    regular_loss_ep/(j+1), regular_loss, transfer_loss_ep/(j+1), transfer_loss, ae_transfer_loss_ep/(j+1), ae_transfer_loss))
                if EARLY_STOPPING:
                    # Early stopping based on validation losses
                    if (regular_loss < regular_loss_min):
                        regular_net.save_model()
                        regular_loss_min = regular_loss
                    if (transfer_loss < transfer_loss_min):
                        transfer_net.save_model()
                        transfer_loss_min = transfer_loss
                    if (ae_transfer_loss_combined < ae_transfer_loss_combined_min):
                        ae_transfer_net.save_model()
                        ae_transfer_loss_combined_min = ae_transfer_loss_combined    
                else:
                    regular_net.save_model()
                    transfer_net.save_model()
                    ae_transfer_net.save_model()
                np.save(f"Trained_Models/{SOURCETASK['Task']}-{TARGETTASK['Task']}/train_losses_sourceTask_{SETTING_STRING}.npy", np.array(train_loss_eps))
                np.save(f"Trained_Models/{SOURCETASK['Task']}-{TARGETTASK['Task']}/valid_losses_sourceTask_{SETTING_STRING}.npy", np.array(valid_loss_eps))

    """ 
    Target Task Training
    """
    MINIBATCH_SIZE = 50
    print("[Target Task] Loading data...")
    g = np.load(f"Data/g_targetTask_{SETTING_STRING}.npy")
    assert np.shape(g)[0] == TARGETTASK['Train'] + TARGETTASK['Valid']
    g_train, g_valid = g[:TARGETTASK['Train']], g[-TARGETTASK['Valid']:]
    assert TARGETTASK['Train'] % MINIBATCH_SIZE == 0
    n_minibatches = int(TARGETTASK['Train'] / MINIBATCH_SIZE)
    print(f"[Target Task] Data Loaded! With {TARGETTASK['Train']} training samples ({n_minibatches} minibatches) and {TARGETTASK['Valid']} validation samples.")

    # Create neural network objects again so they load weights from previous early stopping best checkpoint on source task
    regular_net, transfer_net, ae_transfer_net = \
            Regular_Net().to(DEVICE), Transfer_Net().to(DEVICE), Autoencoder_Transfer_Net().to(DEVICE)
    print("[Target Task] Freeze the neural network parameters up to the feature layers...")
    transfer_net.freeze_parameters()
    ae_transfer_net.freeze_parameters()
    optimizer_regular, optimizer_transfer, optimizer_ae_transfer = \
            optim.Adam(filter(lambda para: para.requires_grad, regular_net.parameters()), lr=LEARNING_RATE_TARGETTASK), \
            optim.Adam(filter(lambda para: para.requires_grad, transfer_net.parameters()), lr=LEARNING_RATE_TARGETTASK), \
            optim.Adam(filter(lambda para: para.requires_grad, ae_transfer_net.parameters()), lr=LEARNING_RATE_TARGETTASK)
    regular_loss_min, transfer_loss_min, ae_transfer_loss_min = np.inf, np.inf, np.inf
    train_loss_eps, valid_loss_eps = [], []
    for i in trange(1, N_EPOCHES_TARGETTASK+1):
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
        if EARLY_STOPPING:
            # Early stopping based on validation losses
            if (regular_loss < regular_loss_min):
                regular_net.save_model()
                regular_loss_min = regular_loss
            if (transfer_loss < transfer_loss_min):
                transfer_net.save_model()
                transfer_loss_min = transfer_loss
            if (ae_transfer_loss < ae_transfer_loss_min):
                ae_transfer_net.save_model()
                ae_transfer_loss_min = ae_transfer_loss    
        else:
            regular_net.save_model()
            transfer_net.save_model()
            ae_transfer_net.save_model()
        np.save(f"Trained_Models/{SOURCETASK['Task']}-{TARGETTASK['Task']}/train_losses_targetTask_{SETTING_STRING}.npy", np.array(train_loss_eps))
        np.save(f"Trained_Models/{SOURCETASK['Task']}-{TARGETTASK['Task']}/valid_losses_targetTask_{SETTING_STRING}.npy", np.array(valid_loss_eps))

    print(f"[{SOURCETASK['Fullname']}]->[{TARGETTASK['Fullname']}] Training finished!")
