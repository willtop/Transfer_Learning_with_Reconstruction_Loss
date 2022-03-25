# Training script for all the models

from random import gammavariate
import numpy as np
from benchmarks import FP_power_control, GP_power_control
import torch
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
    axes = axes.flatten()
    fig.suptitle(f"D2D Loss (Unscaled) over {SETTING_STRING}")
    # Plot for sum rates
    train_losses = np.load(f"Trained_Models/train_losses_Sum-Rate_{SETTING_STRING}.npy")
    valid_losses = np.load(f"Trained_Models/valid_losses_Sum-Rate_{SETTING_STRING}.npy")
    axes[0].set_xlabel("Epoches")
    axes[0].set_ylabel("Training Losses (Sum Rate)")
    axes[0].plot(train_losses[:,0], 'g', label="Regular Network")
    axes[0].plot(train_losses[:,1], 'b', label="Transfer Network")
    axes[0].plot(train_losses[:,2], 'r', label="AE Transfer Network")
    axes[0].plot(train_losses[:,3], 'r--', label="AE Transfer Network Combined")
    axes[0].legend()
    axes[1].set_xlabel("Epoches")
    axes[1].set_ylabel("Validation Losses (Sum Rate)")
    axes[1].plot(valid_losses[:,0], 'g', label="Regular Network")
    axes[1].plot(valid_losses[:,1], 'b', label="Transfer Network")
    axes[1].plot(valid_losses[:,2], 'r', label="AE Transfer Network")
    axes[1].plot(valid_losses[:,3], 'r--', label="AE Transfer Network Combined")
    axes[1].legend()
    # Plot for min rates
    train_losses = np.load(f"Trained_Models/train_losses_Min-Rate_{SETTING_STRING}.npy")
    valid_losses = np.load(f"Trained_Models/valid_losses_Min-Rate_{SETTING_STRING}.npy")
    axes[2].set_xlabel("Epoches")
    axes[2].set_ylabel("Training Losses (Min Rate)")
    axes[2].plot(train_losses[:,0], 'g', label="Regular Network")
    axes[2].plot(train_losses[:,1], 'b', label="Transfer Network")
    axes[2].plot(train_losses[:,2], 'r', label="AE Transfer Network")
    axes[2].legend()
    axes[3].set_xlabel("Epoches")
    axes[3].set_ylabel("Validation Losses (Min Rate)")
    axes[3].plot(valid_losses[:,0], 'g', label="Regular Network")
    axes[3].plot(valid_losses[:,1], 'b', label="Transfer Network")
    axes[3].plot(valid_losses[:,2], 'r', label="AE Transfer Network")
    axes[3].legend()
    plt.show()
    print("Finished plotting.")
    return

def shuffle_divide_batches(inputs, n_batches):
    n_layouts = np.shape(inputs)[0]
    perm = np.arange(n_layouts)
    np.random.shuffle(perm)
    inputs_batches = np.split(inputs[perm], n_batches, axis=0)
    return inputs_batches

EARLY_STOPPING = False
LEARNING_RATE = 1e-4

if(__name__=="__main__"):
    parser = argparse.ArgumentParser(description="main script argument parser")
    parser.add_argument("--plot", default=False, help="Whether plot the training curves (executed after training)")
    args = parser.parse_args()
    if args.plot:
        plot_training_curves()
        exit(0)

    regular_net, transfer_net, ae_transfer_net = \
            Regular_Net().to(DEVICE), Transfer_Net().to(DEVICE), Autoencoder_Transfer_Net().to(DEVICE)

    """ 
    Sum-Rate Training
    """
    N_EPOCHES = 20
    MINIBATCH_SIZE = 2000
    print("[D2D SumRate] Loading data...")
    g_sumRate = np.load(f"Data/g_sumRate_{SETTING_STRING}.npy")
    g_sumRate_train, g_sumRate_valid = g_sumRate[:N_SAMPLES['SumRate']['Train']], g_sumRate[-N_SAMPLES['SumRate']['Valid']:]
    n_train = np.shape(g_sumRate_train)[0]
    assert n_train % MINIBATCH_SIZE == 0
    n_minibatches = int(n_train / MINIBATCH_SIZE)
    print("[D2D SumRate] Data Loaded! With {} training samples ({} minibatches) and {} validation samples.".format(n_train, n_minibatches, np.shape(g_sumRate_valid)[0]))

    optimizer_regular, optimizer_transfer, optimizer_ae_transfer = \
            optim.Adam(regular_net.parameters(), lr=LEARNING_RATE), optim.Adam(transfer_net.parameters(), lr=LEARNING_RATE), optim.Adam(ae_transfer_net.parameters(), lr=LEARNING_RATE)
    regular_loss_min, transfer_loss_min, ae_transfer_loss_combined_min = np.inf, np.inf, np.inf
    train_loss_eps, valid_loss_eps = [], []
    for i in trange(1, N_EPOCHES+1):
        regular_loss_ep, transfer_loss_ep, ae_transfer_loss_ep, ae_transfer_loss_combined_ep = 0, 0, 0, 0
        g_batches = shuffle_divide_batches(g_sumRate_train, n_minibatches)
        for j in range(n_minibatches):
            optimizer_regular.zero_grad()
            optimizer_transfer.zero_grad()
            optimizer_ae_transfer.zero_grad()
            # Regular Net
            _, sumRateAvg = regular_net.sumRate_power_control(torch.tensor(g_batches[j], dtype=torch.float32).to(DEVICE))
            regular_loss = -sumRateAvg
            # Transfer Net
            _, sumRateAvg = transfer_net.sumRate_power_control(torch.tensor(g_batches[j], dtype=torch.float32).to(DEVICE))
            transfer_loss = -sumRateAvg
            # AutoEncoder Transfer Net
            _, sumRateAvg, reconstruct_loss = ae_transfer_net.sumRate_power_control(torch.tensor(g_batches[j], dtype=torch.float32).to(DEVICE))
            ae_transfer_loss = -sumRateAvg
            ae_transfer_loss_combined = ae_transfer_loss + reconstruct_loss
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
                    _, sumRateAvg = regular_net.sumRate_power_control(torch.tensor(g_sumRate_valid, dtype=torch.float32).to(DEVICE))
                    regular_loss = -sumRateAvg.item()
                    _, sumRateAvg = transfer_net.sumRate_power_control(torch.tensor(g_sumRate_valid, dtype=torch.float32).to(DEVICE))
                    transfer_loss = -sumRateAvg.item()
                    _, sumRateAvg, reconstruct_loss = ae_transfer_net.sumRate_power_control(torch.tensor(g_sumRate_valid, dtype=torch.float32).to(DEVICE))
                    ae_transfer_loss = -sumRateAvg.item()
                    ae_transfer_loss_combined = ae_transfer_loss + reconstruct_loss.item()
                train_loss_eps.append([regular_loss_ep/(j+1), transfer_loss_ep/(j+1), ae_transfer_loss_ep/(j+1), ae_transfer_loss_combined_ep/(j+1)])
                valid_loss_eps.append([regular_loss, transfer_loss, ae_transfer_loss, ae_transfer_loss_combined])
                print("[D2D Sum-Rate][Regular] Tr:{:6.3e}; Va:{:6.3e} [Transfer] Tr: {:6.3e}; Va:{:6.3e} [AE Transfer] Tr: {:6.3e}; Va: {:6.3e}".format(
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
                np.save(f"Trained_Models/train_losses_Sum-Rate_{SETTING_STRING}.npy", np.array(train_loss_eps))
                np.save(f"Trained_Models/valid_losses_Sum-Rate_{SETTING_STRING}.npy", np.array(valid_loss_eps))

    """ 
    Min-Rate Training
    """
    N_EPOCHES = 200
    MINIBATCH_SIZE = 1000
    print("[D2D MinRate] Loading data...")
    g_minRate = np.load(f"Data/g_minRate_{SETTING_STRING}.npy")
    g_minRate_train, g_minRate_valid = g_sumRate[:N_SAMPLES['MinRate']['Train']], g_sumRate[N_SAMPLES['MinRate']['Train']:N_SAMPLES['MinRate']['Train']+N_SAMPLES['MinRate']['Valid']]
    n_train = np.shape(g_minRate_train)[0]
    assert n_train % MINIBATCH_SIZE == 0
    n_minibatches = int(n_train / MINIBATCH_SIZE)
    print("[D2D MinRate] Data Loaded! With {} training samples ({} minibatches) and {} validation samples.".format(n_train, n_minibatches, np.shape(g_minRate_valid)[0]))

    print("[D2D MinRate] Freeze the neural network parameters up to the feature layers...")
    transfer_net.freeze_parameters()
    ae_transfer_net.freeze_parameters()
    optimizer_regular, optimizer_transfer, optimizer_ae_transfer = \
            optim.Adam(filter(lambda para: para.requires_grad, regular_net.parameters()), lr=LEARNING_RATE), \
            optim.Adam(filter(lambda para: para.requires_grad, transfer_net.parameters()), lr=LEARNING_RATE), \
            optim.Adam(filter(lambda para: para.requires_grad, ae_transfer_net.parameters()), lr=LEARNING_RATE)
    regular_loss_min, transfer_loss_min, ae_transfer_loss_min = np.inf, np.inf, np.inf
    train_loss_eps, valid_loss_eps = [], []
    for i in trange(1, N_EPOCHES+1):
        regular_loss_ep, transfer_loss_ep, ae_transfer_loss_ep = 0, 0, 0
        g_batches = shuffle_divide_batches(g_minRate_train, n_minibatches)
        for j in range(n_minibatches):
            optimizer_regular.zero_grad()
            optimizer_transfer.zero_grad()
            optimizer_ae_transfer.zero_grad()
            # Regular Net
            _, minRateAvg = regular_net.minRate_power_control(torch.tensor(g_batches[j], dtype=torch.float32).to(DEVICE))
            regular_loss = -minRateAvg
            # Transfer Net
            _, minRateAvg = transfer_net.minRate_power_control(torch.tensor(g_batches[j], dtype=torch.float32).to(DEVICE))
            transfer_loss = -minRateAvg
            # AutoEncoder Transfer Net
            _, minRateAvg = ae_transfer_net.minRate_power_control(torch.tensor(g_batches[j], dtype=torch.float32).to(DEVICE))
            ae_transfer_loss = -minRateAvg
            # Training and recording loss
            regular_loss.backward(); optimizer_regular.step()
            transfer_loss.backward(); optimizer_transfer.step()
            ae_transfer_loss.backward(); optimizer_ae_transfer.step()           
            regular_loss_ep += regular_loss.item()
            transfer_loss_ep += transfer_loss.item()
            ae_transfer_loss_ep += ae_transfer_loss.item()
        # Validation
        with torch.no_grad():
            _, minRateAvg = regular_net.minRate_power_control(torch.tensor(g_minRate_valid, dtype=torch.float32).to(DEVICE))
            regular_loss = -minRateAvg.item()
            _, minRateAvg = transfer_net.minRate_power_control(torch.tensor(g_minRate_valid, dtype=torch.float32).to(DEVICE))
            transfer_loss = -minRateAvg.item()
            _, minRateAvg = ae_transfer_net.minRate_power_control(torch.tensor(g_minRate_valid, dtype=torch.float32).to(DEVICE))
            ae_transfer_loss = -minRateAvg.item()
        train_loss_eps.append([regular_loss_ep/(j+1), transfer_loss_ep/(j+1), ae_transfer_loss_ep/(j+1)])
        valid_loss_eps.append([regular_loss, transfer_loss, ae_transfer_loss])
        print("[D2D Min-Rate][Regular] Tr:{:6.3e}; Va:{:6.3e} [Transfer] Tr: {:6.3e}; Va:{:6.3e} [AE Transfer] Tr: {:6.3e}; Va: {:6.3e}".format(
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
        np.save(f"Trained_Models/train_losses_Min-Rate_{SETTING_STRING}.npy", np.array(train_loss_eps))
        np.save(f"Trained_Models/valid_losses_Min-Rate_{SETTING_STRING}.npy", np.array(valid_loss_eps))


    print(f"Training finished!")
