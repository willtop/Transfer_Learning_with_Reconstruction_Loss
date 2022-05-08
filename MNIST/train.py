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
    fig, axes = plt.subplots(2,3)
    fig.suptitle(f"Training Curves for {TASK_DESCR}")
    # Plot for source task
    train_losses = np.load(f"Trained_Models/{TASK_DESCR}/train_losses_sourcetask.npy")
    valid_losses = np.load(f"Trained_Models/{TASK_DESCR}/valid_losses_sourcetask.npy")
    valid_accuracies = np.load(f"Trained_Models/{TASK_DESCR}/valid_accuracies_sourcetask.npy")
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


EARLY_STOPPING = True
LOSS_FUNC = torch.nn.BCELoss(reduction='mean')

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

    print(f"<<<<<<<<<<<<<<<<<<<<<<<Learn for {TASK_DESCR}>>>>>>>>>>>>>>>>>>>>>>")
    """ 
    Source-Task Training 
    """
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
    train_loss_eps, valid_loss_eps, valid_accuracies_eps = [], [], []
    for i in trange(1, SOURCETASK['Epochs']+1):
        regular_loss_ep, transfer_loss_ep, ae_transfer_loss_ep, ae_transfer_loss_combined_ep = 0, 0, 0, 0
        for j, data, targets in enumerate(train_loader):
            assert data.size() == (SOURCETASK['Minibatch_Size'], 28*28) and \
                   targets.size() == (SOURCETASK['Minibatch_Size'],)
            optimizer_regular.zero_grad()
            optimizer_transfer.zero_grad()
            optimizer_ae_transfer.zero_grad()
            # Regular Net
            predictions = regular_net.sourcetask(data.to(DEVICE))
            regular_loss = LOSS_FUNC(torch.squeeze(predictions), targets.to(DEVICE))
            # Transfer Net
            predictions = transfer_net.sourcetask(data.to(DEVICE))
            transfer_loss = LOSS_FUNC(torch.squeeze(predictions), targets.to(DEVICE))
            # AutoEncoder Transfer Net
            predictions, reconstruct_loss = ae_transfer_net.sourcetask(data.to(DEVICE))
            ae_transfer_loss = LOSS_FUNC(torch.squeeze(predictions), targets.to(DEVICE))
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
                for data, targets in valid_loader: # only load up one batch
                    assert data.size() == (SOURCETASK['Valid'], 28*28) and \
                        targets.size() == (SOURCETASK['Valid'],)
                    with torch.no_grad():
                        predictions = torch.squeeze(regular_net.sourcetask(data.to(DEVICE)))
                        regular_loss = LOSS_FUNC(predictions, targets.to(DEVICE)).item()
                        regular_accuracy = np.mean(predictions.round().detach().numpy() == targets.numpy())
                        predictions = torch.squeeze(transfer_net.sourcetask(data.to(DEVICE)))
                        transfer_loss = LOSS_FUNC(predictions, targets.to(DEVICE)).item()
                        transfer_accuracy = np.mean(predictions.round().detach().numpy() == targets.numpy())
                        predictions, reconstruct_loss = ae_transfer_net.sourcetask(data.to(DEVICE))
                        predictions = torch.squeeze(predictions)
                        ae_transfer_loss = LOSS_FUNC(predictions, targets.to(DEVICE)).item()
                        ae_transfer_loss_combined = ae_transfer_loss + SOURCETASK['Loss_Combine_Weight'] * reconstruct_loss.item()
                        ae_transfer_accuracy = np.mean(predictions.round().detach().numpy() == targets.numpy())
                train_loss_eps.append([regular_loss_ep/(j+1), transfer_loss_ep/(j+1), ae_transfer_loss_ep/(j+1), ae_transfer_loss_combined_ep/(j+1)])
                valid_loss_eps.append([regular_loss, transfer_loss, ae_transfer_loss, ae_transfer_loss_combined])
                valid_accuracies_eps.append([regular_accuracy, transfer_accuracy, ae_transfer_accuracy])
                print("[Source Task][Regular] Tr:{:6.3e}; Va:{:6.3e}; Va Accur: {:.2f}% \
                       [Transfer] Tr: {:6.3e}; Va:{:6.3e}; Va Accur: {:.2f}% \
                       [AE Transfer] Tr: {:6.3e}; Va: {:6.3e}; Va Accur: {:.2f}%".format(
                       regular_loss_ep/(j+1), regular_loss, regular_accuracy*100,
                       transfer_loss_ep/(j+1), transfer_loss, transfer_accuracy*100,
                       ae_transfer_loss_ep/(j+1), ae_transfer_loss, ae_transfer_accuracy*100))
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
                np.save(f"Trained_Models/{TASK_DESCR}/train_losses_sourcetask.npy", np.array(train_loss_eps))
                np.save(f"Trained_Models/{TASK_DESCR}/valid_losses_sourcetask.npy", np.array(valid_loss_eps))
                np.save(f"Trained_Models/{TASK_DESCR}/valid_accuracies_sourcetask.npy", np.array(valid_accuracies_eps))

    """ 
    Target-Task Training 
    """
    # The splits should be reproduciable with torch.manual_seed set in setup.py                                
    target_task_data = random_split(source_data, [TARGETTASK['Train']+TARGETTASK['Valid'], len(source_data)-(TARGETTASK['Train']+TARGETTASK['Valid'])])
    train_data, valid_data = random_split(target_task_data, [TARGETTASK['Train'], TARGETTASK['Valid']])
    train_loader = DataLoader(train_data, batch_size = TARGETTASK['Minibatch_Size'], shuffle=True)    
    valid_loader = DataLoader(valid_data, batch_size = len(valid_data), shuffle=False)
    n_minibatches = int(TARGETTASK['Train'] / TARGETTASK['Minibatch_Size'])
    print(f"[Target Task on {TARGETTASK['Task']}] Data Loaded! With {TARGETTASK['Train']} training samples ({n_minibatches} minibatches) and {TARGETTASK['Valid']} validation samples.")

    # Create neural network objects again so they load weights from previous early-stopping best checkpoint on source task
    regular_net, transfer_net, ae_transfer_net = \
            Regular_Net().to(DEVICE), Transfer_Net().to(DEVICE), Autoencoder_Transfer_Net().to(DEVICE)
    print("[Target Task] Freeze the neural network parameters up to the feature layer...")
    transfer_net.freeze_parameters()
    ae_transfer_net.freeze_parameters()
    optimizer_regular, optimizer_transfer, optimizer_ae_transfer = \
            optim.Adam(filter(lambda para: para.requires_grad, regular_net.parameters()), lr=TARGETTASK['Learning_Rate']), \
            optim.Adam(filter(lambda para: para.requires_grad, transfer_net.parameters()), lr=TARGETTASK['Learning_Rate']), \
            optim.Adam(filter(lambda para: para.requires_grad, ae_transfer_net.parameters()), lr=TARGETTASK['Learning_Rate'])
    regular_loss_min, transfer_loss_min, ae_transfer_loss_min = np.inf, np.inf, np.inf
    train_loss_eps, valid_loss_eps, valid_accuracies_eps = [], [], []
    for i in trange(1, TARGETTASK['Epochs']+1):
        regular_loss_ep, transfer_loss_ep, ae_transfer_loss_ep = 0, 0, 0
        for j, data, targets in enumerate(train_loader):
            assert data.size() == (TARGETTASK['Minibatch_Size'], 28*28) and \
                   targets.size() == (TARGETTASK['Minibatch_Size'],)
            optimizer_regular.zero_grad()
            optimizer_transfer.zero_grad()
            optimizer_ae_transfer.zero_grad()
            # Regular Net
            predictions = regular_net.targettask(data.to(DEVICE))
            regular_loss = LOSS_FUNC(torch.squeeze(predictions), targets.to(DEVICE))
            # Transfer Net
            predictions = transfer_net.targettask(data.to(DEVICE))
            transfer_loss = LOSS_FUNC(torch.squeeze(predictions), targets.to(DEVICE))
            # AutoEncoder Transfer Net
            predictions = ae_transfer_net.targettask(data.to(DEVICE))
            ae_transfer_loss = LOSS_FUNC(torch.squeeze(predictions), targets.to(DEVICE))
            # Training and recording loss
            regular_loss.backward(); optimizer_regular.step()
            transfer_loss.backward(); optimizer_transfer.step()
            ae_transfer_loss.backward(); optimizer_ae_transfer.step()           
            regular_loss_ep += regular_loss.item()
            transfer_loss_ep += transfer_loss.item()
            ae_transfer_loss_ep += ae_transfer_loss.item()
            if (j+1) % min(50,n_minibatches) == 0:
                # Validation
                for data, targets in valid_loader: # only load up one batch
                    assert data.size() == (TARGETTASK['Valid'], 28*28) and \
                        targets.size() == (TARGETTASK['Valid'],)
                    with torch.no_grad():
                        predictions = torch.squeeze(regular_net.targettask(data.to(DEVICE)))
                        regular_loss = LOSS_FUNC(predictions, targets.to(DEVICE)).item()
                        regular_accuracy = np.mean(predictions.round().detach().numpy() == targets.numpy())
                        predictions = torch.squeeze(transfer_net.targettask(data.to(DEVICE)))
                        transfer_loss = LOSS_FUNC(predictions, targets.to(DEVICE)).item()
                        transfer_accuracy = np.mean(predictions.round().detach().numpy() == targets.numpy())
                        predictions = torch.squeeze(ae_transfer_net.targettask(data.to(DEVICE)))
                        ae_transfer_loss = LOSS_FUNC(predictions, targets.to(DEVICE)).item()
                        ae_transfer_accuracy = np.mean(predictions.round().detach().numpy() == targets.numpy())
                train_loss_eps.append([regular_loss_ep/(j+1), transfer_loss_ep/(j+1), ae_transfer_loss_ep/(j+1)])
                valid_loss_eps.append([regular_loss, transfer_loss, ae_transfer_loss])
                valid_accuracies_eps.append([regular_accuracy, transfer_accuracy, ae_transfer_accuracy])
                print("[Target Task][Regular] Tr:{:6.3e}; Va:{:6.3e}; Va Accur: {:.2f}% \
                       [Transfer] Tr: {:6.3e}; Va:{:6.3e}; Va Accur: {:.2f}% \
                       [AE Transfer] Tr: {:6.3e}; Va: {:6.3e}; Va Accur: {:.2f}%".format(
                       regular_loss_ep/(j+1), regular_loss, regular_accuracy*100,
                       transfer_loss_ep/(j+1), transfer_loss, transfer_accuracy*100,
                       ae_transfer_loss_ep/(j+1), ae_transfer_loss, ae_transfer_accuracy*100))
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
                        ae_transfer_loss_combined_min = ae_transfer_loss_combined    
                else:
                    regular_net.save_model()
                    transfer_net.save_model()
                    ae_transfer_net.save_model()
                np.save(f"Trained_Models/{TASK_DESCR}/train_losses_targettask.npy", np.array(train_loss_eps))
                np.save(f"Trained_Models/{TASK_DESCR}/valid_losses_targettask.npy", np.array(valid_loss_eps))
                np.save(f"Trained_Models/{TASK_DESCR}/valid_accuracies_targettask.npy", np.array(valid_accuracies_eps))

    print(f"[{TASK_DESCR}] Training finished!")
