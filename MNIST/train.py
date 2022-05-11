# Training script for all the models

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
from tqdm import trange
import argparse
import matplotlib.pyplot as plt
from setup import *
from neural_nets import Regular_Net, Transfer_Net, Autoencoder_Transfer_Net
import utils


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
    axes[0][2].set_xlabel("Epoches")
    axes[0][2].set_ylabel("Validation Accuracies (Source Task)")
    axes[0][2].plot(valid_accuracies[:,0], 'g', label="Regular Network")
    axes[0][2].plot(valid_accuracies[:,1], 'b', label="Transfer Network")
    axes[0][2].plot(valid_accuracies[:,2], 'r', label="AE Transfer Network")
    axes[0][2].legend()
    # Plot for target task
    train_losses = np.load(f"Trained_Models/{TASK_DESCR}/train_losses_targettask.npy")
    valid_losses = np.load(f"Trained_Models/{TASK_DESCR}/valid_losses_targettask.npy")
    valid_accuracies = np.load(f"Trained_Models/{TASK_DESCR}/valid_accuracies_targettask.npy")
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
    axes[1][2].set_xlabel("Epoches")
    axes[1][2].set_ylabel("Validation Accuracies (Target Task)")
    axes[1][2].plot(valid_accuracies[:,0], 'g', label="Regular Network")
    axes[1][2].plot(valid_accuracies[:,1], 'b', label="Transfer Network")
    axes[1][2].plot(valid_accuracies[:,2], 'r', label="AE Transfer Network")
    axes[1][2].legend()
    plt.show()
    print("Finished plotting!")
    return

# Since CE loss is not tightly correlated with the accuracy. Use validation accuracy
# for early stopping criterion (except for AE transfer learning training loss)
EARLY_STOPPING = True
LOSS_FUNC = torch.nn.CrossEntropyLoss(reduction='mean')

if(__name__=="__main__"):
    parser = argparse.ArgumentParser(description="main script argument parser")
    parser.add_argument("--plot", default=False, help="Whether plot the training curves (executed after training)")
    args = parser.parse_args()
    if args.plot:
        plot_training_curves()
        exit(0)

    # Load data (Don't transform targets here yet)
    # All the splits should be reproduciable with torch.manual_seed set in setup.py                                
    if APPLICATION == 'MNIST':
        original_data = MNIST(root='Data/', train=True, download=True, 
            transform=transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize(size=(IMAGE_LENGTH, IMAGE_LENGTH)),
                                          transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                                          transforms.Lambda(lambda x: x.flatten())]))
    else:
        original_data = FashionMNIST(root='Data/', train=True, download=True, 
            transform=transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize(size=(IMAGE_LENGTH, IMAGE_LENGTH)),
                                          transforms.Lambda(lambda x: x.flatten())]))
    assert len(original_data) == 60000
    sourcetask_data, targettask_data = random_split(original_data, [SOURCETASK['Train']+SOURCETASK['Valid'], TARGETTASK['Train']+TARGETTASK['Valid']])

    print(f"<<<<<<<<<<<<<<<<<<<<<<<Learn for {TASK_DESCR}>>>>>>>>>>>>>>>>>>>>>>")
    """ 
    Source-Task Training 
    """
    print(f"Loading {APPLICATION} source data...")
    regular_net, transfer_net, ae_transfer_net = \
            Regular_Net().to(DEVICE), Transfer_Net().to(DEVICE), Autoencoder_Transfer_Net().to(DEVICE)
    train_data, valid_data = random_split(sourcetask_data, [SOURCETASK['Train'], SOURCETASK['Valid']])
    train_loader = DataLoader(train_data, batch_size = SOURCETASK['Minibatch_Size'], shuffle=True, drop_last=True)    
    valid_loader = DataLoader(valid_data, batch_size = len(valid_data), shuffle=False)
    n_minibatches = int(SOURCETASK['Train'] / SOURCETASK['Minibatch_Size'])
    print(f"[Source Task on {SOURCETASK['Task']}] Data Loaded! With {SOURCETASK['Train']} training samples ({n_minibatches} minibatches) and {SOURCETASK['Valid']} validation samples.")

    optimizer_regular, optimizer_transfer, optimizer_ae_transfer = \
            optim.Adam(regular_net.parameters(), lr=SOURCETASK['Learning_Rate']), \
            optim.Adam(transfer_net.parameters(), lr=SOURCETASK['Learning_Rate']), \
            optim.Adam(ae_transfer_net.parameters(), lr=SOURCETASK['Learning_Rate'])
    regular_best_point, transfer_best_point, ae_transfer_best_point = 0, 0, np.inf
    train_loss_eps, valid_loss_eps, valid_accuracies_eps = [], [], []
    for i in trange(1, SOURCETASK['Epochs']+1):
        regular_loss_ep, transfer_loss_ep, ae_transfer_loss_ep, ae_transfer_loss_combined_ep = 0, 0, 0, 0
        for j, (data, targets) in enumerate(train_loader):
            assert data.size() == (SOURCETASK['Minibatch_Size'], INPUT_SIZE) and \
                   targets.size() == (SOURCETASK['Minibatch_Size'], ) 
            targets = utils.convert_targets(targets, SOURCETASK)
            optimizer_regular.zero_grad()
            optimizer_transfer.zero_grad()
            optimizer_ae_transfer.zero_grad()
            # Regular Net
            predictions = regular_net.sourcetask(data.to(DEVICE))
            regular_loss = LOSS_FUNC(predictions, targets.to(DEVICE))
            # Transfer Net
            predictions = transfer_net.sourcetask(data.to(DEVICE))
            transfer_loss = LOSS_FUNC(predictions, targets.to(DEVICE))
            # AutoEncoder Transfer Net
            predictions, reconstruct_loss = ae_transfer_net.sourcetask(data.to(DEVICE))
            ae_transfer_loss = LOSS_FUNC(predictions, targets.to(DEVICE))
            ae_transfer_loss_combined = ae_transfer_loss + SOURCETASK['Loss_Combine_Weight'] * reconstruct_loss
            # Training and recording loss
            regular_loss.backward(); optimizer_regular.step()
            transfer_loss.backward(); optimizer_transfer.step()
            ae_transfer_loss_combined.backward(); optimizer_ae_transfer.step()           
            regular_loss_ep += regular_loss.item()
            transfer_loss_ep += transfer_loss.item()
            ae_transfer_loss_ep += ae_transfer_loss.item()
            ae_transfer_loss_combined_ep += ae_transfer_loss_combined.item()
            if (j+1) % 50 == 0 or (j+1) % n_minibatches == 0:
                # Validation
                for data, targets in valid_loader: # only load up one batch
                    assert data.size() == (SOURCETASK['Valid'], INPUT_SIZE) and \
                        targets.size() == (SOURCETASK['Valid'], )
                    targets = utils.convert_targets(targets, SOURCETASK)
                    with torch.no_grad():
                        predictions = regular_net.sourcetask(data.to(DEVICE))
                        regular_loss = LOSS_FUNC(predictions, targets.to(DEVICE)).item()
                        regular_accuracy = utils.compute_accuracy(predictions.detach().cpu(), targets)
                        predictions = transfer_net.sourcetask(data.to(DEVICE))
                        transfer_loss = LOSS_FUNC(predictions, targets.to(DEVICE)).item()
                        transfer_accuracy = utils.compute_accuracy(predictions.detach().cpu(), targets)
                        predictions, reconstruct_loss = ae_transfer_net.sourcetask(data.to(DEVICE))
                        ae_transfer_loss = LOSS_FUNC(predictions, targets.to(DEVICE)).item()
                        ae_transfer_loss_combined = ae_transfer_loss + SOURCETASK['Loss_Combine_Weight'] * reconstruct_loss.item()
                        ae_transfer_accuracy = utils.compute_accuracy(predictions.detach().cpu(), targets)
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
                    if (regular_accuracy > regular_best_point):
                        regular_net.save_model()
                        regular_best_point = regular_accuracy
                    if (transfer_accuracy > transfer_best_point):
                        transfer_net.save_model()
                        transfer_best_point = transfer_accuracy
                    if (ae_transfer_loss_combined < ae_transfer_best_point):
                        ae_transfer_net.save_model()
                        ae_transfer_best_point = ae_transfer_loss_combined    
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
    train_data, valid_data = random_split(targettask_data, [TARGETTASK['Train'], TARGETTASK['Valid']])
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
    regular_best_point, transfer_best_point, ae_transfer_best_point = 0, 0, 0
    train_loss_eps, valid_loss_eps, valid_accuracies_eps = [], [], []
    for i in trange(1, TARGETTASK['Epochs']+1):
        regular_loss_ep, transfer_loss_ep, ae_transfer_loss_ep = 0, 0, 0
        for j, (data, targets) in enumerate(train_loader):
            assert data.size() == (TARGETTASK['Minibatch_Size'], INPUT_SIZE) and \
                   targets.size() == (TARGETTASK['Minibatch_Size'], )
            targets = utils.convert_targets(targets, TARGETTASK)                    
            optimizer_regular.zero_grad()
            optimizer_transfer.zero_grad()
            optimizer_ae_transfer.zero_grad()
            # Regular Net
            predictions = regular_net.targettask(data.to(DEVICE))
            regular_loss = LOSS_FUNC(predictions, targets.to(DEVICE))
            # Transfer Net
            predictions = transfer_net.targettask(data.to(DEVICE))
            transfer_loss = LOSS_FUNC(predictions, targets.to(DEVICE))
            # AutoEncoder Transfer Net
            predictions = ae_transfer_net.targettask(data.to(DEVICE))
            ae_transfer_loss = LOSS_FUNC(predictions, targets.to(DEVICE))
            # Training and recording loss
            regular_loss.backward(); optimizer_regular.step()
            transfer_loss.backward(); optimizer_transfer.step()
            ae_transfer_loss.backward(); optimizer_ae_transfer.step()           
            regular_loss_ep += regular_loss.item()
            transfer_loss_ep += transfer_loss.item()
            ae_transfer_loss_ep += ae_transfer_loss.item()
            if (j+1) % 50 == 0 or (j+1) % n_minibatches == 0:
                # Validation
                for data, targets in valid_loader: # only load up one batch
                    assert data.size() == (TARGETTASK['Valid'], INPUT_SIZE) and \
                        targets.size() == (TARGETTASK['Valid'], )
                    targets = utils.convert_targets(targets, TARGETTASK)                    
                    with torch.no_grad():
                        predictions = regular_net.targettask(data.to(DEVICE))
                        regular_loss = LOSS_FUNC(predictions, targets.to(DEVICE)).item()
                        regular_accuracy = utils.compute_accuracy(predictions.detach().cpu(), targets)
                        predictions = transfer_net.targettask(data.to(DEVICE))
                        transfer_loss = LOSS_FUNC(predictions, targets.to(DEVICE)).item()
                        transfer_accuracy = utils.compute_accuracy(predictions.detach().cpu(), targets)
                        predictions = ae_transfer_net.targettask(data.to(DEVICE))
                        ae_transfer_loss = LOSS_FUNC(predictions, targets.to(DEVICE)).item()
                        ae_transfer_accuracy = utils.compute_accuracy(predictions.detach().cpu(), targets)
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
                    if (regular_accuracy > regular_best_point):
                        regular_net.save_model()
                        regular_best_point = regular_accuracy
                    if (transfer_accuracy > transfer_best_point):
                        transfer_net.save_model()
                        transfer_best_point = transfer_accuracy
                    if (ae_transfer_accuracy > ae_transfer_best_point):
                        ae_transfer_net.save_model()
                        ae_transfer_best_point = ae_transfer_accuracy    
                else:
                    regular_net.save_model()
                    transfer_net.save_model()
                    ae_transfer_net.save_model()
                np.save(f"Trained_Models/{TASK_DESCR}/train_losses_targettask.npy", np.array(train_loss_eps))
                np.save(f"Trained_Models/{TASK_DESCR}/valid_losses_targettask.npy", np.array(valid_loss_eps))
                np.save(f"Trained_Models/{TASK_DESCR}/valid_accuracies_targettask.npy", np.array(valid_accuracies_eps))

    print(f"[{TASK_DESCR}] Training finished!")
