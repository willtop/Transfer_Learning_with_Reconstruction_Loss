# Script for evaluating MIMO network objectives: beamforming and localization

import numpy as np
import torch
import matplotlib.pyplot as plt
from neural_nets import Regular_Net, Transfer_Net, Autoencoder_Transfer_Net
from utils import *
from setup import *

VISUALIZE_ALLOCATIONS = False
GP_INCLUDED = True
EVALUATE_EARLY_STOP = False
PLOT_STYLES = {
    "Regular Learning": "m--",
    "Conventional Transfer": "g-.",
    "Transfer with Reconstruct": "r-",
    "Random Localization": "k:",
    "Random Beamformers": "k:",
    "Perfect Beamformers": "y:"
}

# Numpy computation
def compute_beamformer_gains(beamformers, channels):
    n_networks = np.shape(beamformers)[0]
    assert np.shape(channels) == (n_networks, N_BS, N_BS_ANTENNAS) and \
           np.shape(beamformers) == (n_networks, N_BS, N_BS_ANTENNAS)
    # ensure beamformers are normalized to unit power
    beamformer_powers = np.linalg.norm(beamformers, axis=-1).flatten()
    assert np.all(beamformer_powers<1.01) and np.all(beamformer_powers>0.99)
    # compute beamformer gain across all BSs 
    channel_gains = np.sum(beamformers * channels.conj(), axis=-1)
    channel_gains = np.power(np.abs(channel_gains), 2)
    assert np.shape(channel_gains) == (n_networks, N_BS)
    # keep track of gains at individual BSs
    return channel_gains

# Aggregate channel gains across all BSs for SNR value
def compute_snrs(gains):
    n_networks = np.shape(gains)[0]
    assert np.shape(gains) == (n_networks, N_BS)
    return np.sum(gains,axis=1)/NOISE_POWER

if(__name__ =='__main__'):
    uelocs = np.load("Data/uelocs_test.npy")
    channels = np.load("Data/channels.npy")
    assert np.shape(uelocs) == (N_TEST_SAMPLES, 3) and \
           np.shape(channels) == (N_TEST_SAMPLES, N_BS, N_BS_ANTENNAS)
    print(f"[MIMO] Evaluate {SOURCETASK['Task']}->{TARGETTASK['Task']} over {N_TEST_SAMPLES} layouts.")
    # Take uplink pilots measurements
    measures = obtain_measured_uplink_signals(channels)

    regular_net, transfer_net, ae_transfer_net = \
         Regular_Net(EVALUATE_EARLY_STOP).to(DEVICE), \
         Transfer_Net(EVALUATE_EARLY_STOP).to(DEVICE), \
         Autoencoder_Transfer_Net(EVALUATE_EARLY_STOP).to(DEVICE)

    task_type = "Source Task" if SOURCETASK['Task'] == "Beamforming" else "Target Task"
    print(f"<<<<<<<<<<<Beamforming Task as {task_type}>>>>>>>>>")
    print("Collecting beamforming solutions...")
    beamformers_all = {}
    if SOURCETASK['Task'] == "Beamforming":
        beamformers_all['Regular Learning'] = regular_net.sourcetask(torch.tensor(measures, dtype=torch.cfloat).to(DEVICE)).detach().cpu().numpy()
        beamformers_all['Conventional Transfer'] = transfer_net.sourcetask(torch.tensor(measures, dtype=torch.cfloat).to(DEVICE)).detach().cpu().numpy()
        beamformers_all['Transfer with Reconstruct'] = ae_transfer_net.sourcetask(torch.tensor(measures, dtype=torch.cfloat).to(DEVICE)).detach().cpu().numpy()
    else:
        beamformers_all['Regular Learning'] = regular_net.targettask(torch.tensor(measures, dtype=torch.cfloat).to(DEVICE)).detach().cpu().numpy()
        beamformers_all['Conventional Transfer'] = transfer_net.targettask(torch.tensor(measures, dtype=torch.cfloat).to(DEVICE)).detach().cpu().numpy()
        beamformers_all['Transfer with Reconstruct'] = ae_transfer_net.targettask(torch.tensor(measures, dtype=torch.cfloat).to(DEVICE)).detach().cpu().numpy()
    beamformers_all['Perfect Beamformers'] = channels/np.linalg.norm(channels, axis=-1, keepdims=True)
    tmp = generate_circular_gaussians(size_to_generate=(N_TEST_SAMPLES, N_BS, N_BS_ANTENNAS))
    beamformers_all['Random Beamformers'] = tmp/np.linalg.norm(tmp, axis=-1, keepdims=True)
        
    print("Evaluating beamforming performances...")
    gains_all, snrs_all = {}, {}
    for method_key, beamformers in beamformers_all.items():
        gains = compute_beamformer_gains(beamformers, channels)
        snrs = compute_snrs(gains)
        assert np.shape(gains) == (N_TEST_SAMPLES, N_BS) and \
               np.shape(snrs) == (N_TEST_SAMPLES, )
        gains_all[method_key] = gains
        snrs_all[method_key] = snrs
        
    # reiterate to get the percentage w.r.t. perfect beamformers
    for method_key in beamformers_all.keys():
        print("[{}]: avg gain per BS: {:.3f} ({:.2f}% of perfect beamformers); avg SNR level per network: {:.3f}dB ({:.2f}% of perfect beamformers in absolute scale)".format(
                method_key, np.mean(gains_all[method_key]), np.mean(gains_all[method_key]/gains_all["Perfect Beamformers"])*100,
                np.mean(10*np.log10(snrs_all[method_key])), np.mean(snrs_all[method_key]/snrs_all["Perfect Beamformers"])*100))

    # Plot the beamforming gains CDF curve
    # get the lower bound plot
    lowerbound_plot, upperbound_plot = np.inf, -np.inf
    for val in gains_all.values():
        lowerbound_plot = min(lowerbound_plot, np.percentile(val,q=10, interpolation="lower"))
        upperbound_plot = max(upperbound_plot, np.percentile(val,q=90, interpolation="lower"))

    fig = plt.figure()
    plt.xlabel(f"Beamforming Gains ({task_type})", fontsize=20)
    plt.ylabel("Cumulative Distribution of Beamforming Gains Values", fontsize=20)
    plt.xticks(fontsize=21)
    plt.yticks(np.linspace(start=0, stop=1, num=5), ["{}%".format(int(i*100)) for i in np.linspace(start=0, stop=1, num=5)], fontsize=21)
    plt.grid(linestyle="dotted")
    plt.ylim(bottom=0)
    for method_key, gains in gains_all.items():
        plt.plot(np.sort(gains), np.arange(1,N_TEST_SAMPLES+1)/N_TEST_SAMPLES, PLOT_STYLES[method_key], linewidth=2.0, label=method_key)
    plt.xlim(left=lowerbound_plot/1e6, right=upperbound_plot/1e6)
    plt.legend(prop={'size':20}, loc='lower right')
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    plt.show()   

    # Visualize power control solutions and achieved rates for selected layouts
    if VISUALIZE_ALLOCATIONS:
        rand_idxes = np.random.randint(N_TEST_SAMPLES, size=3)    
        for id in rand_idxes:
            fig, axs = plt.subplots(5,4)
            fig.suptitle(f"{SOURCETASK['Task']}-{TARGETTASK['Task']} Test Layout #{id}")
            # plot channels
            axs[0][0].set_title("Channels")
            axs[0][0].plot(g[id].flatten())
            plot_label_direct_channels(axs[0][0])
            # plot for source task
            if SOURCETASK['Task'] == "Sum-Rate":
                optimal_benchmark = "FP" 
            elif SOURCETASK['Task'] == "Min-Rate" and GP_INCLUDED:
                optimal_benchmark = "GP"
            else:
                optimal_benchmark = "Regular Learning"
            methods_plotted = set([optimal_benchmark, "Regular Learning", "Conventional Transfer Learning", "Autoencoder Transfer Learning"])
            for i, method_key in enumerate(methods_plotted):
                axs[1][i].plot(np.arange(1, N_LINKS+1), power_controls[SOURCETASK['Type']][method_key][id], label="{}_pc".format(method_key))
                axs[1][i].legend()
                axs[2][i].plot(np.arange(1, N_LINKS+1), rates_all[SOURCETASK['Type']][method_key][id], label="{}_rates".format(method_key))
                axs[2][i].legend()
            # plot for target task
            if TARGETTASK['Task'] == "Sum-Rate":
                optimal_benchmark = "FP" 
            elif TARGETTASK['Task'] == "Min-Rate" and GP_INCLUDED:
                optimal_benchmark = "GP"
            else:
                optimal_benchmark = "Regular Learning"
            methods_plotted = set([optimal_benchmark, "Regular Learning", "Conventional Transfer Learning", "Autoencoder Transfer Learning"])
            for i, method_key in enumerate(methods_plotted):
                axs[3][i].plot(np.arange(1, N_LINKS+1), power_controls[TARGETTASK['Type']][method_key][id], label="{}_pc".format(method_key))
                axs[3][i].legend()
                axs[4][i].plot(np.arange(1, N_LINKS+1), rates_all[TARGETTASK['Type']][method_key][id], label="{}_rates".format(method_key))
                axs[4][i].legend()
            plt.show()
    print("Evaluation Finished Successfully!")
