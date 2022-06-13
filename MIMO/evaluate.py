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

if(__name__ =='__main__'):
    uelocs = np.load("Data/uelocs_test.npy")
    channels = np.load("Data/channels.npy")
    assert np.shape(uelocs) == (N_TEST_SAMPLES, 3) and \
           np.shape(channels) == (N_TEST_SAMPLES, N_BS, N_BS_ANTENNAS)
    print(f"[MIMO] Evaluate {SOURCETASK['Task']}->{TARGETTASK['Task']} over {N_TEST_SAMPLES} layouts.")

    regular_net, transfer_net, ae_transfer_net = \
         Regular_Net(EVALUATE_EARLY_STOP).to(DEVICE), \
         Transfer_Net(EVALUATE_EARLY_STOP).to(DEVICE), \
         Autoencoder_Transfer_Net(EVALUATE_EARLY_STOP).to(DEVICE)

    print("<<<<<<<<<<<Evaluating Beamforming Performances>>>>>>>>>")
    beamformers = {}
    beamformers['Regular Learning'] = 
    beamformers['Perfect Beamformers'] = channels/np.linalg.norm(channels, axis=-1, keepdims=True)
    tmp = generate_circular_gaussians(size_to_generate=(N_TEST_SAMPLES, N_BS, N_BS_ANTENNAS))
    beamformers['Random Beamformers'] = tmp/np.linalg.norm(tmp, axis=-1, keepdims=True)
        elif task['Task'] == "Min-Rate" and GP_INCLUDED:
            optimal_benchmark = "GP"
            # Geometric Programming
            power_controls[task['Type']]["GP"] = GP_power_control()
            plot_colors["GP"] = 'b'
            plot_linestyles["GP"] = '--'
        else:
            optimal_benchmark = "Regular Learning"
        # Deep Learning methods
        if task['Type'] == "Source-Task":
            pc, _ = regular_net.sourceTask_powerControl(torch.tensor(g, dtype=torch.float32).to(DEVICE))
            power_controls[task['Type']]["Regular Learning"] = pc.detach().cpu().numpy()
            pc, _ = transfer_net.sourceTask_powerControl(torch.tensor(g, dtype=torch.float32).to(DEVICE))
            power_controls[task['Type']]["Conventional Transfer Learning"] = pc.detach().cpu().numpy()
            pc, _, _ = ae_transfer_net.sourceTask_powerControl(torch.tensor(g, dtype=torch.float32).to(DEVICE))
            power_controls[task['Type']]["Autoencoder Transfer Learning"] = pc.detach().cpu().numpy()
        else:
            pc, _ = regular_net.targetTask_powerControl(torch.tensor(g, dtype=torch.float32).to(DEVICE))
            power_controls[task['Type']]["Regular Learning"] = pc.detach().cpu().numpy()
            pc, _ = transfer_net.targetTask_powerControl(torch.tensor(g, dtype=torch.float32).to(DEVICE))
            power_controls[task['Type']]["Conventional Transfer Learning"] = pc.detach().cpu().numpy()
            pc, _ = ae_transfer_net.targetTask_powerControl(torch.tensor(g, dtype=torch.float32).to(DEVICE))
            power_controls[task['Type']]["Autoencoder Transfer Learning"] = pc.detach().cpu().numpy()
        
        # Trivial Benchmarks
        power_controls[task['Type']]["Random Power"] = np.random.uniform(size=[N_TEST_SAMPLES, N_LINKS])
        power_controls[task['Type']]["Full Power"] = np.ones(shape=[N_TEST_SAMPLES, N_LINKS], dtype=float)
        

        print(f"Power Allocation Percentages on {task['Type']} {task['Task']}: ")
        for method_key, power_percentages in power_controls[task['Type']].items():
            print("[{}]: {:.1f}%;".format(method_key, np.mean(power_percentages)*100), end="")
        print("\n")

        print(f"{task['Type']} {task['Task']} Performances: ")
        sinrs_all, rates_all[task['Type']], objectives = {}, {}, {}
        for method_key, power_percentages in power_controls[task['Type']].items():
            sinrs = compute_SINRs(power_percentages, g)
            rates = compute_rates(sinrs)
            assert np.shape(sinrs) == np.shape(rates) == (N_TEST_SAMPLES, N_LINKS)
            sinrs_all[method_key] = sinrs
            rates_all[task['Type']][method_key] = rates
            if task['Task'] == 'Sum-Rate':
                objectives[method_key] = np.sum(rates, axis=1)
            else:
                objectives[method_key] = np.min(rates, axis=1)
            assert np.shape(objectives[method_key]) == (N_TEST_SAMPLES, )
        
        # Ensure SINR mostly at positive decibels
        print("Percentages of layout with worst link SINR above 0dB")
        for method_key, sinrs in sinrs_all.items():
            pert = np.mean(convert_SINRs_to_dB(np.min(sinrs,axis=1))>0)*100
            print("[{}] {:.1f}%;".format(method_key, pert)) 
        print("\n")

        for method_key, objective in objectives.items():
            # Different statistics monitored for different tasks
            print("[{}]: {:.3f}Mbps;".format(method_key, np.mean(objective)/1e6), end="")
        print("\n")
        for method_key, objective in objectives.items():
            if method_key == optimal_benchmark:
                continue
            print("[{}]: {:.2f}% of {};".format(method_key, np.mean(objective/objectives[optimal_benchmark])*100, optimal_benchmark), end="")
        print("\n")


        # Plot the CDF curve
        # get the lower bound plot
        lowerbound_plot, upperbound_plot = np.inf, -np.inf
        for val in objectives.values():
            lowerbound_plot = min(lowerbound_plot, np.percentile(val,q=10, interpolation="lower"))
            upperbound_plot = max(upperbound_plot, np.percentile(val,q=90, interpolation="lower"))

        fig = plt.figure()
        plt.xlabel(f"{task['Type']} {task['Task']} (Mbps)", fontsize=20)
        plt.ylabel(f"Cumulative Distribution of {task['Task']} Values", fontsize=20)
        plt.xticks(fontsize=21)
        plt.yticks(np.linspace(start=0, stop=1, num=5), ["{}%".format(int(i*100)) for i in np.linspace(start=0, stop=1, num=5)], fontsize=21)
        plt.grid(linestyle="dotted")
        plt.ylim(bottom=0)
        for method_key in power_controls[task['Type']].keys():
            plt.plot(np.sort(objectives[method_key])/1e6, np.arange(1,N_TEST_SAMPLES+1)/N_TEST_SAMPLES, color=plot_colors[method_key], linestyle=plot_linestyles[method_key], linewidth=2.0, label=method_key)
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
