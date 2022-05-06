# Script for evaluating D2D network objectives: sum rate or min rate

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import savemat
from neural_nets import Regular_Net, Transfer_Net, Autoencoder_Transfer_Net
from utils import *
from setup import *

VISUALIZE_SOURCETASK = False
VISUALIZE_TARGETTASK = True

if(__name__ =='__main__'):
    g = np.load("Data/g_test_{}.npy".format(SETTING_STRING))
    assert np.shape(g) == (N_TEST_SAMPLES, N_LINKS, N_LINKS)
    print(f"[D2D] Evaluate {SETTING_STRING} over {N_TEST_SAMPLES} layouts.")

    regular_net, transfer_net, ae_transfer_net = Regular_Net().to(DEVICE), Transfer_Net().to(DEVICE), Autoencoder_Transfer_Net().to(DEVICE)
    for task in [SOURCETASK, TARGETTASK]:
        print(f"Evaluating {task['Type']}: {task['Task']}...")
        power_controls, plot_colors, plot_linestyles = {}, {}, {}
        if task['Task'] == "Sum":
            optimal_benchmark = "FP"
            # Fractional Programming
            power_controls["FP"] = FP_power_control(g)
            plot_colors["FP"] = 'b'
            plot_linestyles["FP"] = '--' 
        elif task['Task'] == "Min":
            optimal_benchmark = "GP"
            # Geometric Programming
            power_controls["GP"] = GP_power_control()
            plot_colors["GP"] = 'b'
            plot_linestyles["GP"] = '--'
        else:
            optimal_benchmark = "Regular Learning"
        # Deep Learning methods
        if task['Type'] == "Source-Task":
            pc, _ = regular_net.sourceTask_powerControl(torch.tensor(g, dtype=torch.float32).to(DEVICE))
            power_controls["Regular Learning"] = pc.detach().cpu().numpy()
            pc, _ = transfer_net.sourceTask_powerControl(torch.tensor(g, dtype=torch.float32).to(DEVICE))
            power_controls["Conventional Transfer Learning"] = pc.detach().cpu().numpy()
            pc, _, _ = ae_transfer_net.sourceTask_powerControl(torch.tensor(g, dtype=torch.float32).to(DEVICE))
            power_controls["Autoencoder Transfer Learning"] = pc.detach().cpu().numpy()
        else:
            pc, _ = regular_net.targetTask_powerControl(torch.tensor(g, dtype=torch.float32).to(DEVICE))
            power_controls["Regular Learning"] = pc.detach().cpu().numpy()
            pc, _ = transfer_net.targetTask_powerControl(torch.tensor(g, dtype=torch.float32).to(DEVICE))
            power_controls["Conventional Transfer Learning"] = pc.detach().cpu().numpy()
            pc, _ = ae_transfer_net.targetTask_powerControl(torch.tensor(g, dtype=torch.float32).to(DEVICE))
            power_controls["Autoencoder Transfer Learning"] = pc.detach().cpu().numpy()
        plot_colors["Regular Learning"] = 'm'
        plot_linestyles["Regular Learning"] = '--'
        plot_colors["Conventional Transfer Learning"] = 'g'
        plot_linestyles["Conventional Transfer Learning"] = '-.'
        plot_colors["Autoencoder Transfer Learning"] = 'r'
        plot_linestyles["Autoencoder Transfer Learning"] = '-'
        # Trivial Benchmarks
        power_controls["Random Power"] = np.random.uniform(size=[N_TEST_SAMPLES, N_LINKS])
        plot_colors["Random Power"] = 'k'
        plot_linestyles["Random Power"] = ':'
        power_controls["Full Power"] = np.ones(shape=[N_TEST_SAMPLES, N_LINKS], dtype=float)
        plot_colors["Full Power"] = 'y'
        plot_linestyles["Full Power"] = ':'

        print(f"Power Allocation Percentages on {task['Type']} {task['Task']}: ")
        for method_key, power_percentages in power_controls.items():
            print("[{}]: {:.1f}%;".format(method_key, np.mean(power_percentages)*100), end="")
        print("\n")

        print(f"{task['Type']} {task['Task']} Performances: ")
        sinrs_all, rates_all, objectives = {}, {}, {}
        for method_key, power_percentages in power_controls.items():
            sinrs = compute_SINRs(power_percentages, g)
            rates = compute_rates(sinrs)
            assert np.shape(sinrs) == np.shape(rates) == (N_TEST_SAMPLES, N_LINKS)
            sinrs_all[method_key] = sinrs
            rates_all[method_key] = rates
            if task['Task'] == 'Sum':
                objectives[method_key] = np.sum(rates, axis=1)
            elif task['Task'] == 'Min':
                objectives[method_key] = np.min(rates, axis=1)
            elif task['Task'] == 'Harmonic':
                objectives[method_key] = N_LINKS / np.sum(1/(rates/1e6), axis=1)
            else:
                print(f"Invalid task type: {task['Task']}! Exiting ...")
                exit(1)
            assert np.shape(objectives[method_key]) == (N_TEST_SAMPLES, )
        
        # Ensure SINR mostly at positive decibels
        print("Percentages of layout with worst link SINR above 0dB")
        for method_key, sinrs in sinrs_all.items():
            pert = np.mean(convert_SINRs_to_dB(np.min(sinrs,axis=1))>0)*100
            print("[{}] {:.1f}%;".format(method_key, pert)) 
        print("\n")

        for method_key, objective in objectives.items():
            # Different statistics monitored for different tasks
            if task['Task'] in ["Sum", "Min"]:
                print("[{}]: {:.3f}Mbps;".format(method_key, np.mean(objective)/1e6), end="")
            elif task['Task'] == "Harmonic":
                print("[{}]: {:.3f}Mbps;".format(method_key, np.mean(objective)), end="")
            else:
                print(f"Invalid task {task['Task']}! Exiting...")
                exit(1)
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
        plt.xlabel(f"{task['Type']} {task['Fullname']} (Mbps)", fontsize=20)
        plt.ylabel(f"Cumulative Distribution of {task['Fullname']} Values", fontsize=20)
        plt.xticks(fontsize=21)
        plt.yticks(np.linspace(start=0, stop=1, num=5), ["{}%".format(int(i*100)) for i in np.linspace(start=0, stop=1, num=5)], fontsize=21)
        plt.grid(linestyle="dotted")
        plt.ylim(bottom=0)
        for method_key in power_controls.keys():
            plt.plot(np.sort(objectives[method_key])/1e6, np.arange(1,N_TEST_SAMPLES+1)/N_TEST_SAMPLES, color=plot_colors[method_key], linestyle=plot_linestyles[method_key], linewidth=2.0, label=method_key)
        plt.xlim(left=lowerbound_plot/1e6, right=upperbound_plot/1e6)
        plt.legend(prop={'size':20}, loc='lower right')
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
        plt.show()   

        # Visualize power control solutions and achieved rates for selected layouts
        if (task['Type']=="Source-Task" and VISUALIZE_SOURCETASK) or (task['Type']=="Target-Task" and VISUALIZE_TARGETTASK):
            # If optimal benchmark is regular learning, then put a set to remove duplicate
            methods_plotted = set([optimal_benchmark, "Regular Learning", "Conventional Transfer Learning", "Autoencoder Transfer Learning", "Random Power"])
            rand_idxes = np.random.randint(N_TEST_SAMPLES, size=3)    
            for id in rand_idxes:
                fig, axs = plt.subplots(3,1)
                fig.suptitle(f"{task['Fullname']} Test Layout #{id}")
                axs = axs.flatten()
                # plot channels
                axs[0].set_title("Channels")
                axs[0].plot(g[id].flatten())
                # plot power allocations
                axs[1].set_title("Power Allocations")
                for method_key in methods_plotted:
                    axs[1].plot(np.arange(1, N_LINKS+1), power_controls[method_key][id], label=method_key)
                axs[1].legend()
                # plot achieved rates
                axs[2].set_title("Link Rate")
                for method_key in methods_plotted:
                    axs[2].plot(np.arange(1, N_LINKS+1), rates_all[method_key][id], label=method_key)
                axs[2].legend()
                plt.show()
    print("Evaluation Finished Successfully!")
