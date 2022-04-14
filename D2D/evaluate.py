# Script for evaluating D2D network objectives: sum rate or min rate

import numpy as np
import torch
import matplotlib.pyplot as plt
from neural_nets import Regular_Net, Transfer_Net, Autoencoder_Transfer_Net
from utils import *
from setup import *

VISUALIZE_POWERCONTROL = False

if(__name__ =='__main__'):
    g = np.load("Data/g_test_{}.npy".format(SETTING_STRING))
    assert np.shape(g) == (N_SAMPLES['Test'], N_LINKS, N_LINKS)
    print(f"[D2D] Evaluate {SETTING_STRING} over {N_SAMPLES['Test']} layouts.")

    for task in ['Source Task', 'Target Task']:
        print(f"Evaluatin {task}...")
        power_controls, plot_colors, plot_linestyles = {}, {}, {}
        optimal_benchmark = "FP" if "Source" in task else "GP"
        if "Source" in task:
            # Fractional Programming
            power_controls["FP"] = FP_power_control(g)
            plot_colors["FP"] = 'b'
            plot_linestyles["FP"] = '--'
        else:
            # Geometric Programming
            power_controls["GP"] = GP_power_control()
            plot_colors["GP"] = 'b'
            plot_linestyles["GP"] = '--'
        # Deep Learning methods
        regular_net, transfer_net, ae_transfer_net = Regular_Net().to(DEVICE), Transfer_Net().to(DEVICE), Autoencoder_Transfer_Net().to(DEVICE)
        if "Source" in task:
            pc, _ = regular_net.sourceTask_powerControl(torch.tensor(g, dtype=torch.float32).to(DEVICE))
            power_controls["Regular Learning"] = pc.detach().cpu().numpy()
            pc, _ = transfer_net.sourceTask_powerControl(torch.tensor(g, dtype=torch.float32).to(DEVICE))
            power_controls["Transfer Learning"] = pc.detach().cpu().numpy()
            pc, _, _ = ae_transfer_net.sourceTask_powerControl(torch.tensor(g, dtype=torch.float32).to(DEVICE))
            power_controls["Autoencoder Transfer Learning"] = pc.detach().cpu().numpy()
        else:
            pc, _ = regular_net.targetTask_powerControl(torch.tensor(g, dtype=torch.float32).to(DEVICE))
            power_controls["Regular Learning"] = pc.detach().cpu().numpy()
            pc, _ = transfer_net.targetTask_powerControl(torch.tensor(g, dtype=torch.float32).to(DEVICE))
            power_controls["Transfer Learning"] = pc.detach().cpu().numpy()
            pc, _ = ae_transfer_net.targetTask_powerControl(torch.tensor(g, dtype=torch.float32).to(DEVICE))
            power_controls["Autoencoder Transfer Learning"] = pc.detach().cpu().numpy()
        plot_colors["Regular Learning"] = 'm'
        plot_linestyles["Regular Learning"] = '--'
        plot_colors["Transfer Learning"] = 'g'
        plot_linestyles["Transfer Learning"] = '-.'
        plot_colors["Autoencoder Transfer Learning"] = 'r'
        plot_linestyles["Autoencoder Transfer Learning"] = '-'
        # Trivial Benchmarks
        power_controls["Random Power"] = np.random.uniform(size=[N_SAMPLES['Test'], N_LINKS])
        plot_colors["Random Power"] = 'k'
        plot_linestyles["Random Power"] = ':'
        power_controls["Full Power"] = np.ones(shape=[N_SAMPLES['Test'], N_LINKS], dtype=float)
        plot_colors["Full Power"] = 'y'
        plot_linestyles["Full Power"] = ':'

        print(f"Power Allocation Percentages on {task}: ")
        for method_key, power_percentages in power_controls.items():
            print("[{}]: {:.3f}%;".format(method_key, np.mean(power_percentages)*100), end="")
        print("\n")

        print(f"{task} Performances: ")
        objectives = {}
        for method_key, power_percentages in power_controls.items():
            rates = compute_rates(compute_SINRs(power_percentages, g))
            assert np.shape(rates) == (N_SAMPLES['Test'], N_LINKS)
            if "Source" in task:
                objectives[method_key] = np.sum(rates, axis=1)
            else:
                objectives[method_key] = np.min(rates, axis=1)
        print("[{}]: {:.3f}Mbps".format(optimal_benchmark, np.mean(objectives[optimal_benchmark])/1e6))
        for method_key, objective in objectives.items():
            if method_key == optimal_benchmark:
                continue
            print("[{}]: {:.3f}Mbps, {:.2f}% of {};".format(method_key, np.mean(objective)/1e6, np.mean(objective/objectives[optimal_benchmark])*100, optimal_benchmark), end="")
        print("\n")


        # Plot the CDF curve
        # get the lower bound plot
        lowerbound_plot, upperbound_plot = np.inf, -np.inf
        for val in objectives.values():
            lowerbound_plot = min(lowerbound_plot, np.percentile(val,q=10, interpolation="lower"))
            upperbound_plot = max(upperbound_plot, np.percentile(val,q=90, interpolation="lower"))

        fig = plt.figure()
        plt.xlabel(f"{task} (Mbps)", fontsize=20)
        plt.ylabel(f"Cumulative Distribution of {task} Values", fontsize=20)
        plt.xticks(fontsize=21)
        plt.yticks(np.linspace(start=0, stop=1, num=5), ["{}%".format(int(i*100)) for i in np.linspace(start=0, stop=1, num=5)], fontsize=21)
        plt.grid(linestyle="dotted")
        plt.ylim(bottom=0)
        for method_key in power_controls.keys():
            plt.plot(np.sort(objectives[method_key])/1e6, np.arange(1,N_SAMPLES['Test']+1)/N_SAMPLES['Test'], color=plot_colors[method_key], linestyle=plot_linestyles[method_key], linewidth=2.0, label=method_key)
        plt.xlim(left=lowerbound_plot/1e6, right=upperbound_plot/1e6)
        plt.legend(prop={'size':20}, loc='lower right')
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
        plt.show()   

        # Visualize power control solutions for selected layouts
        if VISUALIZE_POWERCONTROL:
            rand_idxes = np.random.randint(N_SAMPLES['Test'], size=3)    
            for id in rand_idxes:
                fig, axs = plt.subplots(2,2)
                fig.suptitle(f"{task} Power Allocation on Test Layout #{id}")
                axs = axs.flatten()
                for i, method_key in enumerate([optimal_benchmark, "Regular Learning", "Transfer Learning", "Autoencoder Transfer Learning"]):
                    axs[i].set_title(method_key)
                    axs[i].plot(power_controls[method_key][id])
                plt.show()
    print("Evaluation Finished Successfully!")
