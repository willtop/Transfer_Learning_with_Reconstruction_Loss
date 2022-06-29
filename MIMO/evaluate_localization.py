# Script for evaluating MIMO network objective: localization

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from neural_nets import Regular_Net, Transfer_Net, Autoencoder_Transfer_Net
from utils import *
from setup import *

EVALUATE_EARLY_STOP = True
PLOT_STYLES = {
    "Regular Learning": "m--",
    "Conventional Transfer": "g-.",
    "Transfer with Reconstruct": "r-",
    "Random Localization": "c:",
    "Center Localization": "y:"
}

def postprocess_location_predictions(locs):
    n_networks = np.shape(locs)[0]
    assert np.shape(locs)==(n_networks, 2)
    locs_postprocessed = np.concatenate([locs, np.zeros(shape=(n_networks, 1), dtype=float)], axis=1)
    return locs_postprocessed

# Numpy computation
def compute_localization_errors(uelocs_predicted, uelocs):
    n_networks = np.shape(uelocs_predicted)[0]
    assert np.shape(uelocs_predicted) == np.shape(uelocs) == (n_networks, 3)
    # ensure localization predictions are within the bounds
    assert np.min(uelocs_predicted[:,0])>=UE_LOCATION_XMIN and np.max(uelocs_predicted[:,0])<=UE_LOCATION_XMAX and \
           np.min(uelocs_predicted[:,1])>=UE_LOCATION_YMIN and np.max(uelocs_predicted[:,1])<=UE_LOCATION_YMAX and \
           np.all(uelocs_predicted[:,2]==0)
    # compute the localization error in terms of the euclidean distance to the true locations
    localization_errors = np.linalg.norm(uelocs-uelocs_predicted, axis=1)
    return localization_errors


if(__name__ =='__main__'):
    uelocs = np.load("Data/uelocs_test.npy")
    channels = np.load("Data/channels_test.npy")
    measures = np.load("Data/measures_test.npy")
    assert np.shape(uelocs) == (N_TEST_SAMPLES, 3) and \
           np.shape(channels) == (N_TEST_SAMPLES, N_BS, N_BS_ANTENNAS) and \
           np.shape(measures) == (N_TEST_SAMPLES, N_BS, N_PILOTS)
    print(f"[MIMO] Evaluate {SOURCETASK['Task']}->{TARGETTASK['Task']} over {N_TEST_SAMPLES} layouts.")

    regular_net, transfer_net, ae_transfer_net = \
         Regular_Net(EVALUATE_EARLY_STOP).to(DEVICE), \
         Transfer_Net(EVALUATE_EARLY_STOP).to(DEVICE), \
         Autoencoder_Transfer_Net(EVALUATE_EARLY_STOP).to(DEVICE)

    task_type = "Source Task" if SOURCETASK['Task'] == "Localization" else "Target Task"
    print(f"<<<<<<<<<<<Localization Task as {task_type}>>>>>>>>>")
    print("Collecting localization solutions...")
    uelocs_predicted_all = {}
    if SOURCETASK['Task'] == "Localization":
        uelocs_predicted_all['Regular Learning'] = postprocess_location_predictions(regular_net.sourcetask(torch.tensor(measures, dtype=torch.cfloat).to(DEVICE)).detach().cpu().numpy())
        uelocs_predicted_all['Conventional Transfer'] = postprocess_location_predictions(transfer_net.sourcetask(torch.tensor(measures, dtype=torch.cfloat).to(DEVICE)).detach().cpu().numpy())
        tmp, _ = ae_transfer_net.sourcetask(torch.tensor(measures, dtype=torch.cfloat).to(DEVICE))
        uelocs_predicted_all['Transfer with Reconstruct'] = postprocess_location_predictions(tmp.detach().cpu().numpy())
    else:
        uelocs_predicted_all['Regular Learning'] = postprocess_location_predictions(regular_net.targettask(torch.tensor(measures, dtype=torch.cfloat).to(DEVICE)).detach().cpu().numpy())
        uelocs_predicted_all['Conventional Transfer'] = postprocess_location_predictions(transfer_net.targettask(torch.tensor(measures, dtype=torch.cfloat).to(DEVICE)).detach().cpu().numpy())
        uelocs_predicted_all['Transfer with Reconstruct'] = postprocess_location_predictions(ae_transfer_net.targettask(torch.tensor(measures, dtype=torch.cfloat).to(DEVICE)).detach().cpu().numpy())
    uelocs_predicted_all['Random Localization'] = np.concatenate([np.random.uniform(low=UE_LOCATION_XMIN, high=UE_LOCATION_XMAX, size=(N_TEST_SAMPLES, 1)), \
                                                                  np.random.uniform(low=UE_LOCATION_YMIN, high=UE_LOCATION_YMAX, size=(N_TEST_SAMPLES, 1)), \
                                                                  np.zeros(shape=(N_TEST_SAMPLES,1),dtype=float)], axis=1)
    # For center localization, just guess the center of the possible UE distribution region
    uelocs_predicted_all['Center Localization'] = np.concatenate([np.ones(shape=(N_TEST_SAMPLES, 1),dtype=float)*(UE_LOCATION_XMIN+UE_LOCATION_XMAX)/2, \
                                                                  np.ones(shape=(N_TEST_SAMPLES, 1),dtype=float)*(UE_LOCATION_YMIN+UE_LOCATION_YMAX)/2, \
                                                                  np.zeros(shape=(N_TEST_SAMPLES,1),dtype=float)], axis=1)
         
    print("Evaluating localization performances...")
    errors_all = {}
    for method_key, uelocs_predicted in uelocs_predicted_all.items():
        errors = compute_localization_errors(uelocs_predicted, uelocs)
        assert np.shape(errors) == (N_TEST_SAMPLES, )
        errors_all[method_key] = errors
        print("[{}]: avg error: {:.3f}m".format(method_key, np.mean(errors)))

    # Plot the localization errors CDF curve
    # get the lower bound plot
    lowerbound_plot, upperbound_plot = np.inf, -np.inf
    for val in errors_all.values():
        lowerbound_plot = min(lowerbound_plot, np.percentile(val, q=2, interpolation="lower"))
        upperbound_plot = max(upperbound_plot, np.percentile(val, q=50, interpolation="lower"))

    fig = plt.figure()
    plt.xlabel(f"[{task_type}] Localization Errors (m)", fontsize=20)
    plt.ylabel("Cumulative Distribution of Localization Errors Values", fontsize=20)
    plt.xticks(fontsize=21)
    plt.yticks(np.linspace(start=0, stop=1, num=5), ["{}%".format(int(i*100)) for i in np.linspace(start=0, stop=1, num=5)], fontsize=21)
    plt.grid(linestyle="dotted")
    plt.ylim(bottom=0)
    for method_key, errors in errors_all.items():
        plt.plot(np.sort(errors), np.arange(1,N_TEST_SAMPLES+1)/N_TEST_SAMPLES, PLOT_STYLES[method_key], linewidth=2.0, label=method_key)
    plt.xlim(left=lowerbound_plot, right=upperbound_plot)
    plt.legend(prop={'size':20}, loc='lower right')
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    plt.show()   

    # Visualize localization results by each method
    rand_idxes = np.random.randint(N_TEST_SAMPLES, size=3)    
    for id in rand_idxes:
        fig = plt.figure(constrained_layout=True)
        # plot network
        ueloc = uelocs[id]
        ax = fig.add_subplot(1,1,1,projection="3d")
        visualize_network(ax, ueloc)
        for method_key, uelocs_predicted in uelocs_predicted_all.items():
            ueloc_predicted = uelocs_predicted[id]
            plot_location_in_network(ax, ueloc_predicted, PLOT_STYLES[method_key][0], method_key)
        ax.legend(prop={'size':10}, loc='upper right')
        ax.autoscale_view('tight')
        plt.show()

    print("Localization Evaluation Finished Successfully!")
