import numpy as np
from setup import *

def obtain_measured_uplink_signals(channels):
    # the pilots and sensing vectors are the same throughout
    pilots = np.load("Trained_Models/pilots.npy")
    sensing_vectors = np.load("Trained_Models/sensing_vectors.npy")
    n_networks = np.shape(channels)[0]
    assert np.shape(pilots)==(N_PILOTS,)
    assert np.shape(channels)==(n_networks, N_BS, N_BS_ANTENNAS)
    assert np.shape(sensing_vectors)==(N_BS, N_PILOTS, N_BS_ANTENNAS)
    # compute received uplink signals one BS at a time
    signals = []
    for i in range(N_BS):
        channels_oneBS = np.expand_dims(channels[:,i,:],axis=-1)
        sensing_vectors_oneBS = np.expand_dims(np.conjugate(sensing_vectors[i]),axis=0)
        signals_tmp = np.matmul(sensing_vectors_oneBS, channels_oneBS)
        assert np.shape(signals_tmp)==(n_networks, N_PILOTS, 1)
        signals_oneBS = np.squeeze(signals_tmp) * pilots
        signals.append(np.expand_dims(signals_oneBS,axis=1))
    signals = np.concatenate(signals, axis=1)
    assert np.shape(signals)==(n_networks, N_BS, N_PILOTS)
    # add in noise
    # assume uplink and downlink with same noise level
    noises = generate_circular_gaussians(size_to_generate=(n_networks, N_BS, N_PILOTS))
    noises = noises / np.abs(noises) * np.sqrt(NOISE_POWER)
    measures = signals + noises
    assert np.shape(measures)==(n_networks, N_BS, N_PILOTS)
    return measures

def visualize_network(ax, ue_loc):
    assert np.shape(ue_loc) == (3,)
    ax.set_xlim(left=0,right=FIELD_LENGTH)
    ax.set_ylim(bottom=0,top=FIELD_LENGTH)
    ax.set_zlim(bottom=0,top=FIELD_HEIGHT)
    # plot basestations
    ax.scatter3D(xs=BS_LOCATIONS[:,0], ys=BS_LOCATIONS[:,1], zs=BS_LOCATIONS[:,2], marker="1", s=100)
    # plot user equipment
    ax.scatter3D(xs=ue_loc[0], ys=ue_loc[1], zs=ue_loc[2], marker="*", s=50)
    return

def plot_location_in_network(ax, location, plot_color, plot_label):
    assert np.shape(location) == (3,)
    ax.scatter3D(xs=location[0], ys=location[1], zs=location[2], color=plot_color, marker="o", s=50, label=plot_label)
    return

def generate_circular_gaussians(size_to_generate):
    return np.random.normal(size=size_to_generate) + \
            1j * np.random.normal(size=size_to_generate)