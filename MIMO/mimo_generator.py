import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import itertools
import utils
from setup import *

CHECK_SETTING = False
GENERATE_DATA_SOURCETASK = True
GENERATE_DATA_TARGETTASK = True
GENERATE_DATA_TEST = True

def compute_pathloss(dists):
    pathlosses_tmp = 32.6+36.7*np.log(dists)
    pathlosses = np.power(10, -pathlosses_tmp/10)
    return pathlosses

def get_channels_to_BS(ue_locs, BS_ID):
    n_networks = np.shape(ue_locs)[0]
    # get coordinate differences between UE of all networks and three fixed-location BSs
    ue_bs_coord_diffs = ue_locs - BS_LOCATIONS[BS_ID] 
    ue_bs_dists = np.linalg.norm(ue_bs_coord_diffs, axis=1)
    # let theta denote the azimuth angle, and phi denote the elevation angle
    # assume 2D BS antenna array lining in x-y plane with negligible antenna length
    # the singal travel difference for the steering angle would be:
    # along the x-axis:
    # antenna_spacing X sin(theta) X cos(phi) = antenna_spacing X (x_UE-x_BS)/distance_UE->BS
    sin_theta_cos_phi = ue_bs_coord_diffs[:,0]/ue_bs_dists
    # along the y-axis:
    # antenna_spacing X cos(theta) X cos(phi) = antenna_spacing X (y_UE-y_BS)/distance_UE->BS
    cos_theta_cos_phi = ue_bs_coord_diffs[:,1]/ue_bs_dists
    # incorporate antenna indexing: 
    # flatten all antenna indexing into one vector with row-major:
    # each row index antennas with same x-coordinate, with first row indexing highest x-coordinate
    # in each row, antenna to the right most (highest y coordinate) is indexed the first
    steer_vec_tmp = np.concatenate([np.expand_dims(sin_theta_cos_phi,axis=1),
                                    np.expand_dims(cos_theta_cos_phi, axis=1)],axis=1)
    indices = np.transpose(np.array(list(itertools.product(np.arange(np.sqrt(N_BS_ANTENNAS)), repeat=2))))
    steer_vec_tmp = np.matmul(steer_vec_tmp, indices)
    assert np.shape(steer_vec_tmp) == (n_networks, N_BS_ANTENNAS)
    channels_LOS = np.exp(1j*ANTENNA_SPACING_PHASE_SHIFT*steer_vec_tmp)  
    channels_NLOS = np.random.normal(size=(n_networks, N_BS_ANTENNAS)) + 1j * np.random.normal(size=(n_networks, N_BS_ANTENNAS))
    channels_fading = np.sqrt(RICIAN_FACTOR/(1+RICIAN_FACTOR))*channels_LOS + \
                        np.sqrt(1/(1+RICIAN_FACTOR))*channels_NLOS
    # compute path losses
    pathlosses = compute_pathloss(np.tile(np.expand_dims(ue_bs_dists, axis=1), (1, N_BS_ANTENNAS)))
    assert np.shape(pathlosses)==np.shape(channels_fading)==(n_networks, N_BS_ANTENNAS)
    channels_one_BS = pathlosses*channels_fading
    # collect factors: one tuple from each BS (distance, sin_theta_cos_phi, cos_theta_cos_phi)
    factors_one_BS = np.concatenate([np.expand_dims(ue_bs_dists, axis=1), \
                                     np.expand_dims(sin_theta_cos_phi, axis=1), \
                                     np.expand_dims(cos_theta_cos_phi, axis=1)], axis=1)
    assert np.shape(channels_one_BS) == (n_networks, N_BS_ANTENNAS) and \
            np.shape(factors_one_BS) == (n_networks, N_FACTORS)
    return channels_one_BS, factors_one_BS

def generate_MIMO_networks(n_networks):
    # generate user location in 3D space
    # one UE in each network
    ue_locs = np.concatenate([np.random.uniform(low=0, high=FIELD_LENGTH, size=(n_networks, 2)), \
                             np.random.uniform(low=0, high=FIELD_HEIGHT, size=(n_networks, 1))], axis=1) 
    assert np.shape(ue_locs) == (n_networks, 3)
    # compute channels from the UEs to each of the BSs
    channels, factors = [], []
    for i in range(N_BS):
        channels_one_BS, factors_one_BS = get_channels_to_BS(ue_locs, i)
        channels.append(np.expand_dims(channels_one_BS,axis=1))
        factors.append(np.expand_dims(factors_one_BS,axis=1))
    channels, factors = np.concatenate(channels, axis=1), np.concatenate(factors, axis=1)
    assert np.shape(channels) == (n_networks, N_BS, N_BS_ANTENNAS) and \
            np.shape(factors) == (n_networks, N_BS, N_FACTORS)
    return ue_locs, channels, factors


if __name__=="__main__":
    if CHECK_SETTING:
        ue_locs, channels, factors = generate_MIMO_networks(4)
        # visualize the location and channels generated
        for i in range(4):
            fig = plt.figure()
            ax = fig.add_subplot(2,1,1,projection="3d")
            utils.visualize_network(ax, ue_locs[i])
            ax = fig.add_subplot(2,1,2)
            for j in range(N_BS):
                ax.plot(np.arange(1, N_BS_ANTENNAS+1), np.power(np.abs(channels[i][j]),2), label=f'BS_{j+1}')
            ax.legend()
            plt.show()
            print("factors of all UEs: ", factors[i].flatten())
        exit(0)

    if GENERATE_DATA_SOURCETASK:
        print(f"Generate data for {SOURCETASK['Type']} {SOURCETASK['Task']} training, including statistics for input normalization......")
        ue_locs, channels, factors = generate_MIMO_networks(SOURCETASK['Train']+SOURCETASK['Valid'])
        np.save("Data/uelocs_sourcetask.npy", ue_locs)
        np.save("Data/channels_sourcetask.npy", channels)
        np.save("Data/factors_sourcetask.npy", factors)
        # Use the source-task train data for input normalization stats
        np.save("Trained_Models/Channels_Stats/channels_train_mean.npy", np.mean(channels[:SOURCETASK['Train']], axis=0))
        np.save("Trained_Models/Channels_Stats/channels_train_std.npy", np.std(channels[:SOURCETASK['Train']], axis=0))
        # Randomly generate uplink pilots used by the user-equipment (same across all networks)
        pilots = utils.generate_circular_gaussians(size_to_generate=(N_PILOTS,))
        # normalize each pilot to unit power
        pilots = pilots / np.abs(pilots)
        np.save("Trained_Models/pilots.npy", pilots)
        # Randomly generate uplink pilots sensing vectors for all base-stations
        sensing_vectors = utils.generate_circular_gaussians(size_to_generate=(N_BS, N_PILOTS, N_BS_ANTENNAS))
        # normalize the sensing vectors to be uniform power within each base-station
        sensing_vectors = sensing_vectors / np.linalg.norm(sensing_vectors, axis=-1, keepdims=True)
        np.save("Trained_Models/sensing_vectors.npy", sensing_vectors)

    if GENERATE_DATA_TARGETTASK:
        print(f"Generate data for {TARGETTASK['Type']} {TARGETTASK['Task']} training......")
        ue_locs, channels, factors = generate_MIMO_networks(TARGETTASK['Train']+TARGETTASK['Valid'])
        np.save("Data/uelocs_targettask.npy", ue_locs)
        np.save("Data/channels_targettask.npy", channels)
        np.save("Data/factors_targettask.npy", factors)

    if GENERATE_DATA_TEST:
        print(f"Generate data for testing......")
        # No need to keep track of factors during testing
        ue_locs, channels, _ = generate_MIMO_networks(N_TEST_SAMPLES)
        np.save("Data/uelocs_test.npy", ue_locs)
        np.save("Data/channels_test.npy", channels)

    print("Script Completed!")