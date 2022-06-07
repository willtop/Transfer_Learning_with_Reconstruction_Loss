import numpy as np
import itertools
from setup import *

def get_channels_to_BS(ue_locs, BS_ID):
    n_networks = np.shape(ue_locs)[0]
    # get coordinate differences between UE of all networks and three fixed-location BSs
    ue_bs_coord_diffs = ue_locs - BS_LOCATIONS[BS_ID] 
    # let theta denote the azimuth angle, and phi denote the elevation angle
    # assume 2D BS antenna array lining in x-y plane with negligible antenna length
    # the singal travel difference for the steering angle would be:
    # along the y-axis:
    # antenna_spacing X cos(theta) X cos(phi) = antenna_spacing X (y_UE-y_BS)/distance_UE->BS
    # along the x-axis:
    # antenna_spacing X sin(theta) X cos(phi) = antenna_spacing X (x_UE-x_BS)/distance_UE->BS
    ue_bs_dists = np.linalg.norm(ue_bs_coord_diffs, axis=1)
    steer_vec_tmp_x = ue_bs_coord_diffs[:,0]/ue_bs_dists
    steer_vec_tmp_y = ue_bs_coord_diffs[:,1]/ue_bs_dists
    # incorporate antenna indexing: 
    # flatten all antenna indexing into one vector with row-major:
    # each row index antennas with same x-coordinate, with first row indexing highest x-coordinate
    # in each row, antenna to the right most (highest y coordinate) is indexed the first
    steer_vec_tmp = np.concatenate([np.expand_dims(steer_vec_tmp_x,axis=1),
                                    np.expand_dims(steer_vec_tmp_y, axis=1)],axis=1)
    indices = np.transpose(np.array(list(itertools.product(np.arange(np.sqrt(N_BS_ANTENNAS)), repeat=2))))
    steer_vec_tmp = np.matmul(steer_vec_tmp, indices)
    assert np.shape(steer_vec_tmp) == (n_networks, N_BS_ANTENNAS)
    channels_LOS = np.exp(ANTENNA_SPACING_PHASE_SHIFT*steer_vec_tmp)  
    channels_NLOS = np.random.normal(size=(n_networks, N_BS_ANTENNAS)) + 1j * np.random.normal(size=(n_networks, N_BS_ANTENNAS))
    channels_fading = np.sqrt(RICIAN_FACTOR/(1+RICIAN_FACTOR))*channels_LOS + \
                        np.sqrt(1/(1+RICIAN_FACTOR))*channels_NLOS
    # compute path losses
    ue_bs_dists_tmp = np.tile(np.expand_dims(ue_bs_dists, axis=1), (1, N_BS_ANTENNAS))
    pathlosses = 32.6 + 36.7*np.log(ue_bs_dists_tmp)
    assert np.shape(pathlosses)==np.shape(channels_fading)==(n_networks, N_BS_ANTENNAS)
    channels_one_BS = pathlosses*channels_fading
    return channels_one_BS

def generate_MIMO_networks(n_networks):
    # generate user location in 3D space
    # one UE in each network
    ue_locs = np.concatenate([np.random.uniform(low=0, high=FIELD_LENGTH, size=(n_networks, 2)), \
                             np.random.uniform(low=0, high=FIELD_HEIGHT, size=(n_networks, 1))], axis=1) 
    assert np.shape(ue_locs) == (n_networks, 3)
    # compute channels from the UEs to each of the BSs
    channels = []
    for i in range(N_BS):
        channels_one_BS = get_channels_to_BS(ue_locs, i)
        assert np.shape(channels_one_BS)==(n_networks, N_BS_ANTENNAS)
        channels.append(np.expand_dims(channels_one_BS,axis=1))
    channels = np.concatenate(channels, axis=1)
    assert np.shape(n_networks, N_BS, N_BS_ANTENNAS)
    return ue_locs, channels

if __name__=="__main__":
    
