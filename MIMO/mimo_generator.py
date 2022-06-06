import numpy as np
from setup import *

def get_channels_to_BS(ue_locs, BS_ID):
    n_networks = np.shape(ue_locs)[0]
    # get coordinate differences between UE of all networks and three fixed-location BSs
    ue_bs_coord_diffs = ue_locs - BS_LOCATIONS[BS_ID] 
    # let theta denote the azimuth angle, and phi denote the elevation angle
    # assume linear array of BS antennas lining in the direction of (0,1,0)
    # assume all BS antennas point towards (0,0,1) direction with negligible length
    # the single travel difference for the steering angle would be:
    # antenna_spacing X cos(theta) X cos(phi) = antenna_spacing X (y_UE-y_BS)/distance_UE->BS
    ue_bs_dists = np.linalg.norm(ue_bs_coord_diffs, axis=1)
    steer_vec_tmp = ue_bs_coord_diffs[:,1]/ue_bs_dists
    # incorporate antenna indexing: antenna to the right-most (highest y coordinate) is indexed the first
    steer_vec_tmp = np.arange(N_BS_ANTENNAS)*np.expand_dims(steer_vec_tmp,axis=1)
    channels_LOS = np.exp(ANTENNA_SPACING_PHASE_SHIFT*steer_vec_tmp)  
    assert np.shape(channels_LOS) == (n_networks, N_BS_ANTENNAS)
    channels_NLOS = np.random.normal(size=(n_networks, N_BS_ANTENNAS)) + 1j * np.random.normal(size=(n_networks, N_BS_ANTENNAS))
    

def generate_MIMO_networks(n_networks):
    # generate user location in 3D space
    # one UE in each network
    ue_locs = np.concatenate([np.random.uniform(low=0, high=FIELD_LENGTH, size=(n_networks, 2)), \
                             np.random.uniform(low=0, high=FIELD_HEIGHT, size=(n_networks, 1))], axis=1) 
    assert np.shape(ue_locs) == (n_networks, 3)
    # compute channels from the UEs to each of the BSs
    for i in range(N_BS):
        get_channels_to_BS(ue_locs, i)
    
    return ue_locs

if __name__=="__main__":

