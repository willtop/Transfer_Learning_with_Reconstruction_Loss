import numpy as np
from setup import *
from tqdm import trange

# Generate layout one at a time
def generate_one_D2D_layout():
    # first, generate transmitters' coordinates
    tx_xs = np.random.uniform(low=0, high=FIELD_LENGTH, size=[N_LINKS,1])
    tx_ys = np.random.uniform(low=0, high=FIELD_LENGTH, size=[N_LINKS,1])
    while(True): # loop until a valid layout generated
        # generate rx one by one rather than N together to ensure checking validity one by one
        rx_xs = []; rx_ys = []
        for i in range(N_LINKS):
            got_valid_rx = False
            while(not got_valid_rx):
                pair_dist = np.random.uniform(low=SHORTEST_DIRECTLINK, high=LONGEST_DIRECTLINK)
                pair_angles = np.random.uniform(low=0, high=np.pi*2)
                rx_x = tx_xs[i] + pair_dist * np.cos(pair_angles)
                rx_y = tx_ys[i] + pair_dist * np.sin(pair_angles)
                if(0<=rx_x<=FIELD_LENGTH and 0<=rx_y<=FIELD_LENGTH):
                    got_valid_rx = True
            rx_xs.append(rx_x); rx_ys.append(rx_y)
        # For now, assuming equal weights and equal power, so not generating them
        layout = np.concatenate((tx_xs, tx_ys, rx_xs, rx_ys), axis=1)
        distances = np.zeros([N_LINKS, N_LINKS])
        # compute distance between every possible Tx/Rx pair
        for rx_index in range(N_LINKS):
            for tx_index in range(N_LINKS):
                tx_coor = layout[tx_index][0:2]
                rx_coor = layout[rx_index][2:4]
                # according to paper notation convention, Hij is from jth transmitter to ith receiver
                distances[rx_index][tx_index] = np.linalg.norm(tx_coor - rx_coor)
        if(np.min(distances+np.eye(N_LINKS)*SHORTEST_CROSSLINK)<SHORTEST_CROSSLINK):
            pass
        else:
            break # go ahead and return the layout
    return layout, distances

def generate_D2D_layouts(n_layouts):
    print("<<<<<<<<<<<<<{} layouts: {}>>>>>>>>>>>>".format(n_layouts, SETTING_STRING))
    layouts_all, distances_all = [], []
    for i in trange(n_layouts):
        layouts, distances = generate_one_D2D_layout()
        layouts_all.append(layouts)
        distances_all.append(distances)
    layouts_all, distances_all = np.array(layouts_all), np.array(distances_all)
    assert np.shape(layouts_all)==(n_layouts, N_LINKS, 4)
    assert np.shape(distances_all)==(n_layouts, N_LINKS, N_LINKS)
    return layouts_all, distances_all

def generate_D2D_channelGains(n_layouts):
    layouts, distances = generate_D2D_layouts(n_layouts)
    assert np.shape(distances) == (n_layouts, N_LINKS, N_LINKS)
    ############ Path Losses #############
    h1, h2 = TX_HEIGHT, RX_HEIGHT
    signal_lambda = 2.998e8 / CARRIER_FREQUENCY
    # compute relevant quantity
    Rbp = 4 * h1 * h2 / signal_lambda
    Lbp = abs(20 * np.log10(np.power(signal_lambda, 2) / (8 * np.pi * h1 * h2)))
    # compute coefficient matrix for each Tx/Rx pair
    sum_term = 20 * np.log10(distances / Rbp)
    Tx_over_Rx = Lbp + 6 + sum_term + ((distances > Rbp).astype(int)) * sum_term  # adjust for longer path loss
    pathLosses = -Tx_over_Rx + np.eye(N_LINKS) * ANTENNA_GAIN_DB  
    pathLosses = np.power(10, (pathLosses / 10))  # convert from decibel to absolute
    ############# Shadowing and Fast Fading ##########
    # generate shadowing coefficients
    shadowing = np.random.normal(size=np.shape(pathLosses), loc=0, scale=8.0)
    shadowing = np.power(10.0, shadowing / 10.0)
    # generate fast fading factors with circular Gaussian
    ff_real = np.random.normal(size=np.shape(pathLosses))
    ff_imag = np.random.normal(size=np.shape(pathLosses))
    ff_realizations = (np.power(ff_real, 2) + np.power(ff_imag, 2)) / 2
    channelGains = pathLosses * shadowing * ff_realizations    
    return channelGains, layouts # Shape: n_layouts X N X N, n_layouts X N

def get_directLink_channels(channels):
    assert channels.ndim==3
    return np.diagonal(channels, axis1=1, axis2=2)

def get_crossLink_channels(channels):
    return channels*(1.0-np.identity(N_LINKS, dtype=float))

def compute_SINRs(pc, channels):
    dl = get_directLink_channels(channels)
    cl = get_crossLink_channels(channels)
    signals = pc * dl * TX_POWER # layouts X N
    interferences = np.squeeze(np.matmul(cl, np.expand_dims(pc, axis=-1))) * TX_POWER # layouts X N
    sinrs = signals / ((interferences + NOISE_POWER)*SINR_GAP)   # layouts X N
    return sinrs

def compute_rates(sinrs):
    return BANDWIDTH * np.log2(1 + sinrs) 



