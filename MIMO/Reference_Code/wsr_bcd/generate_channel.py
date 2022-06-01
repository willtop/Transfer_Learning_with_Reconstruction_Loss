import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random as random
from mpl_toolkits import mplot3d
import math

def generate_location(num_users):
    location_user = np.empty([num_users, 3])
    for k in range(num_users):
        #x = np.random.uniform(5, 55) 
        # y = np.random.uniform(-50, 50)
        x = np.random.uniform(5, 35) 
        y = np.random.uniform(-35, 35)
        z = -20
        coordinate_k = np.array([x, y, z])
        location_user[k, :] = coordinate_k
    return location_user

def generate_location_2D(num_users):
    location_user = np.empty([num_users, 2])
    for k in range(num_users):
        x = np.random.uniform(5, 35) 
        y = np.random.uniform(-35, 35)
        coordinate_k = np.array([x, y])
        location_user[k, :] = coordinate_k
    return location_user

def generate_location_AOA_groups(num_users):  # 32 users, 8,8,8,8
    location_user = np.empty([num_users, 3])
    group_size = int(num_users/4)
    #group1
    sigmax = 1.2 # 3,7
    sigmay = 3
    mu = (12.5,-17.5)
    for k in range(group_size):
        x = random.gauss(mu[0], sigmax)
        y = random.gauss(mu[1], sigmay)
        z = -20
        coordinate_k = np.array([x, y, z])
        location_user[k, :] = coordinate_k
    
    #group2
    mu = (27.5,-17.5)
    for k in range(group_size):
        x = random.gauss(mu[0], sigmax)
        y = random.gauss(mu[1], sigmay)
        z = -20
        coordinate_k = np.array([x, y, z])
        location_user[group_size+k, :] = coordinate_k

    #group3
    mu = (12.5,17.5)
    for k in range(group_size):
        x = random.gauss(mu[0], sigmax)
        y = random.gauss(mu[1], sigmay)
        z = -20
        coordinate_k = np.array([x, y, z])
        location_user[group_size*2+k, :] = coordinate_k

    #group4
    mu = (27.5,17.5)
    for k in range(group_size):
        x = random.gauss(mu[0], sigmax)
        y = random.gauss(mu[1], sigmay)
        z = -20
        coordinate_k = np.array([x, y, z])
        location_user[group_size*3+k, :] = coordinate_k
    

    return location_user


def path_loss_r(d):
    loss = 30 + 22.0 * np.log10(d)
    return loss


def path_loss_d(d):
    loss = 32.6 + 36.7 * np.log10(d)
    return loss


def generate_pathloss_aoa_aod(location_user, location_bs, location_irs):
    """
    :param location_user: array (num_user,2)
    :param location_bs: array (2,)
    :param location_irs: array (2,)
    :return: pathloss = (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user)
            cos_phi = (cos_phi_1, cos_phi_2, cos_phi_3)
    """
    num_user = location_user.shape[0]
    # ========bs-irs==============
    d0 = np.linalg.norm(location_bs - location_irs)
    pathloss_irs_bs = path_loss_r(d0)
    aoa_bs = ( location_irs[0] - location_bs[0]) / d0
    aod_irs_y = (location_bs[1]-location_irs[1]) / d0
    aod_irs_z = (location_bs[2]-location_irs[2]) / d0
    # =========irs-user=============
    pathloss_irs_user = []
    aoa_irs_y = []
    aoa_irs_z = []
    for k in range(num_user):
        d_k = np.linalg.norm(location_user[k] - location_irs)
        pathloss_irs_user.append(path_loss_r(d_k))
        aoa_irs_y_k = (location_user[k][1] - location_irs[1]) / d_k
        aoa_irs_z_k = (location_user[k][2] - location_irs[2]) / d_k
        aoa_irs_y.append(aoa_irs_y_k)
        aoa_irs_z.append(aoa_irs_z_k)
    aoa_irs_y = np.array(aoa_irs_y)
    aoa_irs_z = np.array(aoa_irs_z)

    # =========bs-user=============
    pathloss_bs_user = np.zeros([num_user, 1])
    for k in range(num_user):
        d_k = np.linalg.norm(location_user[k] - location_bs)
        pathloss_bs_user_k = path_loss_d(d_k)
        pathloss_bs_user[k, :] = pathloss_bs_user_k

    pathloss = (pathloss_irs_bs, np.array(pathloss_irs_user), np.array(pathloss_bs_user))
    aoa_aod = (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z)
    return pathloss, aoa_aod

def generate_pathloss_aoa_aod_MultiBS_rician(location_user, location_bs):
    """
    :param location_user: array (num_user,2)
    :param location_bs: array (2,)
    :param location_irs: array (2,)
    :return: pathloss = (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user)
            cos_phi = (cos_phi_1, cos_phi_2, cos_phi_3)
    """
    num_user = location_user.shape[0]
    # =========BS-user=============
    pathloss_BS_user = []
    aoa_BS_y = []
    for k in range(num_user):
        d_k = np.linalg.norm(location_user[k] - location_bs)
        pathloss_BS_user.append(path_loss_d(d_k))
        aoa_BS_y_k = (location_user[k][0] - location_bs[0]) / d_k
        aoa_BS_y.append(aoa_BS_y_k)
    aoa_BS_y = np.array(aoa_BS_y)

    pathloss = np.array(pathloss_BS_user)
    aoa_aod = aoa_BS_y
    return pathloss, aoa_aod

def generate_pathloss_aoa_aod_MultiBS(location_user, location_bs):
    """
    :param location_user: array (num_user,2)
    :param location_bs: array (2,)
    :param location_irs: array (2,)
    :return: pathloss = (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user)
            cos_phi = (cos_phi_1, cos_phi_2, cos_phi_3)
    """
    num_user = location_user.shape[0]
    
    # =========bs-user=============
    pathloss_bs_user = np.zeros([num_user, 1])
    for k in range(num_user):
        d_k = np.linalg.norm(location_user[k] - location_bs)
        pathloss_bs_user_k = path_loss_d(d_k)
        pathloss_bs_user[k, :] = pathloss_bs_user_k

    pathloss =  np.array(pathloss_bs_user)
    return pathloss

def generate_pathloss_aoa_aod_fullRician(location_user, location_bs, location_irs):
    """
    :param location_user: array (num_user,2)
    :param location_bs: array (2,)
    :param location_irs: array (2,)
    :return: pathloss = (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user)
            cos_phi = (cos_phi_1, cos_phi_2, cos_phi_3)
    """
    num_user = location_user.shape[0]
    # ========bs-irs==============
    d0 = np.linalg.norm(location_bs - location_irs)
    pathloss_irs_bs = path_loss_r(d0)
    aoa_bs = ( location_irs[0] - location_bs[0]) / d0
    aod_irs_y = (location_bs[1]-location_irs[1]) / d0
    aod_irs_z = (location_bs[2]-location_irs[2]) / d0
    # =========irs-user=============
    pathloss_irs_user = []
    aoa_irs_y = []
    aoa_irs_z = []
    for k in range(num_user):
        d_k = np.linalg.norm(location_user[k] - location_irs)
        pathloss_irs_user.append(path_loss_r(d_k))
        aoa_irs_y_k = (location_user[k][1] - location_irs[1]) / d_k
        aoa_irs_z_k = (location_user[k][2] - location_irs[2]) / d_k
        aoa_irs_y.append(aoa_irs_y_k)
        aoa_irs_z.append(aoa_irs_z_k)
    aoa_irs_y = np.array(aoa_irs_y)
    aoa_irs_z = np.array(aoa_irs_z)

    # =========bs-user=============
    aoa_bs_y = []
    aoa_bs_z = []
    pathloss_bs_user = np.zeros([num_user, 1])
    for k in range(num_user):
        d_k = np.linalg.norm(location_user[k] - location_bs)
        pathloss_bs_user_k = path_loss_d(d_k)
        pathloss_bs_user[k, :] = pathloss_bs_user_k
        aod_bs_k_y = (location_bs[1]-location_irs[1]) / d_k
        aod_bs_k_z = (location_bs[2]-location_irs[2]) / d_k
        aoa_bs_y.append(aod_bs_k_y)
        aoa_bs_z.append(aod_bs_k_z)
    aoa_bs_y = np.array(aoa_bs_y)
    aoa_bs_z = np.array(aoa_bs_z)

    pathloss = (pathloss_irs_bs, np.array(pathloss_irs_user), np.array(pathloss_bs_user))
    aoa_aod = (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z, aoa_bs_y , aoa_bs_z)
    return pathloss, aoa_aod

def generate_pathloss_aoa_aod_2D(location_user, location_bs, location_irs):
    """
    :param location_user: array (num_user,2)
    :param location_bs: array (2,)
    :param location_irs: array (2,)
    :return: pathloss = (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user)
            cos_phi = (cos_phi_1, cos_phi_2, cos_phi_3)
    """
    num_user = location_user.shape[0]
    # ========bs-irs==============
    d0 = np.linalg.norm(location_bs - location_irs)
    pathloss_irs_bs = path_loss_r(d0)
    aoa_bs = ( location_irs[0] - location_bs[0]) / d0
    aod_irs_y = (location_bs[1]-location_irs[1]) / d0
    #aod_irs_z = (location_bs[2]-location_irs[2]) / d0
    # =========irs-user=============
    pathloss_irs_user = []
    aoa_irs_y = []
    #aoa_irs_z = []
    for k in range(num_user):
        d_k = np.linalg.norm(location_user[k] - location_irs)
        pathloss_irs_user.append(path_loss_r(d_k))
        aoa_irs_y_k = (location_user[k][1] - location_irs[1]) / d_k
        #aoa_irs_z_k = (location_user[k][2] - location_irs[2]) / d_k
        aoa_irs_y.append(aoa_irs_y_k)
        #aoa_irs_z.append(aoa_irs_z_k)
    aoa_irs_y = np.array(aoa_irs_y)
    #aoa_irs_z = np.array(aoa_irs_z)

    # =========bs-user=============
    pathloss_bs_user = np.zeros([num_user, 1])
    for k in range(num_user):
        d_k = np.linalg.norm(location_user[k] - location_bs)
        pathloss_bs_user_k = path_loss_d(d_k)
        pathloss_bs_user[k, :] = pathloss_bs_user_k

    pathloss = (pathloss_irs_bs, np.array(pathloss_irs_user), np.array(pathloss_bs_user))
    aoa_aod = (aoa_bs, aod_irs_y, aoa_irs_y)
    return pathloss, aoa_aod

######################################################################################################################## distance not halfed
def generate_channel(params_system, location_bs=np.array([100, -100, 0]), location_irs=np.array([0, 0, 0]),
                     location_user_initial=None, Rician_factor=10, scale_factor=100, num_samples=100,irs_Nh = 16):
    # scale_factor: can be viewed as (downlink noise_power_dB- downlink Pt)

    (num_antenna_bs, num_elements_irs, num_user) = params_system

    channel_bs_irs, channel_bs_user, channel_irs_user, set_location_user = [], [], [], []
    for ii in range(num_samples):
        if location_user_initial is None:
            location_user = generate_location(num_user)
            set_location_user.append(location_user)
        else:
            location_user = location_user_initial
            set_location_user.append(location_user)
        

        pathloss, aoa_aod = generate_pathloss_aoa_aod(location_user, location_bs, location_irs)
        (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user) = pathloss
        (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z) = aoa_aod

        pathloss_bs_user = pathloss_bs_user - scale_factor
        pathloss_irs_bs = pathloss_irs_bs - scale_factor / 2
        pathloss_irs_user = pathloss_irs_user - scale_factor / 2
        pathloss_bs_user = np.sqrt(10 ** ((-pathloss_bs_user) / 10))
        pathloss_irs_user = np.sqrt(10 ** ((-pathloss_irs_user) / 10))
        pathloss_irs_bs = np.sqrt(10 ** ((-pathloss_irs_bs) / 10))

        # tmp:(num_antenna_bs,num_user) channel between BS and user
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user])
        tmp = tmp * pathloss_bs_user.reshape(1, num_user)

        # ######################################################################
        # # some user have LOS, some does not have LOS, make hd equal to 0.
        
        # user_no_los = random.sample(range(0, num_user), int(np.floor(num_user/3)))
        # tmp[:,user_no_los] = np.zeros([num_antenna_bs,1])
        # #tmp[:,user_no_los] = [0.0000000001]
        
        # ######################################################################
        channel_bs_user.append(tmp)

        # tmp: (num_antenna_bs,num_elements_irs) channel between IRS and BS
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_elements_irs]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_elements_irs])
        a_bs = np.exp(1j * np.pi * aoa_bs * np.arange(num_antenna_bs))
        a_bs = np.reshape(a_bs, [num_antenna_bs, 1])

        i1 = np.mod(np.arange(num_elements_irs),irs_Nh)
        i2 = np.floor(np.arange(num_elements_irs)/irs_Nh)
        a_irs_bs = np.exp(1j * np.pi * (i1*aod_irs_y+i2*aod_irs_z))
        a_irs_bs =  np.reshape(a_irs_bs, [num_elements_irs, 1])
        los_irs_bs = a_bs @ np.transpose(a_irs_bs.conjugate())
        tmp = np.sqrt(Rician_factor / (1 + Rician_factor)) * los_irs_bs + np.sqrt(1/(1 + Rician_factor)) * tmp
        tmp = tmp * pathloss_irs_bs
        channel_bs_irs.append(tmp)

        # tmp:(num_elements_irs,num_user) channel between IRS and user
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs, num_user])
        for k in range(num_user):
            a_irs_user = np.exp(1j * np.pi * (i1 * aoa_irs_y[k] + i2 * aoa_irs_z[k]))
            tmp[:, k] = np.sqrt(Rician_factor/(1+Rician_factor))*a_irs_user+np.sqrt(1/(1+Rician_factor))*tmp[:, k]
            tmp[:, k] = tmp[:, k] * pathloss_irs_user[k]
        channel_irs_user.append(tmp)
    channels = (np.array(channel_bs_user), np.array(channel_irs_user), np.array(channel_bs_irs))
    return channels, set_location_user

def generate_channel_fullRician(params_system, location_bs=np.array([100, -100, 0]), location_irs=np.array([0, 0, 0]),
                     location_user_initial=None, Rician_factor=10, scale_factor=100, num_samples=100,irs_Nh = 16):
    # scale_factor: can be viewed as (downlink noise_power_dB- downlink Pt)

    (num_antenna_bs, num_elements_irs, num_user) = params_system

    channel_bs_irs, channel_bs_user, channel_irs_user, set_location_user = [], [], [], []
    for ii in range(num_samples):
        if location_user_initial is None:
            location_user = generate_location(num_user)
            set_location_user.append(location_user)
        else:
            if location_user_initial.ndim >= 3: # for 2 RIS
                location_user = location_user_initial[ii,:,:]
            else:
                location_user = location_user_initial
            set_location_user.append(location_user)
        pathloss, aoa_aod = generate_pathloss_aoa_aod_fullRician(location_user, location_bs, location_irs)
        (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user) = pathloss
        (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z, aoa_bs_y, aoa_bs_z) = aoa_aod

        pathloss_bs_user = pathloss_bs_user - scale_factor
        pathloss_irs_bs = pathloss_irs_bs - scale_factor / 2
        pathloss_irs_user = pathloss_irs_user - scale_factor / 2
        pathloss_bs_user = np.sqrt(10 ** ((-pathloss_bs_user) / 10))
        pathloss_irs_user = np.sqrt(10 ** ((-pathloss_irs_user) / 10))
        pathloss_irs_bs = np.sqrt(10 ** ((-pathloss_irs_bs) / 10))

        # tmp:(num_antenna_bs,num_user) channel between BS and user
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user])
        i1 = np.arange(1)
        for k in range(num_user):
            a_bs_user = np.exp(1j * np.pi * 0)
            tmp[:, k] = np.sqrt(Rician_factor/(1+Rician_factor))*a_bs_user+np.sqrt(1/(1+Rician_factor))*tmp[:, k]
            tmp[:, k] = tmp[:, k] * pathloss_bs_user[k]
 
        channel_bs_user.append(tmp)

        #################        channel between IRS and BS
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_elements_irs]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_elements_irs])
        a_bs = np.exp(1j * np.pi * aoa_bs * np.arange(num_antenna_bs))
        a_bs = np.reshape(a_bs, [num_antenna_bs, 1])

        i1 = np.mod(np.arange(num_elements_irs),irs_Nh)
        i2 = np.floor(np.arange(num_elements_irs)/irs_Nh)
        a_irs_bs = np.exp(1j * np.pi * (i1*aod_irs_y+i2*aod_irs_z))
        a_irs_bs =  np.reshape(a_irs_bs, [num_elements_irs, 1])
        los_irs_bs = a_bs @ np.transpose(a_irs_bs.conjugate())
        tmp = np.sqrt(Rician_factor / (1 + Rician_factor)) * los_irs_bs + np.sqrt(1/(1 + Rician_factor)) * tmp
        tmp = tmp * pathloss_irs_bs
        channel_bs_irs.append(tmp)

        ###############         channel between IRS and user
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs, num_user])
        for k in range(num_user):
            a_irs_user = np.exp(1j * np.pi * (i1 * aoa_irs_y[k] + i2 * aoa_irs_z[k]))
            tmp[:, k] = np.sqrt(Rician_factor/(1+Rician_factor))*a_irs_user+np.sqrt(1/(1+Rician_factor))*tmp[:, k]
            tmp[:, k] = tmp[:, k] * pathloss_irs_user[k]
        channel_irs_user.append(tmp)
    channels = (np.array(channel_bs_user), np.array(channel_irs_user), np.array(channel_bs_irs))
    return channels, set_location_user

def generate_channel_fullRician_radiomap(params_system, location_bs=np.array([100, -100, 0]), location_irs=np.array([0, 0, 0]),
                     location_user_initial=None, Rician_factor=10, scale_factor=100, num_samples=100,irs_Nh = 16):
    # scale_factor: can be viewed as (downlink noise_power_dB- downlink Pt)

    (num_antenna_bs, num_elements_irs, num_user) = params_system

    channel_bs_irs, channel_bs_user, channel_irs_user, set_location_user = [], [], [], []
    for ii in range(num_samples):
        if location_user_initial is None:
            location_user = generate_location(num_user)
            set_location_user.append(location_user)
        else:
            location_user = location_user_initial
            set_location_user.append(location_user)
        pathloss, aoa_aod = generate_pathloss_aoa_aod_fullRician(location_user, location_bs, location_irs)
        (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user) = pathloss
        (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z, aoa_bs_y, aoa_bs_z) = aoa_aod

        pathloss_bs_user = pathloss_bs_user - scale_factor
        pathloss_irs_bs = pathloss_irs_bs - scale_factor / 2
        pathloss_irs_user = pathloss_irs_user - scale_factor / 2
        pathloss_bs_user = np.sqrt(10 ** ((-pathloss_bs_user) / 10))
        pathloss_irs_user = np.sqrt(10 ** ((-pathloss_irs_user) / 10))
        pathloss_irs_bs = np.sqrt(10 ** ((-pathloss_irs_bs) / 10))

        # tmp:(num_antenna_bs,num_user) channel between BS and user
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user])
        i1 = np.arange(1)
        for k in range(num_user):
            a_bs_user = np.exp(1j * np.pi * 0)
            tmp[:, k] = np.sqrt(Rician_factor/(1+Rician_factor))*a_bs_user+np.sqrt(1/(1+Rician_factor))*tmp[:, k]
            tmp[:, k] = tmp[:, k] * pathloss_bs_user[k]
 
        channel_bs_user.append(tmp)

        #################        channel between IRS and BS
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_elements_irs]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_elements_irs])
        a_bs = np.exp(1j * np.pi * aoa_bs * np.arange(num_antenna_bs))
        a_bs = np.reshape(a_bs, [num_antenna_bs, 1])

        i1 = np.mod(np.arange(num_elements_irs),irs_Nh)
        i2 = np.floor(np.arange(num_elements_irs)/irs_Nh)
        a_irs_bs = np.exp(1j * np.pi * (i1*aod_irs_y+i2*aod_irs_z))
        a_irs_bs =  np.reshape(a_irs_bs, [num_elements_irs, 1])
        los_irs_bs = a_bs @ np.transpose(a_irs_bs.conjugate())
        tmp = np.sqrt(Rician_factor / (1 + Rician_factor)) * los_irs_bs + np.sqrt(1/(1 + Rician_factor)) * tmp
        tmp = tmp * pathloss_irs_bs
        channel_bs_irs.append(tmp)

        ###############         channel between IRS and user
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs, num_user])
        for k in range(num_user):
            a_irs_user = np.exp(1j * np.pi * (i1 * aoa_irs_y[k] + i2 * aoa_irs_z[k]))
            tmp[:, k] = np.sqrt(Rician_factor/(1+Rician_factor))*a_irs_user+np.sqrt(1/(1+Rician_factor))*tmp[:, k]
            tmp[:, k] = tmp[:, k] * pathloss_irs_user[k]
        channel_irs_user.append(tmp)
    channels = (np.array(channel_bs_user), np.array(channel_irs_user), np.array(channel_bs_irs))
    return channels, set_location_user

def generate_channel_MultiBS_rician(params_system, location_bs1,location_bs2,location_bs3,
                     location_user_initial=None, Rician_factor=10, scale_factor=100, num_samples=100,irs_Nh = 16):
    # scale_factor: can be viewed as (downlink noise_power_dB- downlink Pt)

    (num_antenna_bs, num_elements_irs, num_user) = params_system

    channel_bs1_user,channel_bs2_user,channel_bs3_user, set_location_user = [], [], [], []
    for ii in range(num_samples):
        if location_user_initial is None:
            location_user = generate_location(num_user)
            set_location_user.append(location_user)
        else:
            location_user = location_user_initial
            set_location_user.append(location_user)

        pathloss_bs1_user, aoa_aod = generate_pathloss_aoa_aod_MultiBS_rician(location_user, location_bs1)
        aoa_BS1 = aoa_aod
        pathloss_bs2_user, aoa_aod = generate_pathloss_aoa_aod_MultiBS_rician(location_user, location_bs2)
        aoa_BS2 = aoa_aod
        pathloss_bs3_user, aoa_aod = generate_pathloss_aoa_aod_MultiBS_rician(location_user, location_bs3)
        aoa_BS3 = aoa_aod

        pathloss_bs1_user = pathloss_bs1_user - scale_factor
        pathloss_bs1_user = np.sqrt(10 ** ((-pathloss_bs1_user) / 10))
        pathloss_bs2_user = pathloss_bs2_user - scale_factor
        pathloss_bs2_user = np.sqrt(10 ** ((-pathloss_bs2_user) / 10))
        pathloss_bs3_user = pathloss_bs3_user - scale_factor
        pathloss_bs3_user = np.sqrt(10 ** ((-pathloss_bs3_user) / 10))

        # tmp:(num_elements_irs,num_user) channel between IRS and user
        i1 = np.arange(num_antenna_bs)
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user])
        for k in range(num_user):
            a_BS1_user = np.exp(1j * np.pi * (i1 * aoa_BS1[k]))
            tmp[:, k] = np.sqrt(Rician_factor/(1+Rician_factor))*a_BS1_user+np.sqrt(1/(1+Rician_factor))*tmp[:, k]
            tmp[:, k] = tmp[:, k] * pathloss_bs1_user[k]
        channel_bs1_user.append(tmp)

        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user])
        for k in range(num_user):
            a_BS2_user = np.exp(1j * np.pi * (i1 * aoa_BS2[k]))
            tmp[:, k] = np.sqrt(Rician_factor/(1+Rician_factor))*a_BS2_user+np.sqrt(1/(1+Rician_factor))*tmp[:, k]
            tmp[:, k] = tmp[:, k] * pathloss_bs2_user[k]
        channel_bs2_user.append(tmp)

        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user])
        for k in range(num_user):
            a_BS3_user = np.exp(1j * np.pi * (i1 * aoa_BS3[k] ))
            tmp[:, k] = np.sqrt(Rician_factor/(1+Rician_factor))*a_BS3_user+np.sqrt(1/(1+Rician_factor))*tmp[:, k]
            tmp[:, k] = tmp[:, k] * pathloss_bs3_user[k]
        channel_bs3_user.append(tmp)

    channels = (np.array(channel_bs1_user), np.array(channel_bs2_user), np.array(channel_bs3_user))
    return channels, set_location_user


def generate_channel_MultiBS_1_plus_2BS_rician(params_system, location_bs1,location_bs2,location_bs3,
                     location_user_initial=None, Rician_factor=10, scale_factor=100, num_samples=100,irs_Nh = 16):
    # scale_factor: can be viewed as (downlink noise_power_dB- downlink Pt)

    (N_main, N_secondary, num_user) = params_system

    channel_bs1_user,channel_bs2_user,channel_bs3_user, set_location_user = [], [], [], []
    for ii in range(num_samples):
        if location_user_initial is None:
            location_user = generate_location(num_user)
            set_location_user.append(location_user)
        else:
            location_user = location_user_initial
            set_location_user.append(location_user)

        pathloss_bs1_user, aoa_aod = generate_pathloss_aoa_aod_MultiBS_rician(location_user, location_bs1)
        aoa_BS1 = aoa_aod
        pathloss_bs2_user, aoa_aod = generate_pathloss_aoa_aod_MultiBS_rician(location_user, location_bs2)
        aoa_BS2 = aoa_aod
        pathloss_bs3_user, aoa_aod = generate_pathloss_aoa_aod_MultiBS_rician(location_user, location_bs3)
        aoa_BS3 = aoa_aod

        pathloss_bs1_user = pathloss_bs1_user - scale_factor
        pathloss_bs1_user = np.sqrt(10 ** ((-pathloss_bs1_user) / 10))
        pathloss_bs2_user = pathloss_bs2_user - scale_factor
        pathloss_bs2_user = np.sqrt(10 ** ((-pathloss_bs2_user) / 10))
        pathloss_bs3_user = pathloss_bs3_user - scale_factor
        pathloss_bs3_user = np.sqrt(10 ** ((-pathloss_bs3_user) / 10))

        # tmp:(num_elements_irs,num_user) channel between IRS and user
        i1 = np.arange(N_main)
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[N_main, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[N_main, num_user])
        for k in range(num_user):
            a_BS1_user = np.exp(1j * np.pi * 0)
            tmp[:, k] = np.sqrt(Rician_factor/(1+Rician_factor))*a_BS1_user+np.sqrt(1/(1+Rician_factor))*tmp[:, k]
            tmp[:, k] = tmp[:, k] * pathloss_bs1_user[k]
        channel_bs1_user.append(tmp)

        i1 = np.arange(N_secondary)
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[N_secondary, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[N_secondary, num_user])
        for k in range(num_user):
            a_BS2_user = np.exp(1j * np.pi * (i1 * aoa_BS2[k]))
            tmp[:, k] = np.sqrt(Rician_factor/(1+Rician_factor))*a_BS2_user+np.sqrt(1/(1+Rician_factor))*tmp[:, k]
            tmp[:, k] = tmp[:, k] * pathloss_bs2_user[k]
        channel_bs2_user.append(tmp)

        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[N_secondary, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[N_secondary, num_user])
        for k in range(num_user):
            a_BS3_user = np.exp(1j * np.pi * (i1 * aoa_BS3[k] ))
            tmp[:, k] = np.sqrt(Rician_factor/(1+Rician_factor))*a_BS3_user+np.sqrt(1/(1+Rician_factor))*tmp[:, k]
            tmp[:, k] = tmp[:, k] * pathloss_bs3_user[k]
        channel_bs3_user.append(tmp)

    channels = (np.array(channel_bs1_user), np.array(channel_bs2_user), np.array(channel_bs3_user))
    return channels, set_location_user


def generate_channel_MultiBS_main_secondary_rician(params_system, location_bs_main,location_bs2_secondary=np.array([0, 0, 0]),
                     location_user_initial=None, Rician_factor=10, scale_factor=100, num_samples=100,irs_Nh = 16):
    # scale_factor: can be viewed as (downlink noise_power_dB- downlink Pt)

    (N_main, N_secondary, num_user) = params_system

    channel_bs1_user,channel_bs2_user, set_location_user = [], [], []
    for ii in range(num_samples):
        if location_user_initial is None:
            location_user = generate_location(num_user)
            set_location_user.append(location_user)
        else:
            location_user = location_user_initial
            set_location_user.append(location_user)

        pathloss_bs1_user, aoa_aod = generate_pathloss_aoa_aod_MultiBS_rician(location_user, location_bs_main)
        aoa_BS1 = aoa_aod
        pathloss_bs2_user, aoa_aod = generate_pathloss_aoa_aod_MultiBS_rician(location_user, location_bs2_secondary)
        aoa_BS2 = aoa_aod

        pathloss_bs1_user = pathloss_bs1_user - scale_factor
        pathloss_bs1_user = np.sqrt(10 ** ((-pathloss_bs1_user) / 10))
        pathloss_bs2_user = pathloss_bs2_user - scale_factor
        pathloss_bs2_user = np.sqrt(10 ** ((-pathloss_bs2_user) / 10))

        # tmp:(num_elements_irs,num_user) channel between IRS and user
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[N_main, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[N_main, num_user])
        for k in range(num_user):
            a_BS1_user = np.exp(1j * np.pi * 0)
            tmp[:, k] = np.sqrt(Rician_factor/(1+Rician_factor))*a_BS1_user+np.sqrt(1/(1+Rician_factor))*tmp[:, k]
            tmp[:, k] = tmp[:, k] * pathloss_bs1_user[k]
        channel_bs1_user.append(tmp)

        i1 = np.arange(N_secondary)
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[N_secondary, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[N_secondary, num_user])
        for k in range(num_user):
            a_BS2_user = np.exp(1j * np.pi * (i1 * aoa_BS2[k]))
            tmp[:, k] = np.sqrt(Rician_factor/(1+Rician_factor))*a_BS2_user+np.sqrt(1/(1+Rician_factor))*tmp[:, k]
            tmp[:, k] = tmp[:, k] * pathloss_bs2_user[k]
        channel_bs2_user.append(tmp)

    channels = (np.array(channel_bs1_user), np.array(channel_bs2_user))
    return channels, set_location_user

def generate_channel_singleBS_rician(params_system, location_bs=np.array([100, -100, 0]),
                     location_user_initial=None, Rician_factor=10, scale_factor=100, num_samples=100,irs_Nh = 16):
    # scale_factor: can be viewed as (downlink noise_power_dB- downlink Pt)

    (num_antenna_bs, num_elements_irs, num_user) = params_system

    channel_bs1_user,channel_bs2_user,channel_bs3_user, set_location_user = [], [], [], []
    for ii in range(num_samples):
        if location_user_initial is None:
            location_user = generate_location(num_user)
            set_location_user.append(location_user)
        else:
            location_user = location_user_initial
            set_location_user.append(location_user)

        pathloss_bs1_user, aoa_aod = generate_pathloss_aoa_aod_MultiBS_rician(location_user, location_bs)
        (aoa_BS1_y, aoa_BS1_z) = aoa_aod

        pathloss_bs1_user = pathloss_bs1_user - scale_factor
        pathloss_bs1_user = np.sqrt(10 ** ((-pathloss_bs1_user) / 10))

        # tmp:(num_elements_irs,num_user) channel between IRS and user
        i1 = np.arange(num_antenna_bs)
        i2 = np.arange(1)     
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user])
        for k in range(num_user):
            a_BS1_user = np.exp(1j * np.pi * (i1 * aoa_BS1_y[k] + i2 * aoa_BS1_z[k]))
            tmp[:, k] = np.sqrt(Rician_factor/(1+Rician_factor))*a_BS1_user+np.sqrt(1/(1+Rician_factor))*tmp[:, k]
            tmp[:, k] = tmp[:, k] * pathloss_bs1_user[k]
        channel_bs1_user.append(tmp)

    return np.array(channel_bs1_user), set_location_user

def generate_channel_MultiBS(params_system, location_bs1,location_bs2,location_bs3,
                        location_user_initial=None, Rician_factor=10, scale_factor=100, num_samples=100):
    # scale_factor: can be viewed as (downlink noise_power_dB- downlink Pt)

    (num_antenna_bs, num_elements_irs, num_user) = params_system

    channel_bs1_user,channel_bs2_user,channel_bs3_user, set_location_user = [],[],[], []
    for ii in range(num_samples):
        if location_user_initial is None:
            location_user = generate_location(num_user)
            set_location_user.append(location_user)
        else:
            location_user = location_user_initial
            set_location_user.append(location_user)
        
        # ######### BS 1
        pathloss_bs_user = generate_pathloss_aoa_aod_MultiBS(location_user, location_bs1)
        pathloss_bs_user = pathloss_bs_user - scale_factor
        pathloss_bs_user = np.sqrt(10 ** ((-pathloss_bs_user) / 10))
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user])
        tmp = tmp * pathloss_bs_user.reshape(1, num_user)
        channel_bs1_user.append(tmp)

        # ######### BS 2
        pathloss_bs_user = generate_pathloss_aoa_aod_MultiBS(location_user, location_bs2)
        pathloss_bs_user = pathloss_bs_user - scale_factor
        pathloss_bs_user = np.sqrt(10 ** ((-pathloss_bs_user) / 10))
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user])
        tmp = tmp * pathloss_bs_user.reshape(1, num_user)
        channel_bs2_user.append(tmp)

        # ######### BS 3
        pathloss_bs_user = generate_pathloss_aoa_aod_MultiBS(location_user, location_bs3)
        pathloss_bs_user = pathloss_bs_user - scale_factor
        pathloss_bs_user = np.sqrt(10 ** ((-pathloss_bs_user) / 10))
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user])
        tmp = tmp * pathloss_bs_user.reshape(1, num_user)
        channel_bs3_user.append(tmp)

    channels = (np.array(channel_bs1_user),np.array(channel_bs2_user) ,np.array(channel_bs3_user))
    return channels, set_location_user

def generate_channel_debug(params_system, location_bs=np.array([100, -100, 0]), location_irs=np.array([0, 0, 0]),
                     location_user_initial=None, Rician_factor=10, scale_factor=100, num_samples=100,irs_Nh = 16):
    # scale_factor: can be viewed as (downlink noise_power_dB- downlink Pt)

    (num_antenna_bs, num_elements_irs, num_user) = params_system

    channel_bs_irs, channel_bs_user, channel_irs_user, set_location_user = [], [], [], []
    for ii in range(num_samples):
        if location_user_initial is None:
            location_user = generate_location(num_user)
            set_location_user.append(location_user)
        else:
            location_user = location_user_initial
            set_location_user.append(location_user)

        pathloss, aoa_aod = generate_pathloss_aoa_aod(location_user, location_bs, location_irs)
        (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user) = pathloss
        (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z) = aoa_aod

        pathloss_bs_user = pathloss_bs_user - scale_factor
        pathloss_irs_bs = pathloss_irs_bs - scale_factor / 2
        pathloss_irs_user = pathloss_irs_user - scale_factor / 2
        pathloss_bs_user = np.sqrt(10 ** ((-pathloss_bs_user) / 10))
        pathloss_irs_user = np.sqrt(10 ** ((-pathloss_irs_user) / 10))
        pathloss_irs_bs = np.sqrt(10 ** ((-pathloss_irs_bs) / 10))

        # tmp:(num_antenna_bs,num_user) channel between BS and user
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user])
        tmp = tmp * pathloss_bs_user.reshape(1, num_user)

        channel_bs_user.append(tmp)

        # tmp: (num_antenna_bs,num_elements_irs) channel between IRS and BS
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_elements_irs]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_elements_irs])
        a_bs = np.exp(1j * np.pi * aoa_bs * np.arange(num_antenna_bs))
        a_bs = np.reshape(a_bs, [num_antenna_bs, 1])

        i1 = np.mod(np.arange(num_elements_irs),irs_Nh)
        i2 = np.floor(np.arange(num_elements_irs)/irs_Nh)
        a_irs_bs = np.exp(1j * np.pi * (i1*aod_irs_y+i2*aod_irs_z))
        a_irs_bs =  np.reshape(a_irs_bs, [num_elements_irs, 1])
        los_irs_bs = a_bs @ np.transpose(a_irs_bs.conjugate())
        tmp = np.sqrt(Rician_factor / (1 + Rician_factor)) * los_irs_bs + np.sqrt(1/(1 + Rician_factor)) * tmp
        tmp = tmp * pathloss_irs_bs
        channel_bs_irs.append(tmp)

        # tmp:(num_elements_irs,num_user) channel between IRS and user
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs, num_user])
        for k in range(num_user):
            a_irs_user = np.exp(1j * np.pi * (i1 * aoa_irs_y[k] + i2 * aoa_irs_z[k]))
            tmp[:, k] = np.sqrt(Rician_factor/(1+Rician_factor))*a_irs_user+np.sqrt(1/(1+Rician_factor))*tmp[:, k]
            tmp[:, k] = tmp[:, k] * pathloss_irs_user[k]
        channel_irs_user.append(tmp)
    channels = (np.array(channel_bs_user), np.array(channel_irs_user), np.array(channel_bs_irs))
    print(aoa_irs_z)
    print(aoa_irs_y)
    angle_ele = math.asin(aoa_irs_z)
    angle_az = math.asin(aoa_irs_y / math.cos(angle_ele))
    angles = (angle_ele , angle_az)
    return channels, set_location_user, angles

def generate_channel_2D(params_system, location_bs=np.array([100, -100]), location_irs=np.array([0, 0]),
                     location_user_initial=None, Rician_factor=10, scale_factor=100, num_samples=100,irs_Nh = 16):

    (num_antenna_bs, num_elements_irs, num_user) = params_system

    channel_bs_irs, channel_bs_user, channel_irs_user, set_location_user = [], [], [], []
    aoa_irs_y_set = []
    pathloss_irs_user_set = []
    for ii in range(num_samples):
        if location_user_initial is None:
            location_user = generate_location_2D(num_user)
            set_location_user.append(location_user)
        else:
            location_user = location_user_initial
            set_location_user.append(location_user)

        pathloss, aoa_aod = generate_pathloss_aoa_aod_2D(location_user, location_bs, location_irs)
        (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user) = pathloss
        (aoa_bs, aod_irs_y, aoa_irs_y) = aoa_aod

        pathloss_bs_user = pathloss_bs_user - scale_factor
        pathloss_irs_bs = pathloss_irs_bs - scale_factor / 2
        pathloss_irs_user = pathloss_irs_user - scale_factor / 2
        pathloss_bs_user = np.sqrt(10 ** ((-pathloss_bs_user) / 10))
        pathloss_irs_user = np.sqrt(10 ** ((-pathloss_irs_user) / 10))
        pathloss_irs_bs = np.sqrt(10 ** ((-pathloss_irs_bs) / 10))

        # tmp:(num_antenna_bs,num_user) channel between BS and user
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user])
        tmp = tmp * pathloss_bs_user.reshape(1, num_user)

        channel_bs_user.append(tmp)

        # tmp: (num_antenna_bs,num_elements_irs) channel between IRS and BS
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_elements_irs]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_elements_irs])
        a_bs = np.exp(1j * np.pi * aoa_bs * np.arange(num_antenna_bs))
        a_bs = np.reshape(a_bs, [num_antenna_bs, 1])

        i1 = np.arange(num_elements_irs)
        i2 = np.floor(np.arange(num_elements_irs)/irs_Nh)
        #a_irs_bs = np.exp(1j * np.pi * (i1*aod_irs_y+i2*aod_irs_z))
        a_irs_bs = np.exp(1j * np.pi * (i1 *aod_irs_y))
        a_irs_bs =  np.reshape(a_irs_bs, [num_elements_irs, 1])
        los_irs_bs = a_bs @ np.transpose(a_irs_bs.conjugate())
        tmp = np.sqrt(Rician_factor / (1 + Rician_factor)) * los_irs_bs + np.sqrt(1/(1 + Rician_factor)) * tmp
        tmp = tmp * pathloss_irs_bs
        channel_bs_irs.append(tmp)

        # tmp:(num_elements_irs,num_user) channel between IRS and user
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs, num_user])
        for k in range(num_user):
            #a_irs_user = np.exp(1j * np.pi * (i1 * aoa_irs_y[k] + i2 * aoa_irs_z[k]))
            a_irs_user = np.exp(1j * np.pi * (i1 * aoa_irs_y[k]))
            if num_user ==1:
                aoa_irs_y_set.append(aoa_irs_y[k])
                pathloss_irs_user_set.append(pathloss_irs_user)
            else:
                raise RuntimeError('number user greater than 1 !')
            
            tmp[:, k] = np.sqrt(Rician_factor/(1+Rician_factor))*a_irs_user+np.sqrt(1/(1+Rician_factor))*tmp[:, k]
            tmp[:, k] = tmp[:, k] * pathloss_irs_user[k]
        channel_irs_user.append(tmp)
    channels = (np.array(channel_bs_user), np.array(channel_irs_user), np.array(channel_bs_irs))
    return channels, np.array(set_location_user), np.array(aoa_irs_y_set), np.array(pathloss_irs_user_set)



def channel_complex2real(channels):
    channel_bs_user, channel_irs_user, channel_bs_irs = channels
    (num_sample, num_antenna_bs, num_elements_irs) = channel_bs_irs.shape
    num_user = channel_irs_user.shape[2]

    A_T_real = np.zeros([num_sample, 2 * num_elements_irs, 2 * num_antenna_bs, num_user])
    # Hd_real = np.zeros([num_sample, 2 * num_antenna_bs, num_user])
    set_channel_combine_irs = np.zeros([num_sample, num_antenna_bs, num_elements_irs, num_user], dtype=complex)

    for kk in range(num_user):
        channel_irs_user_k = channel_irs_user[:, :, kk]
        channel_combine_irs = channel_bs_irs * channel_irs_user_k.reshape(num_sample, 1, num_elements_irs)
        set_channel_combine_irs[:, :, :, kk] = channel_combine_irs
        A_tmp_tran = np.transpose(channel_combine_irs, (0, 2, 1))
        A_tmp_real1 = np.concatenate([A_tmp_tran.real, A_tmp_tran.imag], axis=2)
        A_tmp_real2 = np.concatenate([-A_tmp_tran.imag, A_tmp_tran.real], axis=2)
        A_tmp_real = np.concatenate([A_tmp_real1, A_tmp_real2], axis=1)
        A_T_real[:, :, :, kk] = A_tmp_real

    Hd_real = np.concatenate([channel_bs_user.real, channel_bs_user.imag], axis=1)

    return A_T_real, Hd_real, np.array(set_channel_combine_irs)


def main():
    num_test = 1000
    num_antenna_bs, num_elements_irs, num_user = 1, 64, 1
    params_system = (num_antenna_bs, num_elements_irs, num_user)
    Rician_factor = 10
    location_user = None

    channel_true, set_location_user = generate_channel_2D(params_system,Rician_factor=Rician_factor,
                                                       num_samples=num_test)
    _, _, channel_bs_irs_user = channel_complex2real(channel_true)

    print('channel_bs_user:\n',np.mean(np.abs(channel_true[0])**2))
    print('channel_irs_user:\n',np.mean(np.abs(channel_true[1])**2))
    print('channel_bs_irs:\n',np.mean(np.abs(channel_true[2])**2))
    print('channel_bs_irs_user:\n',np.mean(np.abs(channel_bs_irs_user)**2))


if __name__ == '__main__':
    main()
