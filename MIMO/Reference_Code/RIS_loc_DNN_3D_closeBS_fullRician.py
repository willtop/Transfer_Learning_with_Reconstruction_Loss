import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import matplotlib.pyplot as plt
from func_codedesign_cont import func_codedesign_cont
import scipy.io as sio
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense
from wsr_bcd.generate_channel import generate_channel_fullRician, channel_complex2real
from util_func import random_beamforming
import time

#from wsr.bcd.generate_received_pilots import generate_received_pilots_batch

'System Information'
N = 1   #Number of BS's antennas
delta_inv = 128 #Number of posterior intervals inputed to DNN 
delta = 1/delta_inv 
S = np.log2(delta_inv) 
OS_rate = 20 #Over sampling rate in each AoA interval (N_s in the paper)
delta_inv_OS = OS_rate*delta_inv #Total number of AoAs for posterior computation
delta_OS = 1/delta_inv_OS 
'Channel Information'

location_bs_new = np.array([40,-40,0])


tau =  6 #[2, 5, 8, 11 ,14 ]
snr_const = 35  #The SNR
snr_const = np.array([snr_const])
Pvec = 10**(snr_const/10) #Set of considered TX powers

mean_true_alpha = 0.0 + 0.0j #Mean of the fading coefficient
std_per_dim_alpha = np.sqrt(0.5) #STD of the Gaussian fading coefficient per real dim.
noiseSTD_per_dim = np.sqrt(0.5) #STD of the Gaussian noise per real dim.
#####################################################
'RIS'
N_ris = 64
num_users = 1
params_system = (N,N_ris,num_users)
Rician_factor = 10
location_user = None

#####################################################
'Learning Parameters'
initial_run = 1 #0: Continue training; 1: Starts from the scratch
n_epochs = 500 #Num of epochs
learning_rate = 0.0005 #Learning rate
batch_per_epoch = 100 #Number of mini batches per epoch
batch_size_order = 8 #Mini_batch_size = batch_size_order*delta_inv
val_size_order = 782 #Validation_set_size = val_size_order*delta_inv
scale_factor = 1 #Scaling the number of tests
test_size_order = 782 #Test_set_size = test_size_order*delta_inv*scale_factor
######################################################
tf.reset_default_graph() #Reseting the graph
he_init = tf.variance_scaling_initializer() #Define initialization method
######################################## Place Holders
loc_input = tf.placeholder(tf.float32, shape=(None,1,3), name="loc_input")
channel_bs_irs_user = tf.placeholder(tf.float32, shape=(None, 2 * N_ris, 2 * N, num_users), name="channel_bs_irs_user")
channel_bs_user = tf.placeholder(tf.float32, shape=(None, 2 * N, num_users), name="channel_bs_user")
theta_T = tf.placeholder(tf.float32, shape=(None, tau, 2 * N_ris), name="theta_T")
######################################################

path_pilots = './loc/DNN_fixSNR/theta_training_tau_'+ str(tau) +'_'+'SNR'+str(snr_const[0])+'.mat'
if initial_run == 0: # continue training
    data_loadout = sio.loadmat(path_pilots)
    the_theta = data_loadout['the_theta']
else:   # training from scratch
    _, the_theta = random_beamforming(tau, N , N_ris, num_users)
    sio.savemat(path_pilots, {'the_theta': the_theta})
the_theta = np.concatenate([the_theta.real, the_theta.imag], axis=1)
the_theta = np.reshape(the_theta, [-1, tau, 2 * N_ris]) 

##################### NETWORK
with tf.name_scope("array_response_construction"):
    lay = {}
    lay['P'] = tf.constant(1.0)
    ###############

with tf.name_scope("channel_sensing"):

    w_dict = []
    posterior_dict = []
    idx_est_dict = []

    A_T_k = channel_bs_irs_user[:, :, :, 0] # since 1 user
    
    for t in range(tau): 
        theta_T_tau = theta_T[:,t:t+1,:]    # ? , 1, 128
        theta_A_k_T = tf.matmul(theta_T_tau, A_T_k)                             # (? , 1 , 2 * N ) 

        h_d = channel_bs_user[:,:,0]
        h_d_T = tf.reshape(h_d, [-1, 1 , 2*N])

        h_d_plus_h_cas = h_d_T + theta_A_k_T
        h_d_plus_h_cas_re = h_d_plus_h_cas[:,:,0]
        h_d_plus_h_cas_im = h_d_plus_h_cas[:,:,1]

        noise =  tf.complex(tf.random_normal(tf.shape(h_d_plus_h_cas_re), mean = 0.0, stddev = noiseSTD_per_dim),\
                    tf.random_normal(tf.shape(h_d_plus_h_cas_re), mean = 0.0, stddev = noiseSTD_per_dim))
        y_complex = tf.complex(tf.sqrt(lay['P']),0.0)*tf.complex(h_d_plus_h_cas_re,h_d_plus_h_cas_im) + noise
        y_real = tf.concat([tf.real(y_complex),tf.imag(y_complex)],axis=1)/tf.sqrt(lay['P'])
        if t == 0:
            input_ = y_real
        else:
            input_ = tf.concat([input_, y_real],axis = 1)

    x0 = input_

    x0 = Dense(units=200, activation='relu')(x0)
    x0 = BatchNormalization()(x0)

    x0 = Dense(units=200, activation='relu')(x0)
    x0 = BatchNormalization()(x0)

    x0 = Dense(units=200, activation='relu')(x0)
    x0 = BatchNormalization()(x0)

    loc_hat = Dense(units=3, activation='linear')(x0)  
    
####################################################################################
####### Loss Function
a = tf.math.reduce_euclidean_norm(loc_input[:,0,:]-loc_input[:,0,:], 1)
b = tf.math.reduce_euclidean_norm(loc_hat-loc_input[:,0,:], 1)
loss = tf.keras.losses.mean_squared_error(a,b)
####### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()
#########################################################################
###########  Validation Set
channel_true_val, set_location_user_val = generate_channel_fullRician(params_system, location_bs = location_bs_new,
                                                        num_samples=val_size_order*delta_inv,
                                                       location_user_initial=location_user, Rician_factor=Rician_factor)
(channel_bs_user_val, channel_irs_user_val, channel_bs_irs_val) = channel_true_val
A_T_real_val, Hd_real_val, channel_bs_irs_user_val = channel_complex2real(channel_true_val)
feed_dict_val = {loc_input: np.array(set_location_user_val),
                    channel_bs_irs_user: A_T_real_val,
                    channel_bs_user: Hd_real_val,
                    theta_T: the_theta,
                    lay['P']: Pvec[0]}
###########  Training
with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, './loc/DNN_fixSNR/params_DNN_loc_RIS_closeBS_fullRician_3D_tau_'+ str(tau) +'_snr'+str(snr_const[0])+'_rician_' + str(Rician_factor))
    best_loss, pp = sess.run([loss,posterior_dict], feed_dict=feed_dict_val)
    print(best_loss)
    print(tf.test.is_gpu_available()) #Prints whether or not GPU is on
    for epoch in range(n_epochs):
        batch_iter = 0
        for rnd_indices in range(batch_per_epoch):
            
            P_temp = 10**(snr_const[0]/10)  #
            channel_true_train, set_location_user_train = generate_channel_fullRician(params_system, location_bs = location_bs_new,
                                                        num_samples=batch_size_order*delta_inv,
                                                       location_user_initial=location_user, Rician_factor=Rician_factor)
            A_T_real, Hd_real_train, _ = channel_complex2real(channel_true_train)
            #the_theta_train =  np.repeat(the_theta[np.newaxis,:, :], batch_size_order*delta_inv, axis= 0 )
            
            feed_dict_batch = {loc_input: np.array(set_location_user_train),
                              channel_bs_irs_user: A_T_real,
                              channel_bs_user: Hd_real_train, 
                              theta_T: the_theta,
                              lay['P']: P_temp}
            sess.run(training_op, feed_dict=feed_dict_batch)
            batch_iter += 1
        
        loss_val = sess.run(loss, feed_dict=feed_dict_val)
        print('epoch',epoch,'  loss_test:%2.5f'%loss_val,'  best_test:%2.5f'%best_loss) 
        if epoch%5==1: #Every 5 iterations it checks if the validation performace is improved, then saves parameters
            if loss_val < best_loss:
                save_path = saver.save(sess, './loc/DNN_fixSNR/params_DNN_loc_RIS_closeBS_fullRician_3D_tau_'+ str(tau) +'_snr'+str(snr_const[0])+'_rician_' + str(Rician_factor))
                best_loss = loss_val

    ###########  Final Test    
    performance = np.zeros([1,scale_factor])
    for j in range(scale_factor):
        print(j)
        channel_true_test, set_location_user_test = generate_channel_fullRician(params_system, location_bs = location_bs_new,
                                                        num_samples=test_size_order*delta_inv,
                                                       location_user_initial=location_user, Rician_factor=Rician_factor)
        (channel_bs_user_test, channel_irs_user_test, channel_bs_irs_test) = channel_true_test
        A_T_real_test, Hd_real_test, channel_bs_irs_user_test = channel_complex2real(channel_true_test)
        
        feed_dict_test = {loc_input: np.array(set_location_user_test),
                            channel_bs_irs_user: A_T_real_test,
                            channel_bs_user: Hd_real_test, 
                            theta_T: the_theta,
                            lay['P']: Pvec[0]}

        mse_loss,loc_hat_test= sess.run([loss,loc_hat],feed_dict=feed_dict_test)
        performance[0,j] = mse_loss
            
    performance = np.mean(performance,axis=1)       
            
######### Plot the test result 
plt.semilogy(snr_const, performance)    
plt.grid()
plt.xlabel('SNR (dB)')
plt.ylabel('Average MSE')
plt.show()
sio.savemat('./loc/DNN_fixSNR/data_DNN_loc_RIS_new_closeBS_fullRician_3D_tau_'+ str(tau) +'_snr'+str(snr_const[0]) +'_rician_' + str(Rician_factor) +'.mat',dict(performance= performance,\
                                       snr_const=snr_const,N=N,N_ris = N_ris,epoch = n_epochs,delta_inv=delta_inv,\
                                       mean_true_alpha=mean_true_alpha,\
                                       n_epochs = n_epochs, \
                                       std_per_dim_alpha=std_per_dim_alpha,\
                                       noiseSTD_per_dim=noiseSTD_per_dim, tau=tau))
