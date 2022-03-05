#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import set_up#our codes
import methods
from warnings import simplefilter
simplefilter(action='ignore')# ignore all warnings
smoothness, jump, exploration, randomness = 5, 0.25, 0.6, 0.6
setting = 'setting1'
policy_type,demand_type,logging_type,feature_dist = [setting,randomness],[setting,smoothness,jump],[setting,exploration],'uniform'
feature_dim,train_size,rndseed,sigma = 2,30,int(1000000*(time.time()%1)),0.2#bandwidth of kernels

t_start=time.time()
Gamma=1

Train_data = set_up.prepare_data(rndseed,train_size,policy_type,demand_type,logging_type,feature_dist,feature_dim,sigma)
linear_mse,linear_bias,linear_var,hat_r_z_arr,hat_r_y_arr = methods.simulate_linear_MSE(Train_data,3,1)

hat_r_z,hat_r_y = methods.truncate_r(Train_data,hat_r_z_arr[0],hat_r_y_arr[0])
weights = methods.optimize_bound_method(Train_data,Gamma,[],hat_r_z,hat_r_y,epsilon=0.05)
wc_mse,grad,wc_bias,wc_var = methods.optimize_bound_subroutine(weights,0.05,Train_data,
                                     Gamma,hat_r_z,hat_r_y,returnDetails=1)
print('grad: ',grad)
print('wc bias: ',wc_bias,'wc var: ',wc_var)
mse,bias,var = set_up.true_MSE(weights,Train_data) #this is not honest mse, because hat_r differs for different demand realizations

print('opti bound method: ',mse,bias,var)
print('linear reg: ',linear_mse,linear_bias,linear_var)
w_NW,_ = methods.kernel_regression_oracle(Train_data)

##mse,bias,var = methods.our_method_DR_simulate_MSE(Train_data,Gamma,hat_r_z_arr,hat_r_y_arr,method='opti bound')
##print('opti bound method: ',mse,bias,var)
##time_elapsed = int(round(time.time()-t_start))
##print('total runtime: ',time_elapsed)


