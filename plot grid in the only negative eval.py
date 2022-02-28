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
Gamma=5

#plot subproblem grid search
Train_data = set_up.prepare_data(rndseed,train_size,policy_type,demand_type,logging_type,feature_dist,feature_dim,sigma)
_,_,_,hat_r_z_arr,hat_r_y_arr = methods.simulate_linear_MSE(Train_data,3,1)
hat_r_z,hat_r_y = methods.truncate_r(Train_data,hat_r_z_arr[0],hat_r_y_arr[0])
w1,obj,time1 = methods.our_method_DR_quadratic_cut(Train_data,Gamma,hat_r_z,hat_r_y,brute_force=0,returnDetails=1)

obj_arr,obj_surrogate_arr,evals = methods.subroutine_DR_brute_force(w1,Gamma,Train_data,hat_r_z, hat_r_y,4,solver='Gurobi')
grid = np.arange(-Gamma*(2**0.5),Gamma*(2**0.5),0.1)

from matplotlib import pyplot
pyplot.plot(grid, obj_arr,label='obj')
pyplot.plot(grid, obj_surrogate_arr,label='obj approx')
title='subproblem objective in grid search'
pyplot.title(title)
pyplot.xlabel('x1')
pyplot.ylabel('subproblem objective')
pyplot.legend(loc='best')
pyplot.show()
